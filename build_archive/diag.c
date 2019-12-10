/******************************************************************************
 *                                                                            *
 * DIAG.C                                                                     *
 *                                                                            *
 * DIAGNOSTIC OUTPUT                                                          *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

// Evaluate flux based diagnostics; put results in global variables
// Note this is still per-process
void diag_flux(struct FluidFlux *F)
{
  mdot = edot = ldot = 0.;
  mdot_eh = edot_eh = ldot_eh = 0.;
  int iEH = NG + 5;
  if (global_start[0] == 0) {
#if !INTEL_WORKAROUND
#pragma omp parallel for \
  reduction(+:mdot) reduction(+:edot) reduction(+:ldot) \
  reduction(+:mdot_eh) reduction(+:edot_eh) reduction(+:ldot_eh) \
  collapse(2)
#endif
    JSLOOP(0, N2 - 1) {
      KSLOOP(0, N3 - 1) {
        mdot += -F->X1[RHO][k][j][NG]*dx[2]*dx[3];
        edot += (F->X1[UU][k][j][NG] - F->X1[RHO][k][j][NG])*dx[2]*dx[3];
        ldot += F->X1[U3][k][j][NG]*dx[2]*dx[3];
        mdot_eh += -F->X1[RHO][k][j][iEH]*dx[2]*dx[3];
        edot_eh += (F->X1[UU][k][j][iEH] - F->X1[RHO][k][j][iEH])*dx[2]*dx[3];
        ldot_eh += F->X1[U3][k][j][iEH]*dx[2]*dx[3];
      }
    }
  }
}

void diag(struct GridGeom *G, struct FluidState *S, int call_code)
{
  static FILE *ener_file;

  if (call_code == DIAG_INIT) {
    // Set things up
    if(mpi_io_proc()) {
      ener_file = fopen("dumps/log.out", "a");
      if (ener_file == NULL) {
        fprintf(stderr, "Error opening log file!\n");
        exit(1);
      }
    }
  }

  double pp = 0.;
  double divbmax = 0.;
  double rmed = 0.;
  double e = 0.;
  // Calculate conserved quantities
  if (call_code == DIAG_INIT || call_code == DIAG_LOG ||
       call_code == DIAG_FINAL) {

    get_state_vec(G, S, CENT, 0, N3 - 1, 0, N2 - 1, 0, N1 - 1);
    prim_to_flux_vec(G, S, 0, CENT, 0, N3 - 1, 0, N2 - 1, 0, N1 - 1, S->U);
#if !INTEL_WORKAROUND
#pragma omp parallel for \
  reduction(+:rmed) reduction(+:pp) reduction(+:e) \
  reduction(max:divbmax) collapse(3)
#endif
    ZLOOP {
      rmed += S->U[RHO][k][j][i]*dV;
      pp += S->U[U3][k][j][i]*dV;
      e += S->U[UU][k][j][i]*dV;

      double divb = flux_ct_divb(G, S, i, j, k);

      if (divb > divbmax) {
        divbmax = divb;
      }
    }
  }

  rmed = mpi_reduce(rmed);
  pp = mpi_reduce(pp);
  e = mpi_reduce(e);
  divbmax = mpi_max(divbmax);

  double mass_proc = 0.;
  double egas_proc = 0.;
  double Phi_proc = 0.;
  double jet_EM_flux_proc = 0.;
  double lum_eht_proc = 0.;
#if !INTEL_WORKAROUND
#pragma omp parallel for \
  reduction(+:mass_proc) reduction(+:egas_proc) reduction(+:Phi_proc) \
  reduction(+:jet_EM_flux_proc) reduction(+:lum_eht_proc) \
  collapse(3)
#endif
  ZLOOP {
    mass_proc += S->U[RHO][k][j][i]*dV;
    egas_proc += S->U[UU][k][j][i]*dV;
    double rho = S->P[RHO][k][j][i];
    double Pg = (gam - 1.)*S->P[UU][k][j][i];
    double bsq = bsq_calc(S, i, j, k);
    double Bmag = sqrt(bsq);
    double C_eht = 0.2;
    double j_eht = pow(rho,3.)*pow(Pg,-2.)*exp(-C_eht*pow(rho*rho/(Bmag*Pg*Pg),1./3.));
    lum_eht_proc += j_eht*dV*G->gdet[CENT][j][i];
    if (global_start[0] + i == 5+NG) {
      // This is as close as I can figure to Narayan '12, after accounting for HARM B_unit
      Phi_proc += 0.5*fabs(sqrt(4*M_PI)*S->P[B1][k][j][i])*dx[2]*dx[3]*G->gdet[CENT][j][i];

      // TODO port properly.  Needs speculative get_state
//      double P_EM[NVAR];
//      PLOOP P_EM[ip] = S->P[ip][k][j][i];
//      P_EM[RHO] = 0.;
//      P_EM[UU] = 0.;
//      get_state(P_EM, &(ggeom[i][j][CENT]), &q);
//      double sig = bsq/S->P[RHO][k][j][i];
//      if (sig > 1.) {
//        primtoflux(P_EM, &q, 1, &(ggeom[i][j][CENT]), U);
//        jet_EM_flux_proc += -U[U1]*dx[2]*dx[3];
//      }
    }
  }
  double mass = mpi_reduce(mass_proc);
  double egas = mpi_reduce(egas_proc);
  double Phi = mpi_reduce(Phi_proc);
  double jet_EM_flux = mpi_reduce(jet_EM_flux_proc);
  double lum_eht = mpi_reduce(lum_eht_proc);

  if ((call_code == DIAG_INIT && !is_restart) ||
    call_code == DIAG_DUMP || call_code == DIAG_FINAL) {
    dump(G, S);
    dump_cnt++;
  }

  if (call_code == DIAG_INIT || call_code == DIAG_LOG ||
      call_code == DIAG_FINAL) {
    double mdot_all = mpi_reduce(mdot);
    double edot_all = mpi_reduce(edot);
    double ldot_all = mpi_reduce(ldot);
    double mdot_eh_all = mpi_reduce(mdot_eh);
    double edot_eh_all = mpi_reduce(edot_eh);
    double ldot_eh_all = mpi_reduce(ldot_eh);

    //mdot will be negative w/scheme above
    double phi = Phi/sqrt(fabs(mdot_all) + SMALL);

    if(mpi_io_proc()) {
      fprintf(stdout, "LOG      t=%g \t divbmax: %g\n",
        t,divbmax);
      fprintf(ener_file, "%10.5g %10.5g %10.5g %10.5g %15.8g %15.8g ",
        t, rmed, pp, e,
        S->P[UU][N3/2][N2/2][N1/2]*pow(S->P[RHO][N3/2][N2/2][N1/2], -gam),
        S->P[UU][N3/2][N2/2][N1/2]);
      fprintf(ener_file, "%15.8g %15.8g %15.8g ", mdot_all, edot_all, ldot_all);
      fprintf(ener_file, "%15.8g %15.8g ", mass, egas);
      fprintf(ener_file, "%15.8g %15.8g %15.8g ", Phi, phi, jet_EM_flux);
      fprintf(ener_file, "%15.8g ", divbmax);
      fprintf(ener_file, "%15.8g ", lum_eht);
      fprintf(ener_file, "%15.8g %15.8g %15.8g ", mdot_eh_all, edot_eh_all, ldot_eh_all);
      fprintf(ener_file, "\n");
      fflush(ener_file);
    }
  }
}

// Diagnostic routines
double flux_ct_divb(struct GridGeom *G, struct FluidState *S, int i, int j, int k)
{
  #if N3 > 1
  if(i > 0 + NG && j > 0 + NG && k > 0 + NG &&
     i < N1 + NG && j < N2 + NG && k < N3 + NG) {
  #elif N2 > 1
  if(i > 0 + NG && j > 0 + NG &&
     i < N1 + NG && j < N2 + NG) {
  #elif N1 > 1
  if(i > 0 + NG &&
     i < N1 + NG) {
  #else
  if (0) {
  #endif
    return fabs(0.25*(
      S->P[B1][k][j][i]*G->gdet[CENT][j][i]
      + S->P[B1][k][j-1][i]*G->gdet[CENT][j-1][i]
      + S->P[B1][k-1][j][i]*G->gdet[CENT][j][i]
      + S->P[B1][k-1][j-1][i]*G->gdet[CENT][j-1][i]
      - S->P[B1][k][j][i-1]*G->gdet[CENT][j][i-1]
      - S->P[B1][k][j-1][i-1]*G->gdet[CENT][j-1][i-1]
      - S->P[B1][k-1][j][i-1]*G->gdet[CENT][j][i-1]
      - S->P[B1][k-1][j-1][i-1]*G->gdet[CENT][j-1][i-1]
      )/dx[1] +
      0.25*(
      S->P[B2][k][j][i]*G->gdet[CENT][j][i]
      + S->P[B2][k][j][i-1]*G->gdet[CENT][j][i-1]
      + S->P[B2][k-1][j][i]*G->gdet[CENT][j][i]
      + S->P[B2][k-1][j][i-1]*G->gdet[CENT][j][i-1]
      - S->P[B2][k][j-1][i]*G->gdet[CENT][j-1][i]
      - S->P[B2][k][j-1][i-1]*G->gdet[CENT][j-1][i-1]
      - S->P[B2][k-1][j-1][i]*G->gdet[CENT][j-1][i]
      - S->P[B2][k-1][j-1][i-1]*G->gdet[CENT][j-1][i-1]
      )/dx[2] +
      0.25*(
      S->P[B3][k][j][i]*G->gdet[CENT][j][i]
      + S->P[B3][k][j-1][i]*G->gdet[CENT][j-1][i]
      + S->P[B3][k][j][i-1]*G->gdet[CENT][j][i-1]
      + S->P[B3][k][j-1][i-1]*G->gdet[CENT][j-1][i-1]
      - S->P[B3][k-1][j][i]*G->gdet[CENT][j][i]
      - S->P[B3][k-1][j-1][i]*G->gdet[CENT][j-1][i]
      - S->P[B3][k-1][j][i-1]*G->gdet[CENT][j][i-1]
      - S->P[B3][k-1][j-1][i-1]*G->gdet[CENT][j-1][i-1]
      )/dx[3]);
  } else {
    return 0.;
  }
}

// Some quick one-off functions for debugging
#if DEBUG

int i_have(int iglobal, int jglobal, int kglobal)
{
  int have_i = (global_start[0] <= iglobal) && (global_stop[0] > iglobal);
  int have_j = (global_start[1] <= jglobal) && (global_stop[1] > jglobal);
  int have_k = (global_start[2] <= kglobal) && (global_stop[2] > kglobal);
  return have_i && have_j && have_k;
}

void global_map(int iglobal, int jglobal, int kglobal, GridPrim prim)
{
  if(i_have(iglobal, jglobal, kglobal)){
    area_map(iglobal-global_start[0]+NG, jglobal-global_start[1]+NG, kglobal-global_start[2]+NG, prim);
    area_map_pflag(iglobal-global_start[0]+NG, jglobal-global_start[1]+NG, kglobal-global_start[2]+NG);
  }
}

double sigma_max (struct GridGeom *G, struct FluidState *S)
{

  double sigma_max = 0;
#if !INTEL_WORKAROUND
#pragma omp parallel for simd collapse(3) reduction(max:sigma_max)
#endif
  ZLOOP {
    get_state(G, S, i, j, k, CENT);
    double bsq = bsq_calc(S, i, j, k);

    if (bsq/S->P[RHO][k][j][i] > sigma_max)
      sigma_max = bsq/S->P[RHO][k][j][i];
  }

  return sigma_max;
}

// Map out region around failure point (local index)
void area_map(int i, int j, int k, GridPrim prim)
{
  PLOOP {
    fprintf(stderr, "variable %d \n", ip);
    fprintf(stderr, "%12.5g %12.5g %12.5g\n",
      prim[ip][k][j + 1][i - 1], prim[ip][k][j + 1][i],
      prim[ip][k][j + 1][i + 1]);
    fprintf(stderr, "%12.5g %12.5g %12.5g\n",
      prim[ip][k][j][i - 1], prim[ip][k][j][i],
      prim[ip][k][j][i + 1]);
    fprintf(stderr, "%12.5g %12.5g %12.5g\n",
      prim[ip][k][j - 1][i - 1], prim[ip][k][j - 1][i],
      prim[ip][k][j - 1][i + 1]);
  }
}

void area_map_pflag(int i, int j, int k)
{
  fprintf(stderr, "pflag: \n");
  fprintf(stderr, "%d %d %d\n",
    pflag[k][j + 1][i - 1], pflag[k][j + 1][i],
    pflag[k][j + 1][i + 1]);
  fprintf(stderr, "%d %d %d\n",
    pflag[k][j][i - 1], pflag[k][j][i],
    pflag[k][j][i + 1]);
  fprintf(stderr, "%d %d %d\n",
    pflag[k][j - 1][i - 1], pflag[k][j - 1][i],
    pflag[k][j - 1][i + 1]);
}

// TODO this function is useful but slow: it doesn't parllelize under intel 18.0.2
// Check the whole fluid state for NaN values
inline void check_nan(struct FluidState *S, const char* flag)
{
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(3)
#endif
  PLOOP ZLOOPALL {
    if (isnan(S->P[ip][k][j][i])) {
      fprintf(stderr, "NaN in prims[%d]: %d, %d, %d as of position %s\n",ip,i,j,k,flag);
      exit(-1);
    }
  }
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(3)
#endif
  DLOOP1 ZLOOPALL {
    if (isnan(S->ucon[mu][k][j][i]) || isnan(S->ucov[mu][k][j][i]) || isnan(S->bcon[mu][k][j][i]) || isnan(S->bcov[mu][k][j][i])){
      fprintf(stderr, "NaN in derived: %d, %d, %d at %s\n",i,j,k,flag);
      exit(-1);
    }
  }
}

inline void update_f(struct FluidFlux *F, GridPrim *dU)
{
#pragma omp parallel for simd collapse(3)
  PLOOP ZLOOP {
    preserve_F.X1[ip][k][j][i] = F->X1[ip][k][j][i];
    preserve_F.X2[ip][k][j][i] = F->X2[ip][k][j][i];
    preserve_F.X3[ip][k][j][i] = F->X3[ip][k][j][i];
    preserve_dU[ip][k][j][i] = (*dU)[ip][k][j][i];
  }
}

// TODO reintroduce this? Just fails the fluid on certain conditions
#if 0
void check_fluid(struct GridGeom *G, struct FluidState *S)
{
#pragma omp parallel for collapse(3)
  ZLOOP {
    // Find fast magnetosonic speed
    double bsq = bsq_calc(S, i, j, k);
    double rho = fabs(S->P[RHO][k][j][i]);
    double u = fabs(S->P[UU][k][j][i]);
    double ef = rho + gam*u;
    double ee = bsq + ef;
    double va2 = bsq/ee;
    double cs2 = gam*(gam - 1.)*u/ef;

    double cms2 = cs2 + va2 - cs2*va2;

    // Sanity checks
    if (cms2 < 0.) {
      fprintf(stderr, "\n\ncms2: %g %g %g\n\n", gam, u, ef);
      fail(G, S, FAIL_COEFF_NEG, i, j, k);
    }
    if (cms2 > 1.) {
      fail(G, S, FAIL_COEFF_SUP, i, j, k);
    }

    // This one requires a lot of context Acon/cov Bcon/cov -> A,B,C -> discr
    if ((discr < 0.0) && (discr > -1.e-10)) {
      discr = 0.0;
    } else if (discr < -1.e-10) {
      fprintf(stderr, "\n\t %g %g %g %g %g\n", A, B, C, discr, cms2);
      fprintf(stderr, "\n\t S->ucon: %g %g %g %g\n", S->ucon[0][k][j][i],
        S->ucon[1][k][j][i], S->ucon[2][k][j][i], S->ucon[3][k][j][i]);
      fprintf(stderr, "\n\t S->bcon: %g %g %g %g\n", S->bcon[0][k][j][i],
        S->bcon[1][k][j][i], S->bcon[2][k][j][i], S->bcon[3][k][j][i]);
      fprintf(stderr, "\n\t Acon: %g %g %g %g\n", Acon[0], Acon[1], Acon[2],
        Acon[3]);
      fprintf(stderr, "\n\t Bcon: %g %g %g %g\n", Bcon[0], Bcon[1], Bcon[2],
        Bcon[3]);
      fail(G, S, FAIL_VCHAR_DISCR, i, j, k);
      discr = 0.;
    }
  }
}
#endif


#endif // DEBUG
