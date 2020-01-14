/******************************************************************************
 *                                                                            *
 * FIXUP.C                                                                    *
 *                                                                            *
 * REPAIR INTEGRATION FAILURES                                                *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

// Floor Codes: bit masks
#define HIT_FLOOR_GEOM_RHO 1
#define HIT_FLOOR_GEOM_U 2
#define HIT_FLOOR_B_RHO 4
#define HIT_FLOOR_B_U 8
#define HIT_FLOOR_TEMP 16
#define HIT_FLOOR_GAMMA 32
#define HIT_FLOOR_KTOT 64

// Point in m, around which to steepen floor prescription, eventually toward r^-3
#define FLOOR_R_CHAR 10

static struct FluidState *Stmp;

void fixup_ceiling(struct GridGeom *G, struct FluidState *S, int i, int j, int k);
void fixup_floor(struct GridGeom *G, struct FluidState *S, int i, int j, int k);

// Apply floors to density, internal energy
void fixup(struct GridGeom *G, struct FluidState *S)
{
  timer_start(TIMER_FIXUP);

  static int firstc = 1;
  if (firstc) {Stmp = calloc(1,sizeof(struct FluidState)); firstc = 0;}

#pragma omp parallel for simd collapse(2)
  ZLOOPALL fflag[k][j][i] = 0;

#pragma omp parallel for collapse(3)
  ZLOOP fixup_ceiling(G, S, i, j, k);

  // Bulk call before bsq calculation below
  get_state_vec(G, S, CENT, 0, N3-1, 0, N2-1, 0, N1-1);

#pragma omp parallel for collapse(3)
  ZLOOP fixup_floor(G, S, i, j, k);

  // Some debug info about floors
#if DEBUG
  int n_geom_rho = 0, n_geom_u = 0, n_b_rho = 0, n_b_u = 0, n_temp = 0, n_gamma = 0, n_ktot = 0;

#pragma omp parallel for collapse(3) reduction(+:n_geom_rho) reduction(+:n_geom_u) \
    reduction(+:n_b_rho) reduction(+:n_b_u) reduction(+:n_temp) reduction(+:n_gamma) reduction(+:n_ktot)
  ZLOOP {
    int flag = fflag[k][j][i];
    if (flag & HIT_FLOOR_GEOM_RHO) n_geom_rho++;
    if (flag & HIT_FLOOR_GEOM_U) n_geom_u++;
    if (flag & HIT_FLOOR_B_RHO) n_b_rho++;
    if (flag & HIT_FLOOR_B_U) n_b_u++;
    if (flag & HIT_FLOOR_TEMP) n_temp++;
    if (flag & HIT_FLOOR_GAMMA) n_gamma++;
    if (flag & HIT_FLOOR_KTOT) n_ktot++;
  }

  n_geom_rho = mpi_reduce_int(n_geom_rho);
  n_geom_u = mpi_reduce_int(n_geom_u);
  n_b_rho = mpi_reduce_int(n_b_rho);
  n_b_u = mpi_reduce_int(n_b_u);
  n_temp = mpi_reduce_int(n_temp);
  n_gamma = mpi_reduce_int(n_gamma);
  n_ktot = mpi_reduce_int(n_ktot);

  LOG("FLOORS:");
  if (n_geom_rho > 0) LOGN("Hit %d GEOM_RHO", n_geom_rho);
  if (n_geom_u > 0) LOGN("Hit %d GEOM_U", n_geom_u);
  if (n_b_rho > 0) LOGN("Hit %d B_RHO", n_b_rho);
  if (n_b_u > 0) LOGN("Hit %d B_U", n_b_u);
  if (n_temp > 0) LOGN("Hit %d TEMPERATURE", n_temp);
  if (n_gamma > 0) LOGN("Hit %d GAMMA", n_gamma);
  if (n_ktot > 0) LOGN("Hit %d KTOT", n_ktot);

#endif

  LOG("End fixup");

  timer_stop(TIMER_FIXUP);
}

inline void fixup_ceiling(struct GridGeom *G, struct FluidState *S, int i, int j, int k)
{
  // First apply ceilings:
  // 1. Limit gamma with respect to normal observer
  double gamma = mhd_gamma_calc(G, S, i, j, k, CENT);

  if (gamma > GAMMAMAX) {
    fflag[k][j][i] |= HIT_FLOOR_GAMMA;

    double f = sqrt((GAMMAMAX*GAMMAMAX - 1.)/(gamma*gamma - 1.));
    S->P[U1][k][j][i] *= f;
    S->P[U2][k][j][i] *= f;
    S->P[U3][k][j][i] *= f;
  }

  // 2. Limit KTOT
#if ELECTRONS
    // Keep to KTOTMAX by controlling u, to avoid anomalous cooling from funnel wall
    // Note: This operates on last iteration's KTOT, meaning the effective value can escape the ceiling
    if (S->P[KTOT][k][j][i] > KTOTMAX) {
      fflag[k][j][i] |= HIT_FLOOR_KTOT;

      S->P[UU][k][j][i] = KTOTMAX*pow(S->P[RHO][k][j][i],gam)/(gam-1.);
      S->P[KTOT][k][j][i] = KTOTMAX;
    }
#endif
}

inline void fixup_floor(struct GridGeom *G, struct FluidState *S, int i, int j, int k)
{
  // Then apply floors:
  // 1. Geometric hard floors, not based on fluid relationships
  double rhoflr_geom, uflr_geom;
  if(METRIC == MKS) {
    double r, th, X[NDIM];
    coord(i, j, k, CENT, X);
    bl_coord(X, &r, &th);

    // New, steeper floor in rho
    // Previously raw r^-2, r^-1.5
    double rhoscal = pow(r, -2.) * 1 / (1 + r/FLOOR_R_CHAR);
    rhoflr_geom = RHOMIN*rhoscal;
    uflr_geom = UUMIN*pow(rhoscal, gam);

    // Impose overall minimum
    // TODO These would only be hit at by r^-3 floors for r_out = 100,000M.  Worth keeping?
    rhoflr_geom = MY_MAX(rhoflr_geom, RHOMINLIMIT);
    uflr_geom = MY_MAX(uflr_geom, UUMINLIMIT);
  } else if (METRIC == MINKOWSKI) {
    rhoflr_geom = RHOMIN*1.e-2;
    uflr_geom = UUMIN*1.e-2;
  }

  // Record Geometric floor hits
  if (rhoflr_geom > S->P[RHO][k][j][i]) fflag[k][j][i] |= HIT_FLOOR_GEOM_RHO;
  if (uflr_geom > S->P[UU][k][j][i]) fflag[k][j][i] |= HIT_FLOOR_GEOM_U;


  // 2. Magnetic floors: impose maximum magnetization sigma = bsq/rho, inverse beta prop. to bsq/U
  //get_state(G, S, i, j, k, CENT); // called above
  double bsq = bsq_calc(S, i, j, k);
  double rhoflr_b = bsq/BSQORHOMAX;
  double uflr_b = bsq/BSQOUMAX;

  // Record Magnetic floor hits
  if (rhoflr_b > S->P[RHO][k][j][i]) fflag[k][j][i] |= HIT_FLOOR_B_RHO;
  if (uflr_b > S->P[UU][k][j][i]) fflag[k][j][i] |= HIT_FLOOR_B_U;

  // Evaluate highest U floor
  double uflr_max = MY_MAX(uflr_geom, uflr_b);

  // 3. Temperature ceiling: impose maximum temperature
  // Take floors on U into account
  double rhoflr_temp = MY_MAX(S->P[UU][k][j][i] / UORHOMAX, uflr_max / UORHOMAX);

  // Record hitting temperature ceiling
  if (rhoflr_temp > S->P[RHO][k][j][i]) fflag[k][j][i] |= HIT_FLOOR_TEMP; // Misnomer for consistency

  // Evaluate highest RHO floor
  double rhoflr_max = MY_MAX(MY_MAX(rhoflr_geom, rhoflr_b), rhoflr_temp);

  if (rhoflr_max > S->P[RHO][k][j][i] || uflr_max > S->P[UU][k][j][i]) { // Apply floors

    // Initialize a dummy fluid parcel
    PLOOP {
      Stmp->P[ip][k][j][i] = 0;
      Stmp->U[ip][k][j][i] = 0;
    }

    // Add mass and internal energy, but not velocity
    Stmp->P[RHO][k][j][i] = MY_MAX(0., rhoflr_max - S->P[RHO][k][j][i]);
    Stmp->P[UU][k][j][i] = MY_MAX(0., uflr_max - S->P[UU][k][j][i]);

    // Get conserved variables for the parcel
    get_state(G, Stmp, i, j, k, CENT);
    prim_to_flux(G, Stmp, i, j, k, 0, CENT, Stmp->U);

    // And for the current state
    //get_state(G, S, i, j, k, CENT); // Called just above, or vectorized above that
    prim_to_flux(G, S, i, j, k, 0, CENT, S->U);

    // Add new conserved variables to current values
    PLOOP {
      S->U[ip][k][j][i] += Stmp->U[ip][k][j][i];
      S->P[ip][k][j][i] += Stmp->P[ip][k][j][i];
    }

    // Recover primitive variables
    // CFG: do we get any failures here?
    pflag[k][j][i] = U_to_P(G, S, i, j, k, CENT);
  }

#if ELECTRONS
    // Reset entropy after floors
    S->P[KTOT][k][j][i] = (gam - 1.)*S->P[UU][k][j][i]/pow(S->P[RHO][k][j][i],gam);
#endif

}

// Replace bad points with values interpolated from neighbors
#define FLOOP for(int ip=0;ip<B1;ip++)
void fixup_utoprim(struct GridGeom *G, struct FluidState *S)
{
  timer_start(TIMER_FIXUP);

  // Flip the logic of the pflag[] so that it now indicates which cells are good
#pragma omp parallel for simd collapse(2)
  ZLOOPALL {
    pflag[k][j][i] = !pflag[k][j][i];
  }

#if DEBUG
  int nbad_utop = 0;
#pragma omp parallel for simd collapse(2) reduction (+:nbad_utop)
  ZLOOP {
    // Count the 0 = bad cells
    nbad_utop += !pflag[k][j][i];
  }
  LOGN("Fixing %d bad cells", nbad_utop);
#endif

  // Make sure we are not using ill defined physical corner regions
  // TODO find a way to do this once, or put it in bounds at least?
  for (int k = 0; k < NG; k++) {
    for (int j = 0; j < NG; j++) {
      for (int i = 0; i < NG; i++) {
        if(global_start[2] == 0 && global_start[1] == 0 && global_start[0] == 0) pflag[k][j][i] = 0;
        if(global_start[2] == 0 && global_start[1] == 0 && global_stop[0] == N1TOT) pflag[k][j][i+N1+NG] = 0;
        if(global_start[2] == 0 && global_stop[1] == N2TOT && global_start[0] == 0) pflag[k][j+N2+NG][i] = 0;
        if(global_stop[2] == N3TOT && global_start[1] == 0 && global_start[0] == 0) pflag[k+N3+NG][j][i] = 0;
        if(global_start[2] == 0 && global_stop[1] == N2TOT && global_stop[0] == N1TOT) pflag[k][j+N2+NG][i+N1+NG] = 0;
        if(global_stop[2] == N3TOT && global_start[1] == 0 && global_stop[0] == N1TOT) pflag[k+N3+NG][j][i+N1+NG] = 0;
        if(global_stop[2] == N3TOT && global_stop[1] == N2TOT && global_start[0] == 0) pflag[k+N3+NG][j+N2+NG][i] = 0;
        if(global_stop[2] == N3TOT && global_stop[1] == N2TOT && global_stop[0] == N1TOT) pflag[k+N3+NG][j+N2+NG][i+N1+NG] = 0;
      }
    }
}

#if DEBUG
  // Keep track of how many points we fix
  int nfixed_utop = 0;
#endif

  // TODO is parallelizing this version okay?
//#pragma omp parallel for collapse(3) reduction(+:bad)
  ZLOOP {
    if (pflag[k][j][i] == 0) {
      double wsum = 0.;
      double sum[B1];
      FLOOP sum[ip] = 0.;
      for (int l = -1; l < 2; l++) {
        for (int m = -1; m < 2; m++) {
          for (int n = -1; n < 2; n++) {
            double w = 1./(abs(l) + abs(m) + abs(n) + 1)*pflag[k+n][j+m][i+l];
            wsum += w;
            FLOOP sum[ip] += w*S->P[ip][k+n][j+m][i+l];
          }
        }
      }
      if(wsum < 1.e-10) {
        fprintf(stderr, "fixup_utoprim: No usable neighbors at %d %d %d\n", i, j, k);
        // TODO set to something ~okay here, or exit screaming
        // This happens /very rarely/
        //exit(-1);
        continue;
      }
      FLOOP S->P[ip][k][j][i] = sum[ip]/wsum;

      // Cell is fixed, could now use for other interpolations
      // However, this is harmful to MPI-determinism
      //pflag[k][j][i] = 1;

#if DEBUG
      nfixed_utop++;
#endif

      // Make sure fixed values still abide by floors
      fixup_ceiling(G, S, i, j, k);
      get_state(G, S, i, j, k, CENT);
      fixup_floor(G, S, i, j, k);
    }
  }

#if DEBUG
  int nleft_utop = nbad_utop - nfixed_utop;
  if(nleft_utop > 0) fprintf(stderr,"Cells STILL BAD after fixup_utoprim: %d\n", nleft_utop);
#endif

  // Reset the pflag
#pragma omp parallel for simd collapse(2)
  ZLOOPALL {
    pflag[k][j][i] = 0;
  }

  timer_stop(TIMER_FIXUP);
}
#undef FLOOP
