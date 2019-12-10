/******************************************************************************
 *                                                                            *
 * PROBLEM.C                                                                  *
 *                                                                            *
 * INITIAL CONDITIONS FOR BONDI INFLOW                                        *
 *                                                                            *
 ******************************************************************************/

#include "bl_coord.h"
#include "decs.h"

// Rootfinding for analytic Bondi solution
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

double C1, C2, n;

double mdot, rs;
void set_problem_params() {
  set_param("mdot", &mdot);
  set_param("rs", &rs);
  set_param("Rhor", &Rhor);
}

// Adapted from M. Chandra
double get_Tfunc(double T, double r)
{
  return pow(1.+(1.+n)*T,2.)*(1.-2./r+pow(C1/r/r/pow(T,n),2.))-C2;
}

double get_T(double r)
{
  double rtol = 1.e-12;
  double ftol = 1.e-14;
  double Tmin = 0.6*(sqrt(C2) - 1.)/(n + 1);
  double Tmax = pow(C1*sqrt(2./r/r/r),1./n);
  double f0, f1, fh;
  double T0, T1, Th;
  T0 = 0.6*Tmin;
  f0 = get_Tfunc(T0, r);
  T1 = Tmax;
  f1 = get_Tfunc(T1, r);

  if (f0*f1 > 0.) {
    printf("Failed solving for T at r = %e C1 = %e C2 = %e\n", r, C1, C2);
    exit(-1);
  }

  Th = (f1*T0 - f0*T1)/(f1 - f0);
  fh = get_Tfunc(Th, r);
  double epsT = rtol*(Tmin + Tmax);
  while (fabs(Th - T0) > epsT && fabs(Th - T1) > epsT && fabs(fh) > ftol) {
    if (fh*f0 < 0.) {
      T0 = Th;
      f0 = fh;
    } else {
      T1 = Th;
      f1 = fh;
    }

    Th = (f1*T0 - f0*T1)/(f1 - f0);
    fh = get_Tfunc(Th, r);
  }

  return Th;
}

void fourvel_to_prim(double ucon[NDIM], GridPrim P,
  struct GridGeom *G, int i, int j, int k)
{
  double alpha, beta[NDIM], gamma;

  alpha = 1.0/sqrt(-G->gcon[CENT][0][0][j][i]);
  beta[1] = alpha*alpha*G->gcon[CENT][0][1][j][i];
  beta[2] = alpha*alpha*G->gcon[CENT][0][2][j][i];
  beta[3] = alpha*alpha*G->gcon[CENT][0][3][j][i];
  gamma = ucon[0]*alpha;

  P[U1][k][j][i] = ucon[1] + beta[1]*gamma/alpha;
  P[U2][k][j][i] = ucon[2] + beta[2]*gamma/alpha;
  P[U3][k][j][i] = ucon[3] + beta[3]*gamma/alpha;
}

void set_ut(double ucon[NDIM], struct of_geom *geom)
{
  double AA, BB, CC;

  AA = geom->gcov[0][0];
  BB = 2.*(geom->gcov[0][1]*ucon[1] +
           geom->gcov[0][2]*ucon[2] +
           geom->gcov[0][3]*ucon[3]);
  CC = 1. + geom->gcov[1][1]*ucon[1]*ucon[1] +
            geom->gcov[2][2]*ucon[2]*ucon[2] +
            geom->gcov[3][3]*ucon[3]*ucon[3] +
       2. *(geom->gcov[1][2]*ucon[1]*ucon[2] +
            geom->gcov[1][3]*ucon[1]*ucon[3] +
            geom->gcov[2][3]*ucon[2]*ucon[3]);

  double discr = BB*BB - 4.*AA*CC;
  ucon[0] = (-BB - sqrt(discr))/(2.*AA);
}

void get_prim_bondi(int i, int j, int k, GridPrim P, struct GridGeom *G)
{
  static int firstc = 1;
  if (firstc) {
    n = 1./(gam - 1.);

    // Solution constants
    double uc = sqrt(mdot/(2.*rs));
    double Vc = -sqrt(pow(uc,2)/(1. - 3.*pow(uc,2)));
    double Tc = -n*pow(Vc,2)/((n + 1.)*(n*pow(Vc,2) - 1.));
    C1 = uc*pow(rs,2)*pow(Tc,n);
    C2 = pow(1. + (1. + n)*Tc,2)*(1. - 2.*mdot/rs + pow(C1,2)/
       (pow(rs,4)*pow(Tc,2*n)));

    firstc = 0;
  }

  double r, th, X[NDIM];
  coord(i, j, k, CENT, X);
  bl_coord(X, &r, &th);

  while (r < Rhor) {
    i++;
    coord(i, j, k, CENT, X);
    bl_coord(X, &r, &th);
  }

  //double T = T_bondi[j][i];
  double T = get_T(r);
  double ur = -C1/(pow(T,n)*pow(r,2));
  double rho = pow(T,n);
  double u = rho*T/(gam - 1.);
  double ucon_bl[NDIM], ucon_ks[NDIM], ucon_mks[NDIM];
  struct of_geom geom_bl;

  blgset(i, j, &geom_bl);

  DLOOP1 {
    ucon_bl[mu] = 0.;
    ucon_ks[mu] = 0.;
    ucon_mks[mu] = 0.;
  }
  ucon_bl[1] = ur;

  set_ut(ucon_bl, &geom_bl);
  bl_to_ks(X, ucon_bl, ucon_ks);

  double dxdX[NDIM][NDIM], dXdx[NDIM][NDIM];
  set_dxdX(X, dxdX);
  invert(&dxdX[0][0], &dXdx[0][0]);
  DLOOP2 {
    ucon_mks[mu] += dXdx[mu][nu]*ucon_ks[nu];
  }

  fourvel_to_prim(ucon_mks, P, G, i, j, k);

  P[RHO][k][j][i] = rho;
  P[UU][k][j][i] = u;
  P[B1][k][j][i] = 0.;
  P[B2][k][j][i] = 0.;
  P[B3][k][j][i] = 0.;

  // Electrons make no physical sense here but are a very useful debug tool
  // At least set them consistently here to test deterministic evolution
#if ELECTRONS
    // Set electron internal energy to constant fraction of internal energy
    double uel = fel0*P[UU][k][j][i];

    // Initialize entropies
    P[KTOT][k][j][i] = (gam-1.)*P[UU][k][j][i]*pow(P[RHO][k][j][i],-gam);
    P[KEL][k][j][i] = (game-1.)*uel*pow(P[RHO][k][j][i],-game);
#endif

}

void init(struct GridGeom *G, struct FluidState *S)
{

  set_grid(G);

  LOG("Grid set");

  ZLOOP {
    get_prim_bondi(i, j, k, S->P, G);
  }

  if (DEBUG && mpi_io_proc()) {
    printf("a = %e Rhor = %e\n", a, Rhor);

    printf("mdot = %e\n", mdot);
    printf("rs   = %e\n", rs);
    printf("n    = %e\n", n);
    printf("C1   = %e\n", C1);
    printf("C2   = %e\n", C2);
  }

#if ELECTRONS
  init_electrons(G, S);
#endif

  //Enforce boundary conditions
  fixup(G, S);
  set_bounds(G, S);
}

void bound_gas_prob_x1r(int i, int j, int k, GridPrim  P, struct GridGeom *G)
{
  get_prim_bondi(i, j, k, P, G);
}
