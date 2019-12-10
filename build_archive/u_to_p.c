/******************************************************************************
 *                                                                            *
 * UTOP.C                                                                     *
 *                                                                            *
 * INVERTS FROM CONSERVED TO PRIMITIVE VARIABLES BASED ON MIGNONE &           *
 * MCKINNEY 2007                                                              *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

#define ERRTOL  1.e-8
#define ITERMAX 8
#define DEL 1.e-5

double Pressure_rho0_u(double rho0, double u);
double Pressure_rho0_w(double rho0, double w);
double err_eqn(double Bsq, double D, double Ep, double QdB, double Qtsq,
  double Wp, int *eflag);
double gamma_func(double Bsq, double D, double QdB, double Qtsq, double Wp,
  int *eflag);
double Wp_func(struct GridGeom *G, struct FluidState *S, int i, int j, int k,
  int loc, int *eflag);

int U_to_P(struct GridGeom *G, struct FluidState *S, int i, int j, int k, 
  int loc)
{

  int eflag = 0 ;

  double gdet = G->gdet[loc][j][i];
  double lapse = G->lapse[loc][j][i];

  // Update the primitive B-fields
  S->P[B1][k][j][i] = S->U[B1][k][j][i]/gdet;
  S->P[B2][k][j][i] = S->U[B2][k][j][i]/gdet;
  S->P[B3][k][j][i] = S->U[B3][k][j][i]/gdet;

  // Catch negative density
  if (S->U[RHO][k][j][i] <= 0.) {
    eflag = -100;
    return (eflag);
  }

  // Convert from conserved variables to four-vectors
  double D = S->U[RHO][k][j][i]*lapse/gdet;

  double Bcon[NDIM];
  Bcon[0] = 0.;
  Bcon[1] = S->U[B1][k][j][i]*lapse/gdet;
  Bcon[2] = S->U[B2][k][j][i]*lapse/gdet;
  Bcon[3] = S->U[B3][k][j][i]*lapse/gdet;

  double Qcov[NDIM];
  Qcov[0] = (S->U[UU][k][j][i] - S->U[RHO][k][j][i])*lapse/gdet;
  Qcov[1] = S->U[U1][k][j][i]*lapse/gdet;
  Qcov[2] = S->U[U2][k][j][i]*lapse/gdet;
  Qcov[3] = S->U[U3][k][j][i]*lapse/gdet;

  double ncov[NDIM];
  ncov[0] = -lapse;
  ncov[1] = 0.;
  ncov[2] = 0.;
  ncov[3] = 0.;

  // Interlaced upper/lower operation
  double Bcov[NDIM], Qcon[NDIM], ncon[NDIM];
  for (int mu = 0; mu < NDIM; mu++) {
    Bcov[mu] = 0.;
    Qcon[mu] = 0.;
    ncon[mu] = 0.;
    for (int nu = 0; nu < NDIM; nu++) {
      Bcov[mu] += G->gcov[CENT][mu][nu][j][i]*Bcon[nu];
      Qcon[mu] += G->gcon[CENT][mu][nu][j][i]*Qcov[nu];
      ncon[mu] += G->gcon[CENT][mu][nu][j][i]*ncov[nu];
    }
  }
  //raise_grid(ncov, ncon, G, i, j, k, loc);
  double Bsq = dot(Bcon, Bcov);
  double QdB = dot(Bcon, Qcov);
  double Qdotn = dot(Qcon, ncov);
  double Qsq = dot(Qcon, Qcov);

  double Qtcon[NDIM];
  DLOOP1 Qtcon[mu] = Qcon[mu] + ncon[mu]*Qdotn;
  double Qtsq = Qsq + Qdotn*Qdotn;

  // Set up eqtn for W'; this is the energy density
  double Ep = -Qdotn - D;

  // Numerical rootfinding
  // Take guesses from primitives.
  double Wp = Wp_func(G, S, i, j, k, loc, &eflag);
  if (eflag)
    return eflag;

  // Step around the guess & evaluate errors
  double Wpm = (1. - DEL)*Wp; //heuristic
  double h = Wp - Wpm;
  double Wpp = Wp + h;
  double errp = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wpp, &eflag);
  double err = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wp, &eflag);
  double errm = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wpm, &eflag);

  // TODO why does I17 botch the following

  // Attempt a Halley/Muller/Bailey/Press step
  double dedW = (errp - errm)/(Wpp - Wpm);
  double dedW2 = (errp - 2.*err + errm)/(h*h);
  double f = 0.5*err*dedW2/(dedW*dedW);
  // Limit size of 2nd derivative correction
  if (f < -0.3) f = -0.3;
  if (f > 0.3) f = 0.3;

  double dW = -err/dedW/(1. - f);
  double Wp1 = Wp;
  double err1 = err;
  // Limit size of step
  if (dW < -0.5*Wp) dW = -0.5*Wp;
  if (dW > 2.0*Wp) dW = 2.0*Wp;

  Wp += dW;
  err = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wp, &eflag);

  // Not good enough?  apply secant method
  int iter = 0;
  for (iter = 0; iter < ITERMAX; iter++) {
    dW = (Wp1 - Wp)*err/(err - err1);

    // TODO should this have limit applied?
    Wp1 = Wp;
    err1 = err;

    // Normal secant increment is dW. Also limit guess to between 0.5 and 2
    // times the current value
    if (dW < -0.5*Wp) dW = -0.5*Wp;
    if (dW > 2.0*Wp) dW = 2.0*Wp;

    Wp += dW;

    if (fabs(dW/Wp) < ERRTOL) {
      //fprintf(stderr, "Breaking!\n");
      break;
    }

    err = err_eqn(Bsq, D, Ep, QdB, Qtsq, Wp, &eflag);
    //fprintf(stderr, "%.15f ", err);

    if (fabs(err/Wp) < ERRTOL) {
      //fprintf(stderr, "Breaking!\n");
      break;
    }
  }
  // Failure to converge; do not set primitives other than B
  if(iter == ITERMAX) {
    return(1);
  }

  // Find utsq, gamma, rho0 from Wp
  double gamma = gamma_func(Bsq, D, QdB, Qtsq, Wp, &eflag);
  if (gamma < 1.) {
    fprintf(stderr,"gamma < 1 failure.\n");
    exit(1);
  }

  // Find the scalars
  double rho0 = D/gamma;
  double W = Wp + D;
  double w = W/(gamma*gamma);
  double P = Pressure_rho0_w(rho0, w);
  double u = w - (rho0 + P);

  // Return without updating non-B primitives
  if (rho0 < 0 && u <0) {
    return 8;
  } else if (rho0 < 0) {
    return 6;
  } else if (u < 0) {
    return 7;
  }

  // Set primitives
  S->P[RHO][k][j][i] = rho0;
  S->P[UU][k][j][i] = u;

  // Find u(tilde); Eqn. 31 of Noble et al.
  S->P[U1][k][j][i] = (gamma/(W + Bsq))*(Qtcon[1] + QdB*Bcon[1]/W);
  S->P[U2][k][j][i] = (gamma/(W + Bsq))*(Qtcon[2] + QdB*Bcon[2]/W);
  S->P[U3][k][j][i] = (gamma/(W + Bsq))*(Qtcon[3] + QdB*Bcon[3]/W);

#if ELECTRONS
  S->P[KEL][k][j][i] = S->U[KEL][k][j][i]/S->U[RHO][k][j][i];
  S->P[KTOT][k][j][i] = S->U[KTOT][k][j][i]/S->U[RHO][k][j][i];
#endif // ELECTRONS

  return 0;
}

inline double err_eqn(double Bsq, double D, double Ep, double QdB, double Qtsq,
  double Wp, int *eflag)
{

  double W = Wp + D ;
  double gamma = gamma_func(Bsq, D, QdB, Qtsq, Wp, eflag);
  double w = W/(gamma*gamma);
  double rho0 = D/gamma;
  double p = Pressure_rho0_w(rho0,w);

  double err = -Ep + Wp - p + 0.5*Bsq + 0.5*(Bsq*Qtsq - QdB*QdB)/((Bsq + W)*(Bsq + W));

  return err;
}

inline double gamma_func(double Bsq, double D, double QdB, double Qtsq, double Wp,
  int *eflag)
{
  double QdBsq, W, utsq, gamma, W2, WB;

  QdBsq = QdB*QdB;
  W = D + Wp;
  W2 = W*W;
  WB = W + Bsq;

  // This is basically inversion of eq. A7 of MM
  utsq = -((W + WB)*QdBsq + W2*Qtsq)/(QdBsq*(W + WB) + W2*(Qtsq - WB*WB));
  gamma = sqrt(1. + fabs(utsq));

  // Catch utsq < 0
  if(utsq < 0. || utsq > 1.e3*GAMMAMAX*GAMMAMAX) {
    *eflag = 2;
  }

  return gamma;
}

//double Wp_func(double *prim, struct of_geom *geom, int *eflag)
inline double Wp_func(struct GridGeom *G, struct FluidState *S, int i, int j, int k,
  int loc, int *eflag)
{
  double rho0, u, utcon[NDIM], utcov[NDIM], utsq, gamma, Wp;

  rho0 = S->P[RHO][k][j][i];
  u    = S->P[UU][k][j][i];

  utcon[0] = 0.;
  utcon[1] = S->P[U1][k][j][i];
  utcon[2] = S->P[U2][k][j][i];
  utcon[3] = S->P[U3][k][j][i];

  //lower_grid(utcon, utcov, G, i, j, k, loc);
  for (int mu = 0; mu < NDIM; mu++) {
    utcov[mu] = 0.;
    for (int nu = 0; nu < NDIM; nu++) {
      utcov[mu] += G->gcov[CENT][mu][nu][j][i]*utcon[nu];
    }
  }
  utsq = dot(utcon, utcov);

  // Catch utsq < 0
  if ((utsq < 0.) && (fabs(utsq) < 1.e-13)) {
    utsq = fabs(utsq);
  }
  if (utsq < 0. || utsq > 1.e3*GAMMAMAX*GAMMAMAX) {
    *eflag = 2 ;
    return rho0 + u; // Not sure what to do here...
  }

  gamma = sqrt(1. + fabs(utsq));

  Wp = (rho0 + u + Pressure_rho0_u(rho0,u))*gamma*gamma - rho0*gamma;

  return Wp;
}

// Equation of state
inline double Pressure_rho0_u(double rho0, double u)
{
  return (gam - 1.)*u;
}

inline double Pressure_rho0_w(double rho0, double w)
{
  return (w - rho0)*(gam - 1.)/gam;
}

