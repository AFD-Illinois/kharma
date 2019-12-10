/******************************************************************************
 *                                                                            *
 * PROBLEM.C                                                                  *
 *                                                                            *
 * INITIAL CONDITIONS FOR ENTROPY WAVE                                        *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"
#include <complex.h>

static int nmode = NMODE;
//void set_problem_params()
//{
//  set_param("nmode", &nmode);
//}

void init(struct GridGeom *G, struct FluidState *S)
{
	double X[NDIM];

  // Mean state
  double rho0 = 1.;
  double u0 = 1.;
  double B10 = 1.;
  double B20 = 0.;
  double B30 = 0.;
  
  // Wavevector
  double k1 = 2.*M_PI;
  double amp = 1.e-4;

  complex omega, drho, du, du1, du2, du3, dB1, dB2, dB3;

  // Eigenmode
  if (nmode == 0) { // Entropy
    omega = 2.*M_PI/5.*I; // To get tf
    drho  = 1.;
    du    = 0.;
    du1   = 0.;
    du2   = 0.;
    du3   = 0.;
    dB1   = 0.;
    dB2   = 0.;
    dB3   = 0.;
  } else if (nmode == 1) { // Slow
    omega = 2.74220688339*I;
    drho  = 0.580429492464;
    du    = 0.773905989952;
    du1   = -0.253320198552;
    du2   = 0.;
    du3   = 0.;
    dB1   = 0.;
    dB2   = 0.;
    dB3   = 0.;
  } else if (nmode == 2) { // Alfven
    omega = 3.44144232573*I;
    drho  = 0.;
    du    = 0.;
    du1   = 0.;
    du2   = 0.480384461415;
    du3   = 0.;
    dB1   = 0.;
    dB2   = 0.877058019307;
    dB3   = 0.;
  } else { // Fast
    omega = 3.44144232573*I;
    drho  = 0.;
    du    = 0.;
    du1   = 0.;
    du2   = 0.;
    du3   = 0.480384461415;
    dB1   = 0.;
    dB2   = 0.;
    dB3   = 0.877058019307;
  }

  t = 0;
  tf = 2.*M_PI/cimag(omega);
  DTd = tf/10.;
  DTl = tf/10.;
  DTr = 10000;  // Restart interval, in timesteps
  DTp = 10;   // Performance interval, in timesteps

  dt = 1.e-6;

  x1Min = 0.;
  x1Max = 1.;
  x2Min = 0.;
  x2Max = 1.;
  x3Min = 0.;
  x3Max = 1.;

  gam = 4./3;
  cour = 2./5;

  zero_arrays();
  set_grid(G);

  printf("grid set\n");

  ZLOOP {
    coord(i, j, k, CENT, X);
    double mode = amp*cos(k1*X[1]);

    S->P[RHO][k][j][i] = rho0 + creal(drho*mode);
    S->P[UU][k][j][i] = u0 + creal(du*mode);
    S->P[U1][k][j][i] = creal(du1*mode);
    S->P[U2][k][j][i] = creal(du2*mode);
    S->P[U3][k][j][i] = creal(du3*mode);
    S->P[B1][k][j][i] = B10 + creal(dB1*mode);
    S->P[B2][k][j][i] = B20 + creal(dB2*mode);
    S->P[B3][k][j][i] = B30 + creal(dB3*mode);

  } // ZLOOP

  //Enforce boundary conditions??
  fixup(G, S);
  set_bounds(G, S);
}

