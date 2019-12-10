/******************************************************************************
 *                                                                            *
 * PROBLEM.C                                                                  *
 *                                                                            *
 * INITIAL CONDITIONS FOR ENTROPY WAVE                                        *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"
#include <complex.h>

static int nmode;

void set_problem_params()
{
  set_param("nmode", &nmode);
}

void init(struct GridGeom *G, struct FluidState *S)
{
  double X[NDIM];

  // Mean state
  double rho0 = 1.;
  double u0 = 1.; // TODO set U{n} for boosted entropy
  double B10 = 0.; // This is set later, see below
  double B20 = 0.;
  double B30 = 0.;

  // Wavevector
  double k1 = 2.*M_PI;
  double k2 = 2.*M_PI;
  double k3 = 2.*M_PI;
  double amp = 1.e-4;

  // "Faux-2D" planar waves direction
  // Set to 0 for "full" 3D wave
  int dir = 0;

  complex omega, drho, du, du1, du2, du3, dB1, dB2, dB3;

  // Default value 0
  omega = 0.;
  drho = 0.;
  du   = 0.;
  du1  = 0.;
  du2  = 0.;
  du3  = 0.;
  dB1  = 0.;
  dB2  = 0.;
  dB3  = 0.;

  // Eigenmode definitions
  if (dir == 0) {
    // 3D (1,1,1) wave
    B10 = 1.;
    B20 = 0.;
    B30 = 0.;
    if (nmode == 0) { // Entropy
     omega = 2.*M_PI/5.*I;
     drho = 1.;
    } else if (nmode == 1) { // Slow
     omega = 2.35896379113*I;
     drho = 0.556500332363;
     du   = 0.742000443151;
     du1  = -0.282334999306;
     du2  = 0.0367010491491;
     du3  = 0.0367010491491;
     dB1  = -0.195509141461;
     dB2  = 0.0977545707307;
     dB3  = 0.0977545707307;
    } else if (nmode == 2) { // Alfven
      omega = - 3.44144232573*I;
      du2   = -0.339683110243;
      du3   = 0.339683110243;
      dB2   = 0.620173672946;
      dB3   = -0.620173672946;
    } else { // Fast
      omega =  6.92915162882*I;
      drho  =  0.481846076323;
      du    =  0.642461435098;
      du1   =  -0.0832240462505;
      du2   =  -0.224080007379;
      du3   =  -0.224080007379;
      dB1   =  0.406380545676;
      dB2   =  -0.203190272838;
      dB3   =  -0.203190272838;
    }
  } else {
    // 2D (1,1,0), (1,0,1), (0,1,1) wave
    // Constant field direction
    if (dir ==1) {
      B20 = 1.;
    } else if (dir == 2) {
      B30 = 1.;
    } else if (dir == 3) {
      B10 = 1.;
    }

    if (nmode == 0) { // Entropy
      omega = 2.*M_PI/5.*I;
      drho = 1.;
    } else if (nmode == 1) { // Slow
      omega = 2.41024185339*I;
      drho = 0.558104461559;
      du   = 0.744139282078;
      if (dir == 1) {
       du2  = -0.277124827421;
       du3  = 0.0630348927707;
       dB2  = -0.164323721928;
       dB3  = 0.164323721928;
      } else if (dir == 2) {
       du3  = -0.277124827421;
       du1  = 0.0630348927707;
       dB3  = -0.164323721928;
       dB1  = 0.164323721928;
      } else if (dir == 3) {
       du1  = -0.277124827421;
       du2  = 0.0630348927707;
       dB1  = -0.164323721928;
       dB2  = 0.164323721928;
      }
    } else if (nmode == 2) { // Alfven
      omega = 3.44144232573*I;
      if (dir == 1) {
       du1 = 0.480384461415;
       dB1 = 0.877058019307;
      } else if (dir == 2) {
       du2 = 0.480384461415;
       dB2 = 0.877058019307;
      } else if (dir == 3) {
       du3 = 0.480384461415;
       dB3 = 0.877058019307;
      }
    } else { // Fast
      omega = 5.53726217331*I;
      drho = 0.476395427447;
      du   = 0.635193903263;
      if (dir == 1) {
       du2  = -0.102965815319;
       du3  = -0.316873207561;
       dB2  = 0.359559114174;
       dB3  = -0.359559114174;
      } else if (dir == 2) {
       du3  = -0.102965815319;
       du1  = -0.316873207561;
       dB3  = 0.359559114174;
       dB1  = -0.359559114174;
      } else if (dir == 3) {
       du1  = -0.102965815319;
       du2  = -0.316873207561;
       dB1  = 0.359559114174;
       dB2  = -0.359559114174;
      }
    }
  }

  // Override tf and the dump and log intervals
  tf = 2.*M_PI/fabs(cimag(omega));
  DTd = tf/5.; // These are set from param.dat
  DTl = tf/5.;

  set_grid(G);

  LOG("Set grid");

  ZLOOP {
    coord(i, j, k, CENT, X);

    double mode = 0;
    if (dir == 1) {
      mode = amp*cos(k1*X[2] + k2*X[3]);
    } else if (dir == 2) {
      mode = amp*cos(k1*X[1] + k2*X[3]);
    } else if (dir == 3) {
      mode = amp*cos(k1*X[1] + k2*X[2]);
    } else {
      mode = amp*cos(k1*X[1] + k2*X[2] + k3*X[3]);
    }

    S->P[RHO][k][j][i] = rho0 + creal(drho*mode);
    S->P[UU][k][j][i] = u0 + creal(du*mode);
    S->P[U1][k][j][i] = creal(du1*mode);
    S->P[U2][k][j][i] = creal(du2*mode);
    S->P[U3][k][j][i] = creal(du3*mode);
    S->P[B1][k][j][i] = B10 + creal(dB1*mode);
    S->P[B2][k][j][i] = B20 + creal(dB2*mode);
    S->P[B3][k][j][i] = B30 + creal(dB3*mode);

  } // ZLOOP

  //Enforce boundary conditions
  set_bounds(G, S);
}
