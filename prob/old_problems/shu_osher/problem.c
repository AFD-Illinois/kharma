/******************************************************************************  
 *                                                                            *  
 * PROBLEM.C                                                                  *  
 *                                                                            *  
 * INITIAL CONDITIONS FOR SHU-OSHTER SHOCKTUBE                                *  
 *                                                                            *  
 ******************************************************************************/

#include "../../core/decs.h"

void init()
{
	double X[NDIM];

	// Adiabatic index
	gam = 1.4;

	// Numerical parameters
	lim = MC;
	failed = 0;
	cour = 0.3;
	tf = 1.8;

  // Make problem nonrelativistic
  double tscale = 1.e-2;
  tf /= tscale;
  dt = tf/1.e6;

  // Set simulation size
  x1Min = -4.5;
  x1Max = 4.5;
  x2Min = 0.;
  x2Max = 1.;
  x3Min = 0.;
  x3Max = 1.;

	t = 0.;

	zero_arrays();
	set_grid();

	// Output choices
	DTd = tf/10.;		// Dump interval
	DTl = tf/100.;	// Log interval
	DTr = 50000;	// Restart interval, in timesteps
  DTp = 10;

	// Diagnostic counters
	dump_cnt = 0;
	rdump_cnt = 0;

  ZLOOP {
    coord(i, j, k, CENT, X);

    if (X[1] < -4.) {
      P[i][j][k][RHO] = 3.857143;
      P[i][j][k][UU] = 10.33333/(gam-1.);
      P[i][j][k][U1] = 2.629369;
    } else {
      P[i][j][k][RHO] = 1. + 0.2*sin(5.*X[1]);
      P[i][j][k][UU] = 1./(gam-1.);
      P[i][j][k][U1] = 0.;
    }

    P[i][j][k][U2] = 0.;
    P[i][j][k][U3] = 0.;
    P[i][j][k][B1] = 0.;
    P[i][j][k][B2] = 0.;
    P[i][j][k][B3] = 0.;
  } // ZLOOP

  // Rescale to be nonrelativistic
  ZLOOP {
    P[i][j][k][UU] *= tscale*tscale;
    P[i][j][k][U1] *= tscale;
    P[i][j][k][U2] *= tscale;
    P[i][j][k][U3] *= tscale;
    P[i][j][k][B1] *= tscale;
    P[i][j][k][B2] *= tscale;
    P[i][j][k][B3] *= tscale;
  }

	// Enforce boundary conditions
	fixup(P);
	bound_prim(P);
}

