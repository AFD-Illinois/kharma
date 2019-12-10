/******************************************************************************  
 *                                                                            *  
 * PROBLEM.C                                                                  *  
 *                                                                            *  
 * INITIAL CONDITIONS FOR SOD SHOCKTUBE                                       *  
 *                                                                            *  
 ******************************************************************************/

#include "../../core/decs.h"

void init(struct GridGeom *G, struct FluidState *S)
{
	double X[NDIM];

	// Adiabatic index
	gam = 1.4;

	// Numerical parameters
	lim = MC;
	failed = 0;
	cour = 0.3;
	tf = 0.25;

  // Make problem nonrelativistic
  double tscale = 1.e-2;
  tf /= tscale;
  dt = tf/1.e6;

  // Set simulation size
  x1Min = 0.;
  x1Max = 1.;
  x2Min = 0.;
  x2Max = 1.;
  x3Min = 0.;
  x3Max = 1.;

  zero_arrays();
  set_grid(G);

	t = 0.;

	// Output choices
	DTd = tf/10.;		// Dump interval
	DTl = tf/100.;	// Log interval
	DTr = 512;	// Restart interval, in timesteps
  DTp = 10;

	// Diagnostic counters
	dump_cnt = 0;
	rdump_cnt = 0;

  ZLOOP {
    coord(i, j, k, CENT, X);

    S->P[RHO][k][j][i] = (X[1] < 0.5 || X[1] > 1.5) ? 1.0 : 0.125;

    double pgas = (X[1] < 0.5 || X[1] > 1.5) ? 1.0 : 0.1;                        
    S->P[UU][k][j][i] = pgas/(gam - 1.);

    S->P[U1][k][j][i] = 0.;
    S->P[U2][k][j][i] = 0.;
    S->P[U3][k][j][i] = 0.;
    S->P[B1][k][j][i] = 0.;
    S->P[B2][k][j][i] = 0.;
    S->P[B3][k][j][i] = 0.;
  } // ZLOOP

  // Rescale to make problem nonrelativistic
  ZLOOP {
    S->P[UU][k][j][i] *= tscale*tscale;
    S->P[U1][k][j][i] *= tscale;
    S->P[U2][k][j][i] *= tscale;
    S->P[U3][k][j][i] *= tscale;
    S->P[B1][k][j][i] *= tscale;
    S->P[B2][k][j][i] *= tscale;
    S->P[B3][k][j][i] *= tscale;
  }

	// Enforce boundary conditions
	fixup(G, S);
	set_bounds(G, S);
}

