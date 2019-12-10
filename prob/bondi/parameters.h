/******************************************************************************
 *                                                                            *
 * PARAMETERS.H                                                               *
 *                                                                            *
 * PROBLEM-SPECIFIC CHOICES                                                   *
 *                                                                            *
 ******************************************************************************/

/* GLOBAL RESOLUTION */
#define N1TOT 256
#define N2TOT 256
#define N3TOT 1

/* MPI DECOMPOSITION */
/* Careful NXCPU < NXTOT!! */
#define N1CPU 2
#define N2CPU 2
#define N3CPU 1

/* METRIC
 *   MINKOWSKI, MKS
 */
#define METRIC MKS
#define DEREFINE_POLES 0

/* FLOORS
 *   Wind term is a small source for torii only
 *   Maximum magnetization parameters should be set high for most problems
 */
#define WIND_TERM 0
#define BSQORHOMAX (100.)
#define UORHOMAX (100.)

/* ELECTRONS AND OPTIONS
 *   SUPPRESS_MAG_HEAT - (0,1) NO ELECTRON HEATING WHEN SIGMA > 1
 *   BETA_HEAT         - (0,1) BETA-DEPENDENT HEATING
 */
#define ELECTRONS 0
#define SUPPRESS_HIGHB_HEAT 1
#define BETA_HEAT 1

/* RECONSTRUCTION ALGORITHM
 *   LINEAR, PPM, WENO, MP5
 */
#define RECONSTRUCTION WENO

/* BOUNDARY CONDITIONS
 *   OUTFLOW PERIODIC POLAR USER
 */
#define X1L_BOUND OUTFLOW
#define X1R_BOUND USER
#define X2L_BOUND POLAR
#define X2R_BOUND POLAR
#define X3L_BOUND PERIODIC
#define X3R_BOUND PERIODIC

#define X1L_INFLOW 0
#define X1R_INFLOW 1
