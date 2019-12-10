/******************************************************************************  
 *                                                                            *  
 * PARAMETERS.H                                                               *  
 *                                                                            *  
 * PROBLEM-SPECIFIC CHOICES                                                   *  
 *                                                                            *  
 ******************************************************************************/

/* GLOBAL RESOLUTION */
#define N1TOT 64
#define N2TOT 64
#define N3TOT 64

/* MPI DECOMPOSITION */
/* DECOMPOSE IN N3 FIRST! Small leading array sizes for linear access */
#define N1CPU 2
#define N2CPU 2
#define N3CPU 4

/* METRIC
 *   MINKOWSKI, MKS
 */
#define METRIC MINKOWSKI

/*
 * FLOORS
 * Wind term is a small source for torii only
 * Maximum magnetization parameters should be set high for most problems
 */
#define WIND_TERM 0
#define BSQORHOMAX (100.)
#define UORHOMAX (100.)

/* ELECTRONS AND OPTIONS
 *   SUPPRESS_MAG_HEAT - (0,1) NO ELECTRON HEATING WHEN SIGMA > 1
 *   BETA_HEAT         - (0,1) BETA-DEPENDENT HEATING
 */
#define ELECTRONS           1
#define SUPPRESS_HIGHB_HEAT 1
#define BETA_HEAT           1

/* RECONSTRUCTION ALGORITHM
 *   LINEAR, PPM, WENO, MP5
 */
#define RECONSTRUCTION LINEAR

/* BOUNDARY CONDITIONS
 *   OUTFLOW PERIODIC POLAR USER
 */
#define X1L_BOUND PERIODIC
#define X1R_BOUND PERIODIC
#define X2L_BOUND PERIODIC
#define X2R_BOUND PERIODIC
#define X3L_BOUND PERIODIC
#define X3R_BOUND PERIODIC

