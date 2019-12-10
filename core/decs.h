/******************************************************************************
 *                                                                            *
 * DECS.H                                                                     *
 *                                                                            *
 * GLOBAL MACROS, FUNCTION DEFINITIONS, INCLUDES, AND DECLARATIONS            *
 *                                                                            *
 ******************************************************************************/

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <errno.h> //Errors for syscalls

#include <omp.h>

// Required globally for pack.c function signatures
#include <hdf5.h>

#include "parameters.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923132169164
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.4142135623730950488016887242
#endif

/*******************************************************************************
      COMPILE-TIME PARAMETERS :
*******************************************************************************/

#define VERSION "iharm-alpha-3.6"

// Number of active zones on each MPI process
#define N1       (N1TOT/N1CPU)
#define N2       (N2TOT/N2CPU)
#define N3       (N3TOT/N3CPU)

// Max size for 1D slice is NMAX
#define N12      (N1 > N2 ? N1 : N2)
#define NMAX     (N12 > N3 ? N12 : N3)

#define NDIM       (4)    // Number of total dimensions
#define NG         (3)    // Number of ghost zones

// Fixup parameters
#define RHOMINLIMIT (1.e-20)
#define UUMINLIMIT  (1.e-20)
#define RHOMIN  (1.e-6)
#define UUMIN (1.e-8)

// Numerical convenience to represent a small (<< 1) non-zero quantity
#define SMALL (1.e-20)
// Set the spatial discretization in numerical derivatives
#define DELTA 1.e-5

// TODO Move this per-problem.  Keep defaults here?
// Floors in terms of bsq
#ifndef BSQORHOMAX
#define BSQORHOMAX (100.)
#endif
#ifndef UORHOMAX
#define UORHOMAX (100.)
#endif
#ifndef BSQOUMAX
#define BSQOUMAX (BSQORHOMAX * UORHOMAX)
#endif
// Extra "wind" source term to avoid floors
// Test problems require disabling this
#ifndef WIND_TERM
#define WIND_TERM 0
#endif

// Maximum value of gamma, the Lorentz factor
#define GAMMAMAX (50.)

// Maximum fractional increase in timestep per timestep
#define SAFE  (1.3)

// Whether to move polar axis slightly off of coordinate singularity
#define COORDSINGFIX 1
#define SINGSMALL (1.E-20)

// Flags. Can be set in compile with e.g. -DDEBUG=1
#ifndef DEBUG
#define DEBUG 0
#endif
#ifndef TIMERS
#define TIMERS 1
#endif
#ifndef STATIC_TIMESTEP
#define STATIC_TIMESTEP 0
#endif

// The Intel compiler is a pain
// Intel 18.0.0 aka 20170811 works
// Intel 18.0.2,3 aka 20180315 and 20180516 crash in places
// So we #if off the crashes with this flag
#ifndef INTEL_WORKAROUND

#if __INTEL_COMPILER_BUILD_DATE > 20180101
#define INTEL_WORKAROUND 1
#else
#define INTEL_WORKAROUND 0
#endif

#endif

// Default string length
#define STRLEN (2048)

// Reconstruction algorithms
#define LINEAR (0)
#define PPM    (1)
#define WENO   (2)
#define MP5    (3)

// Primitive and conserved variables
#define RHO (0)
#define UU  (1)
#define U1  (2)
#define U2  (3)
#define U3  (4)
#define B1  (5)
#define B2  (6)
#define B3  (7)
#if ELECTRONS
#define KEL  (8)
#define KTOT (9)
#define NVAR (10)
#else
#define NVAR (8)
#endif

// Centering of grid functions
#define FACE1 (0)
#define FACE2 (1)
#define CORN  (2)
#define CENT  (3)
// TODO add option to force FACE3 axisymmetric?
#define FACE3 (4)
#define NPG   (5)

// Boundaries
#define OUTFLOW  (0)
#define PERIODIC (1)
#define POLAR    (2)
#define USER     (3)

// Metric
#define MINKOWSKI (0)
#define MKS       (1)

// Diagnostic calls
#define DIAG_INIT  (0)
#define DIAG_DUMP  (1)
#define DIAG_LOG   (2)
#define DIAG_FINAL (3)

// Failure modes
// TODO find+eliminate uses
#define FAIL_UTOPRIM     (0)
#define FAIL_VCHAR_DISCR (1)
#define FAIL_COEFF_NEG   (2)
#define FAIL_COEFF_SUP   (3)
#define FAIL_GAMMA       (4)
#define FAIL_METRIC      (5)

// U to P failure modes


// Timers
#define TIMER_RECON    (1)
#define TIMER_LR_TO_F  (2)
#define TIMER_CMAX     (3)
#define TIMER_FLUX_CT  (4)
#define TIMER_UPDATE_U (5)
#define TIMER_U_TO_P   (6)
#define TIMER_FIXUP    (7)
#define TIMER_BOUND    (8)
#define TIMER_BOUND_COMMS (9) // TODO remove comms timer
#define TIMER_DIAG        (10)
#define TIMER_LR_STATE    (11)
#define TIMER_LR_PTOF     (12)
#define TIMER_LR_VCHAR    (13)
#define TIMER_LR_CMAX     (14)
#define TIMER_LR_FLUX     (15)
#define TIMER_IO          (16)
#define TIMER_RESTART     (17)
#define TIMER_CURRENT     (18)
#define TIMER_ALL         (19)
#if ELECTRONS
#define TIMER_ELECTRON_FIXUP (21)
#define TIMER_ELECTRON_HEAT  (22)
#define NUM_TIMERS           (23)
#else
#define NUM_TIMERS     (20)
#endif

/*******************************************************************************
    GLOBAL TYPES
*******************************************************************************/
typedef int    GridInt[N3+2*NG][N2+2*NG][N1+2*NG];
typedef double GridDouble[N3+2*NG][N2+2*NG][N1+2*NG];
typedef double GridVector[NDIM][N3+2*NG][N2+2*NG][N1+2*NG];
typedef double GridPrim[NVAR][N3+2*NG][N2+2*NG][N1+2*NG];

struct GridGeom {
  double gcov[NPG][NDIM][NDIM][N2+2*NG][N1+2*NG];
  double gcon[NPG][NDIM][NDIM][N2+2*NG][N1+2*NG];
  double gdet[NPG][N2+2*NG][N1+2*NG];
  double lapse[NPG][N2+2*NG][N1+2*NG];
  double conn[NDIM][NDIM][NDIM][N2+2*NG][N1+2*NG];
};

struct FluidState {
  GridPrim P;
  GridPrim U;
  GridVector ucon;
  GridVector ucov;
  GridVector bcon;
  GridVector bcov;
  GridVector jcon;
};

struct FluidFlux {
  GridPrim X1;
  GridPrim X2;
  GridPrim X3;
};

struct FluidEMF {
  GridDouble X1;
  GridDouble X2;
  GridDouble X3;
};

//struct FluidFail {
extern GridInt pflag;
extern GridInt fail_save;
extern GridInt fflag;
//};

#if DEBUG
extern struct FluidFlux preserve_F;
extern GridPrim preserve_dU;
#endif

/*******************************************************************************
    GLOBAL VARIABLES SECTION
*******************************************************************************/

// Physics parameters
extern double a;
extern double gam;
extern double Rhor;
extern double tp_over_te;

// Numerical parameters
extern double Rin, Rout, hslope;
extern double cour;
extern double dV, dx[NDIM], startx[NDIM];
extern double x1Min, x1Max, x2Min, x2Max, x3Min, x3Max;
extern double dt, dt_light;
extern double t, tf;
extern int nstep;
extern int is_restart;

// Output parameters
extern double DTd;
extern double DTf;
extern double DTl;
extern int DTr;
extern int DTp;
extern int dump_cnt;
extern double tdump, tlog;

// Diagnostics
extern double mdot, edot, ldot;
extern double mdot_eh, edot_eh, ldot_eh;
extern int icurr, jcurr, kcurr;

// Parallelism
extern int nthreads;

// Electrons
#if ELECTRONS
// TODO put these in parameters.h? Define MP/ME direct?
#define KTOTMAX (3.)
#define ME (9.1093826e-28  ) // Electron mass
#define MP (1.67262171e-24 ) // Proton mass
extern double game, gamp;
extern double fel0;
extern double tptemin, tptemax;
#endif


extern double poly_norm, poly_xt, poly_alpha, mks_smooth;


// MPI-specific stuff
extern int global_start[3];
extern int global_stop[3];

/*******************************************************************************
    MACROS
*******************************************************************************/
#define ILOOP	\
  for (int i = 0 + NG; i < N1 + NG; i++)
#define ILOOPALL \
  for (int i = 0; i < N1 + 2*NG; i++)
#define JLOOP	\
  for (int j = 0 + NG; j < N2 + NG; j++)
#define JLOOPALL \
  for (int j = 0; j < N2 + 2*NG; j++)
#define KLOOP	\
  for (int k = 0 + NG; k < N3 + NG; k++)
#define KLOOPALL \
  for (int k = 0; k < N3 + 2*NG; k++)
#define ZLOOP	\
  KLOOP JLOOP ILOOP
#define ZLOOPALL \
  KLOOPALL JLOOPALL ILOOPALL
// Transpose loops for forward-index output
#define ZLOOP_OUT \
  ILOOP JLOOP KLOOP
#define ZLOOP_TRANSPOSE \
  ILOOPALL JLOOPALL KLOOPALL

#define ISLOOP(istart,istop) \
  for (int i = (istart) + NG; i <= (istop) + NG; i++)
#define JSLOOP(jstart,jstop) \
  for (int j = (jstart) + NG; j <= (jstop) + NG; j++)
#define KSLOOP(kstart,kstop) \
  for (int k = (kstart) + NG; k <= (kstop) + NG; k++)
#define ZSLOOP(kstart,kstop,jstart,jstop,istart,istop) \
  for (int k = (kstart) + NG; k <= (kstop) + NG; k++) \
  for (int j = (jstart) + NG; j <= (jstop) + NG; j++) \
  for (int i = (istart) + NG; i <= (istop) + NG; i++)
#define ZSLOOP_REVERSE(kstart,kstop,jstart,jstop,istart,istop) \
  for (int k = (kstop) + NG; k >= (kstart) + NG; k--) \
  for (int j = (jstop) + NG; j >= (jstart) + NG; j--) \
  for (int i = (istop) + NG; i >= (istart) + NG; i--)
#define ZSLOOP_OUT(kstart,kstop,jstart,jstop,istart,istop) \
  ISLOOP(istart,istop) JSLOOP(jstart,jstop) KSLOOP(kstart,kstop)

// Loop over primitive variables
#define PLOOP for(int ip = 0; ip < NVAR; ip++)

// Loop over spacetime indices
#define DLOOP1 for (int mu = 0; mu < NDIM; mu++)
#define DLOOP2 for (int mu = 0; mu < NDIM; mu++)	\
               for (int nu = 0; nu < NDIM; nu++)

// For adding quotes to passed arguments e.g. git commit #0
#define HASH(x) #x
#define QUOTE(x) HASH(x)

// Math functions commonly mistyped
#define MY_MIN(fval1,fval2) ( ((fval1) < (fval2)) ? (fval1) : (fval2))
#define MY_MAX(fval1,fval2) ( ((fval1) > (fval2)) ? (fval1) : (fval2))
#define MY_SIGN(fval) ( ((fval) <0.) ? -1. : 1. )
#define delta(i,j) ((i == j) ? 1. : 0.)

// Convenience macros for logging under MPI
#define LOG(msg) if(DEBUG && mpi_io_proc()) {fprintf(stderr,msg); fprintf(stderr,"\n");}
#define LOGN(fmt,x) if(DEBUG && mpi_io_proc()) {fprintf(stderr,fmt,x); fprintf(stderr,"\n");}
// TODO bring the whole MPI ship down too
#define ERROR(msg) {if (mpi_io_proc()) {fprintf(stderr, msg); fprintf(stderr,"\n");} exit(-1);}

// FLAG macros are scattered through the code.  One can place a crude "watch" on a var
// by printing it here -- it will be printed several times during a step.
// eg add double sig_max = mpi_max(sigma_max(G, Stmp)); if(mpi_io_proc()) fprintf(stderr,"sig_max = %f\n",sig_max);
#define FLAG(msg) if(DEBUG) { LOG(msg); mpi_barrier(); }

/*******************************************************************************
    FUNCTION DECLARATIONS
*******************************************************************************/
// bl_coord.c
void bl_coord(const double X[NDIM], double *r, double *th);

// bounds.c
void set_mpi_bounds(struct FluidState *S);
void set_bounds(struct GridGeom *G, struct FluidState *S);
void fix_flux(struct FluidFlux *F);

// coord.c
void coord(int i, int j, int k, int loc, double *X);
void bl_coord(const double X[NDIM], double *r, double *th);
void gcov_func(double *X, double gcov[NDIM][NDIM]);
void set_dxdX(double X[NDIM], double dxdX[NDIM][NDIM]);
void set_points();
void set_grid(struct GridGeom *G);
void set_grid_loc(struct GridGeom *G, int i, int j, int k, int loc);
void zero_arrays();

// current.c
void current_calc(struct GridGeom *G, struct FluidState *S, struct FluidState *Ssave, double dtsave);
void omega_calc(struct GridGeom *G, struct FluidState *S, GridDouble *omega);

// diag.c
void reset_log_variables();
void diag_flux(struct FluidFlux *F);
void diag(struct GridGeom *G, struct FluidState *S, int call_code);
double flux_ct_divb(struct GridGeom *G, struct FluidState *S, int i, int j,
  int k);
#if DEBUG
void global_map(int iglobal, int jglobal, int kglobal, GridPrim prim);
void area_map(int i, int j, int k, GridPrim prim);
void area_map_pflag(int i, int j, int k);
void check_nan(struct FluidState *S, const char* flag);
double sigma_max(struct GridGeom *G, struct FluidState *S);
void update_f(struct FluidFlux *F, GridPrim *dU);
#endif

// electrons.c
#if ELECTRONS
void init_electrons(struct GridGeom *G, struct FluidState *S);
void heat_electrons(struct GridGeom *G, struct FluidState *Sh, struct FluidState *S);
void fixup_electrons(struct FluidState *S);
#endif

// fixup.c
void fixup(struct GridGeom *G, struct FluidState *S);
void fixup_utoprim(struct GridGeom *G, struct FluidState *S);

// fluxes.c
double get_flux(struct GridGeom *G, struct FluidState *S, struct FluidFlux *F);
void flux_ct(struct FluidFlux *F);

// hdf5_utils.c has its own header

// io.c
void init_io();
void dump(struct GridGeom *G, struct FluidState *S);
void dump_grid(struct GridGeom *G);

// metric.c
double gcon_func(double gcov[NDIM][NDIM], double gcon[NDIM][NDIM]);
void get_gcov(struct GridGeom *G, int i, int j, int loc, double gcov[NDIM][NDIM]);
void get_gcon(struct GridGeom *G, int i, int j, int loc, double gcon[NDIM][NDIM]);
void conn_func(struct GridGeom *G, int i, int j, int k);
void lower_grid(GridVector vcon, GridVector vcov, struct GridGeom *G, int i,
  int j, int k, int loc);
void raise_grid(GridVector vcov, GridVector vcon, struct GridGeom *G, int i,
  int j, int k, int loc);
void lower(double ucon[NDIM], double gcov[NDIM][NDIM], double ucov[NDIM]);
void raise(double ucov[NDIM], double gcon[NDIM][NDIM], double ucon[NDIM]);
double dot_grid(GridVector vcon, GridVector vcov, int i, int j, int k);
double dot(double vcon[NDIM], double vcov[NDIM]);
double invert(double *m, double *inv);

// mpi.c
void mpi_initialization(int argc, char *argv[]);
void mpi_finalize();
int sync_mpi_bound_X1(struct FluidState *S);
int sync_mpi_bound_X2(struct FluidState *S);
int sync_mpi_bound_X3(struct FluidState *S);
void mpi_barrier();
int mpi_nprocs();
int mpi_myrank();
double mpi_max(double f);
double mpi_min(double f);
double mpi_reduce(double f);
int mpi_reduce_int(int f);
void mpi_reduce_vector(double *vec_send, double *vec_recv, int len);
int mpi_io_proc();
void mpi_int_broadcast(int *val);
void mpi_dbl_broadcast(double *val);

// pack.c
void pack_write_scalar(double in[N3+2*NG][N2+2*NG][N1+2*NG], const char* name, hsize_t hdf5_type);
void pack_write_int(int in[N3+2*NG][N2+2*NG][N1+2*NG], const char* name);
void pack_write_vector(double in[][N3+2*NG][N2+2*NG][N1+2*NG], int len, const char* name, hsize_t hdf5_type);

void pack_write_axiscalar(double in[N2+2*NG][N1+2*NG], const char* name, hsize_t hdf5_type);
void pack_write_Gtensor(double in[NDIM][NDIM][N2+2*NG][N1+2*NG], const char* name, hsize_t hdf5_type);

// params.c
void set_core_params();
void set_param(char *key, void *data);
void read_params(char *pfname);

// phys.c
void prim_to_flux(struct GridGeom *G, struct FluidState *S, int i, int j, int k,
  int dir, int loc, GridPrim flux);
void prim_to_flux_vec(struct GridGeom *G, struct FluidState *S, int dir,
  int loc, int kstart, int kstop, int jstart, int jstop, int istart, int istop, GridPrim flux);
void bcon_calc(struct FluidState *S, int i, int j, int k);
void mhd_calc(struct FluidState *S, int i, int j, int k, int dir, double *mhd);
void get_fluid_source(struct GridGeom *G, struct FluidState *S, GridPrim *dU);
double bsq_calc(struct FluidState *S, int i, int j, int k);
void get_state(struct GridGeom *G, struct FluidState *S, int i, int j, int k,
  int loc);
void get_state_vec(struct GridGeom *G, struct FluidState *S, int loc,
  int kstart, int kstop, int jstart, int jstop, int istart, int istop);
void ucon_calc(struct GridGeom *G, struct FluidState *S, int i, int j, int k,
  int loc);
double mhd_gamma_calc(struct GridGeom *G, struct FluidState *S, int i, int j,
  int k, int loc);
void mhd_vchar(struct GridGeom *G, struct FluidState *Sr, int i, int j, int k,
  int loc, int dir, GridDouble cmax, GridDouble cmin);

// problem.c
void set_problem_params();
void init(struct GridGeom *G, struct FluidState *S);
// Boundary condition (currently used for Bondi flow)
void bound_gas_prob_x1r(int i, int j, int k, GridPrim  P, struct GridGeom *G);

// random.c
void init_random(int seed);
double get_random();

// reconstruction.c
void reconstruct(struct FluidState *S, GridPrim Pl, GridPrim Pr, int dir);

// restart.c
void restart_write(struct FluidState *S);
void restart_read(char *fname, struct FluidState *S);
int restart_init(struct GridGeom *G, struct FluidState *S);

// step.c
void step(struct GridGeom *G, struct FluidState *S);

// timing.c
void time_init();
void timer_start(int timerCode);
void timer_stop(int timerCode);
void report_performance();

// u_to_p.c
int U_to_P(struct GridGeom *G, struct FluidState *S, int i, int j, int k,
  int loc);
