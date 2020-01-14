/******************************************************************************
 *                                                                            *
 * STEP.C                                                                     *
 *                                                                            *
 * ADVANCES SIMULATION BY ONE TIMESTEP                                        *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

// Declarations
double advance_fluid(struct GridGeom *G, struct FluidState *Si,
  struct FluidState *Ss, struct FluidState *Sf, double Dt);

void step(struct GridGeom *G, struct FluidState *S)
{
  static struct FluidState *Stmp;
  static struct FluidState *Ssave;

  static int first_call = 1;
  if (first_call) {
    Stmp = calloc(1,sizeof(struct FluidState));
    Ssave = calloc(1,sizeof(struct FluidState));
    first_call = 0;
  }

  // Need both P_n and P_n+1 to calculate current
  // Work around ICC 18.0.2 bug in assigning to pointers to structs
  // TODO use pointer tricks to avoid deep copy on both compilers
#if INTEL_WORKAROUND
  memcpy(&(Ssave->P),&(S->P),sizeof(GridPrim));
#else
#pragma omp parallel for simd collapse(3)
  PLOOP ZLOOPALL Ssave->P[ip][k][j][i] = S->P[ip][k][j][i];
#endif
  LOGN("Step %d",nstep);
  FLAG("Start step");
  // TODO add back well-named flags /after/ events

  // Predictor setup
  advance_fluid(G, S, S, Stmp, 0.5*dt);
  FLAG("Advance Fluid Tmp");

#if ELECTRONS
  heat_electrons(G, S, Stmp);
  FLAG("Heat Electrons Tmp");
#endif

  // Fixup routines: smooth over outlier zones
  fixup(G, Stmp);
  FLAG("Fixup Tmp");
#if ELECTRONS
  fixup_electrons(Stmp);
  FLAG("Fixup e- Tmp");
#endif
  // Need an MPI call _before_ fixup_utop to obtain correct pflags
  set_bounds(G, Stmp);
  FLAG("First bounds Tmp");
  fixup_utoprim(G, Stmp);
  FLAG("Fixup U_to_P Tmp");
  set_bounds(G, Stmp);
  FLAG("Second bounds Tmp");

  // Corrector step
  double ndt = advance_fluid(G, S, Stmp, S, dt);
  FLAG("Advance Fluid Full");

#if ELECTRONS
  heat_electrons(G, Stmp, S);
  FLAG("Heat Electrons Full");
#endif

  fixup(G, S);
  FLAG("Fixup Full");
#if ELECTRONS
  fixup_electrons(S);
  FLAG("Fixup e- Full");
#endif
  set_bounds(G, S);
  FLAG("First bounds Full");
  fixup_utoprim(G, S);
  FLAG("Fixup U_to_P Full");
  set_bounds(G, S);
  FLAG("Second bounds Full");

  // Increment time
  t += dt;

  // If we're dumping this step, update the current
  if (t >= tdump) {
    current_calc(G, S, Ssave, dt);
  }

  // New dt proxy to choose fluid or light timestep
  double max_dt = 0, fake_dt = 0;
#if STATIC_TIMESTEP
  if(DEBUG) fake_dt = mpi_min(ndt);
  max_dt = cour*dt_light;
#else
  if(DEBUG) fake_dt = cour*dt_light;
  max_dt = mpi_min(ndt);
#endif

  // Set next timestep
  if (max_dt > SAFE * dt) {
    dt = SAFE * dt;
  } else {
    dt = max_dt;
  }

  LOGN("dt would have been %f",fake_dt);
  LOGN("Instead it is %f",dt);

}

inline double advance_fluid(struct GridGeom *G, struct FluidState *Si,
  struct FluidState *Ss, struct FluidState *Sf, double Dt)
{
  static GridPrim *dU;
  static struct FluidFlux *F;

  static int firstc = 1;
  if (firstc) {
    dU = calloc(1,sizeof(GridPrim));
    F = calloc(1,sizeof(struct FluidFlux));
    firstc = 0;
  }

  // Work around ICC 18.0.2 bug in assigning to pointers to structs
#if INTEL_WORKAROUND
  memcpy(&(Sf->P),&(Si->P),sizeof(GridPrim));
#else
#pragma omp parallel for simd collapse(3)
  PLOOP ZLOOPALL Sf->P[ip][k][j][i] = Si->P[ip][k][j][i];
#endif

  double ndt = get_flux(G, Ss, F);

//  update_f(F, dU);
//  FLAG("Got initial fluxes");

#if METRIC == MKS
  fix_flux(F);
#endif

//  update_f(F, dU);
//  FLAG("Fixed Flux");

  //Constrained transport for B
  flux_ct(F);

//  update_f(F, dU);
//  FLAG("CT Step");

  // Flux diagnostic globals
  // TODO don't compute every step, only for logs?
  diag_flux(F);

//  update_f(F, dU);
//  FLAG("Flux Diags");

  // Update Si to Sf
  timer_start(TIMER_UPDATE_U);
  get_state_vec(G, Ss, CENT, 0, N3 - 1, 0, N2 - 1, 0, N1 - 1);
  get_fluid_source(G, Ss, dU);

  // TODO skip this call if Si,Ss aliased
  get_state_vec(G, Si, CENT, 0, N3 - 1, 0, N2 - 1, 0, N1 - 1);
  prim_to_flux_vec(G, Si, 0, CENT, 0, N3 - 1, 0, N2 - 1, 0, N1 - 1, Si->U);

//  update_f(F, dU);
//  FLAG("Fixed flux. Got Si->U");

#pragma omp parallel for collapse(3)
  PLOOP ZLOOP {
    Sf->U[ip][k][j][i] = Si->U[ip][k][j][i] +
      Dt*((F->X1[ip][k][j][i] - F->X1[ip][k][j][i+1])/dx[1] +
          (F->X2[ip][k][j][i] - F->X2[ip][k][j+1][i])/dx[2] +
          (F->X3[ip][k][j][i] - F->X3[ip][k+1][j][i])/dx[3] +
          (*dU)[ip][k][j][i]);
  }
  timer_stop(TIMER_UPDATE_U);

  //FLAG("Got Sf->U");

  timer_start(TIMER_U_TO_P);
#pragma omp parallel for collapse(3)
  ZLOOP {
    pflag[k][j][i] = U_to_P(G, Sf, i, j, k, CENT);
    // This is too annoying even for debug
    //if (pflag[k][j][i] != 0) LOGN("Pflag is %d\n", pflag[k][j][i]);
  }
  timer_stop(TIMER_U_TO_P);

  //FLAG("Got Sf->P");

  // Not complete without setting four-vectors
  // Done /before/ each call
  //get_state_vec(G, Sf, CENT, 0, N3 - 1, 0, N2 - 1, 0, N1 - 1);

#pragma omp parallel for simd collapse(2)
  ZLOOPALL {
    fail_save[k][j][i] = pflag[k][j][i];
  }

  return ndt;
}
