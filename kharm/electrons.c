/******************************************************************************
 *                                                                            *
 * ELECTRONS.C                                                                *
 *                                                                            *
 * ELECTRON THERMODYNAMICS                                                    *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

#if ELECTRONS

// TODO put these in options with a default in decs.h
#define HOWES 0
#define KAWAZURA 1
#define CONSTANT 3
#define FE_MODEL KAWAZURA

void fixup_electrons_1zone(struct FluidState *S, int i, int j, int k);
void heat_electrons_1zone(struct GridGeom *G, struct FluidState *Sh, struct FluidState *S, int i, int j, int k);
double get_fel(struct GridGeom *G, struct FluidState *S, int i, int j, int k);

void init_electrons(struct GridGeom *G, struct FluidState *S)
{
  ZLOOPALL {
    // Set electron internal energy to constant fraction of internal energy
    double uel = fel0*S->P[UU][k][j][i];

    // Initialize entropies
    S->P[KTOT][k][j][i] = (gam-1.)*S->P[UU][k][j][i]*pow(S->P[RHO][k][j][i],-gam);
    S->P[KEL][k][j][i] = (game-1.)*uel*pow(S->P[RHO][k][j][i],-game);
  }

  // Necessary?  Usually called right afterward
  set_bounds(G, S);
}

// TODO merge these
void heat_electrons(struct GridGeom *G, struct FluidState *Ss, struct FluidState *Sf)
{
  timer_start(TIMER_ELECTRON_HEAT);

#pragma omp parallel for collapse(3)
  ZLOOP {
    heat_electrons_1zone(G, Ss, Sf, i, j, k);
  }

  timer_stop(TIMER_ELECTRON_HEAT);
}

inline void heat_electrons_1zone(struct GridGeom *G, struct FluidState *Ss, struct FluidState *Sf, int i, int j, int k)
{
  // Actual entropy at final time
  double kHarm = (gam-1.)*Sf->P[UU][k][j][i]/pow(Sf->P[RHO][k][j][i],gam);

  //double uel = 1./(game-1.)*S->P[KEL][k][j][i]*pow(S->P[RHO][k][j][i],game);

  double fel = get_fel(G, Ss, i, j, k);

  Sf->P[KEL][k][j][i] += (game-1.)/(gam-1.)*pow(Ss->P[RHO][k][j][i],gam-game)*fel*(kHarm - Sf->P[KTOT][k][j][i]);

  // TODO bhlight calculates Qvisc here instead of this
  //double ugHat = S->P[KTOT][k][j][i]*pow(S->P[RHO][k][j][i],gam)/(gam-1.);
  //double ugHarm = S->P[UU][k][j][i];

  // Update electron internal energy
  //uel += fel*(ugHarm - ugHat)*pow(Sh->P[RHO][k][j][i]/S->P[RHO][k][j][i],gam-game);

  // Convert back to electron entropy
  //S->P[KEL][k][j][i] = uel*(game-1.)*pow(S->P[RHO][k][j][i],-game);

  // Reset total entropy
  Sf->P[KTOT][k][j][i] = kHarm;
}

inline double get_fel(struct GridGeom *G, struct FluidState *S, int i, int j, int k)
{
  double Tpr = (gam-1.)*S->P[UU][k][j][i]/S->P[RHO][k][j][i];
  double uel = 1./(game-1.)*S->P[KEL][k][j][i]*pow(S->P[RHO][k][j][i],game);
  double Tel = (game-1.)*uel/S->P[RHO][k][j][i];
  if(Tel <= 0.) Tel = SMALL;
  if(Tpr <= 0.) Tpr = SMALL;

  double Trat = fabs(Tpr/Tel);

  double pres = S->P[RHO][k][j][i]*Tpr; // Proton pressure

  //TODO can I prevent this call?
  get_state(G, S, i, j, k, CENT);
  double bsq = bsq_calc(S, i, j, k);

  double beta = pres/bsq*2;
  if(beta > 1.e20) beta = 1.e20;

#if FE_MODEL == HOWES
  double logTrat = log10(Trat);
  double mbeta = 2. - 0.2*logTrat;

  double c1 = 0.92;
  double c2 = (Trat <= 1.) ? 1.6/Trat : 1.2/Trat;
  double c3 = (Trat <= 1.) ? 18. + 5.*logTrat : 18.;

  double beta_pow = pow(beta,mbeta);
  double qrat = c1*(c2*c2+beta_pow)/(c3*c3 + beta_pow)*exp(-1./beta)*pow(MP/ME*Trat,.5);
  double fel = 1./(1. + qrat);
#elif FE_MODEL == KAWAZURA
  double QiQe = 35./(1. + pow(beta/15.,-1.4)*exp(-0.1/Trat));
  double fel = 1./(1. + QiQe);
#elif FE_MODEL == CONSTANT
  double fel = fel0;
#endif

#if SUPPRESS_HIGHB_HEAT
  if(bsq/S->P[RHO][k][j][i] > 1.) fel = 0;
#endif

  return fel;
}

void fixup_electrons(struct FluidState *S)
{
  timer_start(TIMER_ELECTRON_FIXUP);

#pragma omp parallel for collapse(3)
  ZLOOP {
    fixup_electrons_1zone(S, i, j, k);
  }

  timer_stop(TIMER_ELECTRON_FIXUP);
}

inline void fixup_electrons_1zone(struct FluidState *S, int i, int j, int k)
{
  double kelmax = S->P[KTOT][k][j][i]*pow(S->P[RHO][k][j][i],gam-game)/(tptemin*(gam-1.)/(gamp-1.) + (gam-1.)/(game-1.));
  double kelmin = S->P[KTOT][k][j][i]*pow(S->P[RHO][k][j][i],gam-game)/(tptemax*(gam-1.)/(gamp-1.) + (gam-1.)/(game-1.));

  // Replace NANs with cold electrons
  if (isnan(S->P[KEL][k][j][i])) S->P[KEL][k][j][i] = kelmin;

  // Enforce maximum Tp/Te
  S->P[KEL][k][j][i] = MY_MAX(S->P[KEL][k][j][i], kelmin);

  // Enforce minimum Tp/Te
  S->P[KEL][k][j][i] = MY_MIN(S->P[KEL][k][j][i], kelmax);
}
#endif // ELECTRONS

