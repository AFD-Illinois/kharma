/******************************************************************************
 *                                                                            *
 * BOUNDS.C                                                                   *
 *                                                                            *
 * PHYSICAL BOUNDARY CONDITIONS                                               *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

// Sanity checks: grid dimensions, supported boundary conditions
#if N2 > 1 && N2 < NG
#error "N2 must be >= NG"
#elif N3 > 1 && N3 < NG
#error "N3 must be >= NG"
#endif


#if X1L_BOUND != PERIODIC && X1L_BOUND != OUTFLOW
#error "Unsupported X1L_BOUND"
#endif
#if X1R_BOUND != PERIODIC && X1R_BOUND != OUTFLOW && X1R_BOUND != USER
#error "Unsupported X1R_BOUND"
#endif

#if X2L_BOUND != PERIODIC && X2L_BOUND != OUTFLOW && X2L_BOUND != POLAR
#error "Unsupported X2L_BOUND"
#endif
#if X2R_BOUND != PERIODIC && X2R_BOUND != OUTFLOW && X2R_BOUND != POLAR
#error "Unsupported X2R_BOUND"
#endif

#if X3L_BOUND != PERIODIC && X3L_BOUND != OUTFLOW
#error "Unsupported X3L_BOUND"
#endif
#if X3R_BOUND != PERIODIC && X3R_BOUND != OUTFLOW
#error "Unsupported X3R_BOUND"
#endif

void inflow_check(struct GridGeom *G, struct FluidState *S, int i, int j,
  int k, int type);

// TODO make the pflag sync optional
void set_bounds(struct GridGeom *G, struct FluidState *S)
{
  timer_start(TIMER_BOUND);

  if(global_start[0] == 0) {
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    KLOOP {
      JLOOP {
        ISLOOP(-NG, -1) {
#if N1 < NG
          int iactive = NG;
          PLOOP S->P[ip][k][j][i] = S->P[ip][k][j][iactive];
          pflag[k][j][i] = pflag[k][j][iactive];
#elif X1L_BOUND == OUTFLOW
            int iz = 0 + NG;
            PLOOP S->P[ip][k][j][i] = S->P[ip][k][j][iz];
            pflag[k][j][i] = pflag[k][j][iz];

            double rescale = G->gdet[CENT][j][iz]/G->gdet[CENT][j][i];
            S->P[B1][k][j][i] *= rescale;
            S->P[B2][k][j][i] *= rescale;
            S->P[B3][k][j][i] *= rescale;
#endif
        }
      }
    }

#if METRIC == MKS
    if(X1L_INFLOW == 0) {
      // Make sure there is no inflow at the inner boundary
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
      KLOOP {
        JLOOP {
          ISLOOP(-NG, -1) {
            inflow_check(G, S, i, j, k, 0);
          }
        }
      }
    }
#endif

  } // global_start[0] == 0

  if(global_stop[0] == N1TOT) {
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    KLOOP {
      JLOOP {
        ISLOOP(N1, N1 - 1 + NG) {
#if N1 < NG
          int iactive = N1 - 1 + NG;
          PLOOP S->P[ip][k][j][i] = S->P[ip][k][j][iactive];
          pflag[k][j][i] = pflag[k][j][iactive];
#elif X1R_BOUND == OUTFLOW
          int iz = N1 - 1 + NG;
          PLOOP S->P[ip][k][j][i] = S->P[ip][k][j][iz];
          pflag[k][j][i] = pflag[k][j][iz];

          double rescale = G->gdet[CENT][j][iz]/G->gdet[CENT][j][i];
          S->P[B1][k][j][i] *= rescale;
          S->P[B2][k][j][i] *= rescale;
          S->P[B3][k][j][i] *= rescale;
#elif X1R_BOUND == USER
          bound_gas_prob_x1r(i, j, k, S->P, G);
#endif
        }
      }
    }

#if METRIC == MKS
    if(X1R_INFLOW == 0) {
      // Make sure there is no inflow at the outer boundary
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
      KLOOP {
        JLOOP {
          ISLOOP(N1, N1 - 1 + NG) {
            inflow_check(G, S, i, j, k, 1);
          }
        }
      }
    }
#endif

  } // global_stop[0] == N1TOT

  timer_start(TIMER_BOUND_COMMS);
  sync_mpi_bound_X1(S);
  timer_stop(TIMER_BOUND_COMMS);

  if(global_start[1] == 0) {
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    KLOOP {
      ILOOPALL {
        JSLOOP(-NG, -1) {
#if N2 < NG
          int jactive = NG;
          PLOOP S->P[ip][k][j][i] = S->P[ip][k][jactive][i];
          pflag[k][j][i] = pflag[k][jactive][i];
#elif X2L_BOUND == OUTFLOW
          int jz = 0 + NG ;
          PLOOP S->P[ip][k][j][i] = S->P[ip][k][jz][i];
          pflag[k][j][i] = pflag[k][jz][i];
#elif X2L_BOUND == POLAR
          // Reflect the zone past NG by NG-j
          int jrefl = NG + (NG - j) - 1;
          PLOOP S->P[ip][k][j][i] = S->P[ip][k][jrefl][i];
          pflag[k][j][i] = pflag[k][jrefl][i];
          S->P[U2][k][j][i] *= -1.;
          S->P[B2][k][j][i] *= -1.;
#endif
        }
      }
    }

  } // global_start[1] == 0

  if(global_stop[1] == N2TOT) {
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    KLOOP {
      ILOOPALL {
        JSLOOP(N2, N2-1+NG) {
#if N2 < NG
          int jactive = N2 - 1 + NG;
          PLOOP S->P[ip][k][j][i] = S->P[ip][k][jactive][i];
          pflag[k][j][i] = pflag[k][jactive][i];
#elif X2R_BOUND == OUTFLOW
          int jz = N2 - 1 + NG;
          PLOOP S->P[ip][k][j][i] = S->P[ip][k][jz][i];
          pflag[k][j][i] = pflag[k][jz][i];
#elif X2R_BOUND == POLAR
          // As j grows beyond N2+NG, reflect the zone that far previous
          int jrefl = (N2 + NG) + (N2 + NG - j) - 1;
          PLOOP S->P[ip][k][j][i] = S->P[ip][k][jrefl][i];
          pflag[k][j][i] = pflag[k][jrefl][i];
          S->P[U2][k][j][i] *= -1.;
          S->P[B2][k][j][i] *= -1.;
#endif
        }
      }
    }
  } // global_stop[1] == N2TOT


  timer_start(TIMER_BOUND_COMMS);
  sync_mpi_bound_X2(S);
  timer_stop(TIMER_BOUND_COMMS);

  if (global_start[2] == 0) {
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    JLOOPALL {
      ILOOPALL {
        KSLOOP(-NG, -1) {
#if N3 < NG
          int kactive = NG;
          PLOOP S->P[ip][k][j][i] = S->P[ip][kactive][j][i];
          pflag[k][j][i] = pflag[kactive][j][i];
#elif X3L_BOUND == OUTFLOW
          int kz = 0 + NG ;
          PLOOP S->P[ip][k][j][i] = S->P[ip][kz][j][i];
          pflag[k][j][i] = pflag[kz][j][i];
#endif
        }
      }
    }
  } // global_start[2] == 0

  if(global_stop[2] == N3TOT) {
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    JLOOPALL {
      ILOOPALL {
        KSLOOP(N3, N3-1+NG) {
#if N3 < NG
          int kactive = N3-1+NG;
          PLOOP S->P[ip][k][j][i] = S->P[ip][kactive][j][i];
          pflag[k][j][i] = pflag[kactive][j][i];
#elif X3R_BOUND == OUTFLOW
          int kz = N3 - 1 + NG;
          PLOOP S->P[ip][k][j][i] = S->P[ip][kz][j][i];
          pflag[k][j][i] = pflag[kz][j][i];
#endif
        }
      }
    }
  } // global_stop[2] == N3TOT

  timer_start(TIMER_BOUND_COMMS);
  sync_mpi_bound_X3(S);
  timer_stop(TIMER_BOUND_COMMS);

  timer_stop(TIMER_BOUND);

}

#if METRIC == MKS
void inflow_check(struct GridGeom *G, struct FluidState *S, int i, int j, int k,
  int type)
{
  double alpha, beta1, vsq;

  ucon_calc(G, S, i, j, k, CENT);

  if (((S->ucon[1][k][j][i] > 0.) && (type == 0)) ||
      ((S->ucon[1][k][j][i] < 0.) && (type == 1)))
  {
    // Find gamma and remove it from S->Pitives
    // TODO check failures in a vectorization-friendly way
    double gamma = mhd_gamma_calc(G, S, i, j, k, CENT);
    S->P[U1][k][j][i] /= gamma;
    S->P[U2][k][j][i] /= gamma;
    S->P[U3][k][j][i] /= gamma;
    alpha = G->lapse[CENT][j][i];
    beta1 = G->gcon[CENT][0][1][j][i]*alpha*alpha;

    // Reset radial velocity so radial 4-velocity is zero
    S->P[U1][k][j][i] = beta1/alpha;

    // Now find new gamma and put it back in
    vsq = 0.;
    for (int mu = 1; mu < NDIM; mu++) {
      for (int nu = 1; nu < NDIM; nu++) {
        vsq += G->gcov[CENT][mu][nu][j][i]*S->P[U1+mu-1][k][j][i]*S->P[U1+nu-1][k][j][i];
      }
    }
    if (fabs(vsq) < 1.e-13)
      vsq = 1.e-13;
    if (vsq >= 1.) {
      vsq = 1. - 1./(GAMMAMAX*GAMMAMAX);
    }
    gamma = 1./sqrt(1. - vsq);
    S->P[U1][k][j][i] *= gamma;
    S->P[U2][k][j][i] *= gamma;
    S->P[U3][k][j][i] *= gamma;
  }
}

void fix_flux(struct FluidFlux *F)
{
  if (global_start[0] == 0 && X1L_INFLOW == 0) {
  // TODO these crash Intel 18.0.2
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    KLOOPALL {
      JLOOPALL {
        F->X1[RHO][k][j][0+NG] = MY_MIN(F->X1[RHO][k][j][0+NG], 0.);
      }
    }
  }

  if (global_stop[0] == N1TOT  && X1R_INFLOW == 0) {
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    KLOOPALL {
      JLOOPALL {
        F->X1[RHO][k][j][N1+NG] = MY_MAX(F->X1[RHO][k][j][N1+NG], 0.);
      }
    }
  }

  if (global_start[1] == 0) {
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    KLOOPALL {
      ILOOPALL {
        F->X1[B2][k][-1+NG][i] = -F->X1[B2][k][0+NG][i];
        F->X3[B2][k][-1+NG][i] = -F->X3[B2][k][0+NG][i];
        PLOOP F->X2[ip][k][0+NG][i] = 0.;
      }
    }
  }

  if (global_stop[1] == N2TOT) {
#if !INTEL_WORKAROUND
#pragma omp parallel for collapse(2)
#endif
    KLOOPALL {
      ILOOPALL {
        F->X1[B2][k][N2+NG][i] = -F->X1[B2][k][N2-1+NG][i];
        F->X3[B2][k][N2+NG][i] = -F->X3[B2][k][N2-1+NG][i];
        PLOOP F->X2[ip][k][N2+NG][i] = 0.;
      }
    }
  }
}
#endif // METRIC
