/**
 * Apply physical and periodic boundary conditions
 */
#pragma once

#include "decs.hpp"
#include "grid.hpp"

// TODO eliminate duplicate code AND somehow split doing left vs right
void periodic_x1(const Grid& G, GridVars P, GridInt pflag)
{
    Kokkos::parallel_for("periodic_x1_l", G.bound_x1_l(),
        KOKKOS_LAMBDA_3D {
            int iz = i + G.n1;

            for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(iz, j, k, p);
            pflag(i, j, k) = pflag(iz, j, k);
        }
    );
    Kokkos::parallel_for("periodic_x1_r", G.bound_x1_r(),
        KOKKOS_LAMBDA_3D {
            int iz = i - G.n1;

            for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(iz, j, k, p);
            pflag(i, j, k) = pflag(iz, j, k);
        }
    );
}
void periodic_x2(const Grid& G, GridVars P, GridInt pflag)
{
    Kokkos::parallel_for("periodic_x2_l", G.bound_x2_l(),
        KOKKOS_LAMBDA_3D {
            int jz = j + G.n2;

            for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(i, jz, k, p);
            pflag(i, j, k) = pflag(i, jz, k);
        }
    );
    Kokkos::parallel_for("periodic_x2_r", G.bound_x2_r(),
        KOKKOS_LAMBDA_3D {
            int jz = j - G.n2;

            for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(i, jz, k, p);
            pflag(i, j, k) = pflag(i, jz, k);
        }
    );
}

void periodic_x3(const Grid& G, GridVars P, GridInt pflag)
{
    Kokkos::parallel_for("periodic_x3_l", G.bound_x3_l(),
        KOKKOS_LAMBDA_3D {
            int kz = k + G.n3;

            for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(i, j, kz, p);
            pflag(i, j, k) = pflag(i, j, kz);
        }
    );
    Kokkos::parallel_for("periodic_x3_r", G.bound_x3_r(),
        KOKKOS_LAMBDA_3D {
            int kz = k - G.n3;

            for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(i, j, kz, p);
            pflag(i, j, k) = pflag(i, j, kz);
        }
    );
}

void outflow_x1(const Grid& G, GridVars P, GridInt pflag)
{
    Kokkos::parallel_for("outflow_x1_l", G.bound_x1_l(),
        KOKKOS_LAMBDA_3D {
            int iz = G.ng;

            for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(iz, j, k, p);
            pflag(i, j, k) = pflag(iz, j, k);

            double rescale = G.gdet(Loci::center, iz, j) / G.gdet(Loci::center, i, j);
            P(i, j, k, prims::B1) *= rescale;
            P(i, j, k, prims::B2) *= rescale;
            P(i, j, k, prims::B3) *= rescale;
        }
    );
    Kokkos::parallel_for("outflow_x1_r", G.bound_x1_r(),
        KOKKOS_LAMBDA_3D {
            int iz = G.n1 + G.ng - 1;

            for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(iz, j, k, p);
            pflag(i, j, k) = pflag(iz, j, k);

            double rescale = G.gdet(Loci::center, iz, j) / G.gdet(Loci::center, i, j);
            P(i, j, k, prims::B1) *= rescale;
            P(i, j, k, prims::B2) *= rescale;
            P(i, j, k, prims::B3) *= rescale;
        }
    );
}

void polar_x2(const Grid& G, GridVars P, GridInt pflag)
{
    Kokkos::parallel_for("reflect_x2_l", G.bound_x2_l(),
        KOKKOS_LAMBDA_3D {
          // Reflect across NG.  The zone j is (NG-j) prior to reflection,
          // set it equal to the zone that far *beyond* NG
          int jrefl = G.ng + (G.ng - j) - 1;
          for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(i, jrefl, k, p);
          pflag(i, j, k) = pflag(i, jrefl, k);

          // TODO These are suspect...
          P(i, j, k, prims::u2) *= -1.;
          P(i, j, k, prims::B2) *= -1.;
        }
    );
    Kokkos::parallel_for("reflect_x2_r", G.bound_x2_r(),
        KOKKOS_LAMBDA_3D {
          // Reflect across (NG+N2).  The zone j is (j - (NG+N2)) after reflection,
          // set it equal to the zone that far *before* (NG+N2)
          int jrefl = (G.ng + G.n2) - (j - (G.ng + G.n2)) - 1;
          for(int p=0; p<G.nvar; ++p) P(i, j, k, p) = P(i, jrefl, k, p);
          pflag(i, j, k) = pflag(i, jrefl, k);

          // TODO These are suspect...
          P(i, j, k, prims::u2) *= -1.;
          P(i, j, k, prims::B2) *= -1.;
        }
    );
}


void set_bounds(const Grid& G, GridVars P, GridInt pflag, Parameters params)
{
    periodic_x1(G, P, pflag);
    periodic_x2(G, P, pflag);
    periodic_x3(G, P, pflag);
}

#if 0
void inflow_check(struct GridGeom *G, struct FluidState *S, int i, int j, int k,
  int type)
{
  double alpha, beta1, vsq;

  ucon_calc(G, S, i, j, k, CENT);

  if (((ucon[1][k][j][i] > 0.) && (type == 0)) ||
      ((ucon[1][k][j][i] < 0.) && (type == 1)))
  {
    // Find gamma and remove it from Pitives
    // TODO check failures in a vectorization-friendly way
    double gamma = mhd_gamma_calc(G, S, i, j, k, CENT);
    P[U1][k][j][i] /= gamma;
    P[U2][k][j][i] /= gamma;
    P[U3][k][j][i] /= gamma;
    alpha = G.lapse[CENT][j][i];
    beta1 = G.gcon[CENT][0][1][j][i]*alpha*alpha;

    // Reset radial velocity so radial 4-velocity is zero
    P[U1][k][j][i] = beta1/alpha;

    // Now find new gamma and put it back in
    vsq = 0.;
    for (int mu = 1; mu < NDIM; mu++) {
      for (int nu = 1; nu < NDIM; nu++) {
        vsq += G.gcov[CENT][mu][nu][j][i]*P[U1+mu-1][k][j][i]*P[U1+nu-1][k][j][i];
      }
    }
    if (fabs(vsq) < 1.e-13)
      vsq = 1.e-13;
    if (vsq >= 1.) {
      vsq = 1. - 1./(GAMMAMAX*GAMMAMAX);
    }
    gamma = 1./sqrt(1. - vsq);
    P[U1][k][j][i] *= gamma;
    P[U2][k][j][i] *= gamma;
    P[U3][k][j][i] *= gamma;
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
