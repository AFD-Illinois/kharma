/******************************************************************************
 *                                                                            *
 * CURRENT.C                                                                  *
 *                                                                            *
 * CALCULATE CURRENT FROM FLUID VARIABLES                                     *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

double Fcon_calc(struct GridGeom *G, struct FluidState *S, int mu, int nu, int i, int j, int k);
int antisym(int a, int b, int c, int d);
int pp(int n, int *P);


static struct FluidState *Sa;

// Calculate the current
void current_calc(struct GridGeom *G, struct FluidState *S, struct FluidState *Ssave, double dtsave)
{
  timer_start(TIMER_CURRENT);

  static int first_run = 1;
  if (first_run) {
    //We only need the primitives, but this is fast
    Sa = calloc(1,sizeof(struct FluidState));
    first_run = 0;
  }

  // Calculate time-centered P
  // Intel 18.0.2 crashes at these parallel directives
#if !INTEL_WORKAROUND
#pragma omp parallel for simd collapse(3)
#endif
  PLOOP {
    ZLOOPALL {
      Sa->P[ip][k][j][i] = 0.5*(S->P[ip][k][j][i] + Ssave->P[ip][k][j][i]);
    }
  }

  // Keep all get_state calls outside the loop so it doesn't modify S{a,save}
  get_state_vec(G, S, CENT, -1, N3, -1, N2, -1, N1);
  get_state_vec(G, Ssave, CENT, -1, N3, -1, N2, -1, N1);
  get_state_vec(G, Sa, CENT, -1, N3, -1, N2, -1, N1);

#if !INTEL_WORKAROUND
#pragma omp parallel for simd collapse(3)
#endif
  DLOOP1 ZLOOPALL S->jcon[mu][k][j][i] = 0.;

  // Calculate j^{\mu} using centered differences for active zones
  // TODO rewrite this vector-style
#pragma omp parallel for collapse(3)
  ZLOOP {
    double gF0p[NDIM], gF0m[NDIM], gF1p[NDIM], gF1m[NDIM], gF2p[NDIM], gF2m[NDIM];
    double gF3p[NDIM], gF3m[NDIM];

    // Get sqrt{-g}*F^{mu nu} at neighboring points

    // X0
    DLOOP1 {
      gF0p[mu] = Fcon_calc(G, S,  0, mu, i, j, k);
      gF0m[mu] = Fcon_calc(G, Ssave, 0, mu, i, j, k);
    }

    // X1
    DLOOP1 {
      gF1p[mu] = Fcon_calc(G, Sa, 1, mu, i+1, j, k);
      gF1m[mu] = Fcon_calc(G, Sa, 1, mu, i-1, j, k);
    }

    // X2
    DLOOP1 {
      gF2p[mu] = Fcon_calc(G, Sa, 2, mu, i, j+1, k);
      gF2m[mu] = Fcon_calc(G, Sa, 2, mu, i, j-1, k);
    }

    // X3
    DLOOP1 {
      gF3p[mu] = Fcon_calc(G, Sa, 3, mu, i, j, k+1);
      gF3m[mu] = Fcon_calc(G, Sa, 3, mu, i, j, k-1);
    }

    // Difference: D_mu F^{mu nu} = 4 \pi j^nu
    DLOOP1 {
      // Extra factor of sqrt(4*PI)*J given HARM's B_unit
      S->jcon[mu][k][j][i] = (1./(sqrt(4.*M_PI)*G->gdet[CENT][j][i]))*(
                           (gF0p[mu] - gF0m[mu])/dtsave +
                           (gF1p[mu] - gF1m[mu])/(2.*dx[1]) +
                           (gF2p[mu] - gF2m[mu])/(2.*dx[2]) +
                           (gF3p[mu] - gF3m[mu])/(2.*dx[3]));
    }
  }

  timer_stop(TIMER_CURRENT);
}

// Calculate field rotation rate
void omega_calc(struct GridGeom *G, struct FluidState *S, GridDouble *omega)
{
  static GridDouble *Fcov01, *Fcov13;

  static int firstc = 1;
  if (firstc) {
    Fcov01 = calloc (1, sizeof(GridDouble));
    Fcov13 = calloc (1, sizeof(GridDouble));
    firstc = 0;
  }

  //TODO test inverting these loops, esp if allows writing to omega sooner
#pragma omp parallel for simd collapse(3)
  DLOOP2 {
    ZLOOP {
      double Fmunu = Fcon_calc(G, S, mu, nu, i, j, k);
      (*Fcov01)[k][j][i] += Fmunu*G->gcov[CENT][mu][0][j][i]*G->gcov[CENT][nu][1][j][i];
      (*Fcov13)[k][j][i] += Fmunu*G->gcov[CENT][mu][1][j][i]*G->gcov[CENT][nu][3][j][i];
    }
  }

#pragma omp parallel for simd collapse(2)
  ZLOOP {
    (*omega)[k][j][i] = (*Fcov01)[k][j][i]/(*Fcov13)[k][j][i];
  }
}

// Return mu, nu component of contravarient Maxwell tensor at grid zone i, j, k
inline double Fcon_calc(struct GridGeom *G, struct FluidState *S, int mu, int nu, int i, int j, int k)
{
  double Fcon;

  if (mu == nu) return 0.;

  //get_state(G,S,i,j,k, CENT); //This has been called

  Fcon = 0.;
  for (int kap = 0; kap < NDIM; kap++) {
    for (int lam = 0; lam < NDIM; lam++) {
      Fcon += (-1./G->gdet[CENT][j][i])*antisym(mu,nu,kap,lam)*S->ucov[kap][k][j][i]*S->bcov[lam][k][j][i];
    }
  }

  return Fcon*G->gdet[CENT][j][i];
}

// Completely antisymmetric 4D symbol
inline int antisym(int a, int b, int c, int d)
{
  // Check for valid permutation
  if (a < 0 || a > 3) return 100;
  if (b < 0 || b > 3) return 100;
  if (c < 0 || c > 3) return 100;
  if (d < 0 || d > 3) return 100;

  // Entries different? 
  if (a == b) return 0;
  if (a == c) return 0;
  if (a == d) return 0;
  if (b == c) return 0;
  if (b == d) return 0;
  if (c == d) return 0;

  // Determine parity of permutation
  int p[4] = {a, b, c, d};

  return pp(4, p);
}

// Due to Norm Hardy; good for general n
inline int pp(int n, int P[n])
{
  int x;
  int p = 0;
  int v[n];

  for (int j = 0; j < n; j++) v[j] = 0;

  for (int j = 0; j < n; j++) {
    if (v[j]) {
      p++;
    } else {
      x = j;
      do {
        x = P[x];
        v[x] = 1;
      } while (x != j);
    }
  }

  if (p % 2 == 0) {
    return 1;
  } else {
    return -1;
  }
}

