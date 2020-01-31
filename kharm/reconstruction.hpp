/**
 * Reconstruction schemes
 */

#include "decs.hpp"


KOKKOS_INLINE_FUNCTION void weno(const double x1, const double x2, const double x3,
                                const double x4, const double x5,
                                double &lout, double &rout);
// Use the pre-processor for poor man's multiple dispatch
void reconstruct(const Grid &G, const GridVars P, GridVars Pl, GridVars Pr, const int dir)
{
  if (dir == 1) {
    Kokkos::parallel_for("recon_1", G.bulk_plus_p(1),
        KOKKOS_LAMBDA_VARS
        {
            weno(P(i-2, j, k, p), P(i-1, j, k, p), P(i, j, k, p),
                 P(i+1, j, k, p), P(i+2, j, k, p),
                 Pl(i, j, k, p), Pr(i, j, k, p));
        }
    );
  } else if (dir == 2) {
    Kokkos::parallel_for("recon_2", G.bulk_plus_p(1),
        KOKKOS_LAMBDA_VARS
        {
            weno(P(i, j-2, k, p), P(i, j-1, k, p), P(i, j, k, p),
                 P(i, j+1, k, p), P(i, j+2, k, p),
                 Pl(i, j, k, p), Pr(i, j, k, p));
        }
    );
  } else if (dir == 3) {
    Kokkos::parallel_for("recon_3", G.bulk_plus_p(1),
        KOKKOS_LAMBDA_VARS
        {
            weno(P(i, j, k-2, p), P(i, j, k-1, p), P(i, j, k, p),
                 P(i, j, k+1, p), P(i, j, k+2, p),
                 Pl(i, j, k, p), Pr(i, j, k, p));
        }
    );
  } else {
      // TODO enforce by enum?
      throw std::invalid_argument("Reconstruction direction must be 1, 2, or 3");
  }
}

// WENO interpolation. See Tchekhovskoy et al. 2007 (T07), Shu 2011 (S11)
// Implemented by Monika Moscibrodzka
KOKKOS_INLINE_FUNCTION void weno(const double x1, const double x2, const double x3, const double x4, const double x5,
                                double &lout, double &rout)
{
  // S11 1, 2, 3
  double vr[3], vl[3];
  vr[0] =  (3./8.)*x1 - (5./4.)*x2 + (15./8.)*x3;
  vr[1] = (-1./8.)*x2 + (3./4.)*x3 + (3./8.)*x4;
  vr[2] =  (3./8.)*x3 + (3./4.)*x4 - (1./8.)*x5;

  vl[0] =  (3./8.)*x5 - (5./4.)*x4 + (15./8.)*x3;
  vl[1] = (-1./8.)*x4 + (3./4.)*x3 + (3./8.)*x2;
  vl[2] =  (3./8.)*x3 + (3./4.)*x2 - (1./8.)*x1;

  // Smoothness indicators, T07 A18 or S11 8
  double beta[3];
  beta[0] = (13./12.)*pow(x1 - 2.*x2 + x3, 2) +
            (1./4.)*pow(x1 - 4.*x2 + 3.*x3, 2);
  beta[1] = (13./12.)*pow(x2 - 2.*x3 + x4, 2) +
            (1./4.)*pow(x4 - x2, 2);
  beta[2] = (13./12.)*pow(x3 - 2.*x4 + x5, 2) +
            (1./4.)*pow(x5 - 4.*x4 + 3.*x3, 2);

  // Nonlinear weights S11 9
  double den, wtr[3], Wr, wr[3], wtl[3], Wl, wl[3], eps;
  eps=1.e-26;

  den = eps + beta[0]; den *= den; wtr[0] = (1./16.)/den;
  den = eps + beta[1]; den *= den; wtr[1] = (5./8. )/den;
  den = eps + beta[2]; den *= den; wtr[2] = (5./16.)/den;
  Wr = wtr[0] + wtr[1] + wtr[2];
  wr[0] = wtr[0]/Wr ;
  wr[1] = wtr[1]/Wr ;
  wr[2] = wtr[2]/Wr ;

  den = eps + beta[2]; den *= den; wtl[0] = (1./16.)/den;
  den = eps + beta[1]; den *= den; wtl[1] = (5./8. )/den;
  den = eps + beta[0]; den *= den; wtl[2] = (5./16.)/den;
  Wl = wtl[0] + wtl[1] + wtl[2];
  wl[0] = wtl[0]/Wl;
  wl[1] = wtl[1]/Wl;
  wl[2] = wtl[2]/Wl;

  lout = vl[0]*wl[0] + vl[1]*wl[1] + vl[2]*wl[2];
  rout = vr[0]*wr[0] + vr[1]*wr[1] + vr[2]*wr[2];
}

