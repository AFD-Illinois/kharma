/**
 * Reconstruction schemes specific to KHARMA.  Currently just WENO5, a.k.a. the best scheme.
 *
 * TODO show Parthenon the light of WENO
 */
#pragma once

#include "athena.hpp"

#include "decs.hpp"

using namespace parthenon;

KOKKOS_INLINE_FUNCTION void weno5(const Real x1, const Real x2, const Real x3,
                                const Real x4, const Real x5,
                                Real &lout, Real &rout);

// TODO these could be significantly faster in X2, X3.
// Easiest to add WENO5 to Parthenon in its own style

void WENO5X1(Container<Real>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr)
{
    FLAG("Recon X1");
    auto& P = rc.Get("c.c.bulk.prims").data;
    auto pmb = rc.pmy_block;

    pmb->par_for("recon_1", 0, NPRIM-1, pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
        KOKKOS_LAMBDA_VARS
        {
            weno5(P(p, k, j, i-2), P(p, k, j, i-1), P(p, k, j, i),
                  P(p, k, j, i+1), P(p, k, j, i+2),
                  Pl(p, k, j, i), Pr(p, k, j, i));
        }
    );
}
void WENO5X2(Container<Real>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr)
{
    FLAG("Recon X2");
    auto& P = rc.Get("c.c.bulk.prims").data;
    auto pmb = rc.pmy_block;

    pmb->par_for("recon_2", 0, NPRIM-1, pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
        KOKKOS_LAMBDA_VARS
        {
            weno5(P(p, k, j-2, i), P(p, k, j-1, i), P(p, k, j, i),
                  P(p, k, j+1, i), P(p, k, j+2, i),
                  Pl(p, k, j, i),  Pr(p, k, j, i));
        }
    );
}
void WENO5X3(Container<Real>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr)
{
    FLAG("Recon X3");
    auto& P = rc.Get("c.c.bulk.prims").data;
    auto pmb = rc.pmy_block;

    pmb->par_for("recon_3", 0, NPRIM-1, pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
        KOKKOS_LAMBDA_VARS
        {
            weno5(P(p, k-2, j, i), P(p, k-1, j, i), P(p, k, j, i),
                  P(p, k+1, j, i), P(p, k+2, j, i),
                  Pl(p, k, j, i),  Pr(p, k, j, i));
        }
    );
}

// WENO interpolation. See Tchekhovskoy et al. 2007 (T07), Shu 2011 (S11)
// Implemented by Monika Moscibrodzka
KOKKOS_INLINE_FUNCTION void weno5(const Real x1, const Real x2, const Real x3, const Real x4, const Real x5,
                                Real &lout, Real &rout)
{
  // S11 1, 2, 3
  Real vr[3], vl[3];
  vr[0] =  (3./8.)*x1 - (5./4.)*x2 + (15./8.)*x3;
  vr[1] = (-1./8.)*x2 + (3./4.)*x3 + (3./8.)*x4;
  vr[2] =  (3./8.)*x3 + (3./4.)*x4 - (1./8.)*x5;

  vl[0] =  (3./8.)*x5 - (5./4.)*x4 + (15./8.)*x3;
  vl[1] = (-1./8.)*x4 + (3./4.)*x3 + (3./8.)*x2;
  vl[2] =  (3./8.)*x3 + (3./4.)*x2 - (1./8.)*x1;

  // Smoothness indicators, T07 A18 or S11 8
  Real beta[3];
  beta[0] = (13./12.)*pow(x1 - 2.*x2 + x3, 2) +
            (1./4.)*pow(x1 - 4.*x2 + 3.*x3, 2);
  beta[1] = (13./12.)*pow(x2 - 2.*x3 + x4, 2) +
            (1./4.)*pow(x4 - x2, 2);
  beta[2] = (13./12.)*pow(x3 - 2.*x4 + x5, 2) +
            (1./4.)*pow(x5 - 4.*x4 + 3.*x3, 2);

  // Nonlinear weights S11 9
  Real den, wtr[3], Wr, wr[3], wtl[3], Wl, wl[3], eps;
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

