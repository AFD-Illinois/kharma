/**
 * Reconstruction schemes specific to KHARMA.  Currently just WENO5, a.k.a. the best scheme.
 *
 * TODO show Parthenon the light of WENO
 */
#pragma once

#include "decs.hpp"

#include "reconstruct/plm_inline.hpp"

using namespace parthenon;

KOKKOS_INLINE_FUNCTION void weno5(const Real x1, const Real x2, const Real x3,
                                const Real x4, const Real x5,
                                Real &lout, Real &rout);

// TODO these could be significantly faster in X2, X3.
// Easiest to add WENO5 to Parthenon in its own style

namespace Reconstruction
{

// BUILD UP (a) LINEAR MC RECONSTRUCTION

// Single-item implementations
KOKKOS_INLINE_FUNCTION Real mc(const Real dm, const Real dp) {
  const Real r = (abs(dp) > 0. ? dm/dp : 2.0);
  return max(0.0, min(2.0, min(2*r,0.5*(1+r))));
}

// Single-row implementations
template <typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX1(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
  const int nq = q.GetDim(4) - 1;
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real dql = q(p, k, j, i) - q(p, k, j, i - 1);
            Real dqr = q(p, k, j, i + 1) - q(p, k, j, i);
            Real dq = mc(dql, dqr)*dqr;
            ql(p, i+1) = q(p, k, j, i) + 0.5*dq;
            qr(p, i) = q(p, k, j, i) - 0.5*dq;
        }
    );
  }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX2(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
  const int nq = q.GetDim(4) - 1;
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real dql = q(p, k, j, i) - q(p, k, j - i, i);
            Real dqr = q(p, k, j + 1, i) - q(p, k, j, i);
            Real dq = mc(dql, dqr)*dqr;
            ql(p, i+1) = q(p, k, j, i) + 0.5*dq;
            qr(p, i) = q(p, k, j, i) - 0.5*dq;
        }
    );
  }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX3(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
  const int nq = q.GetDim(4) - 1;
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real dql = q(p, k, j, i) - q(p, k - 1, j, i);
            Real dqr = q(p, k + 1, j, i) - q(p, k, j, i);
            Real dq = mc(dql, dqr)*dqr;
            ql(p, i+1) = q(p, k, j, i) + 0.5*dq;
            qr(p, i) = q(p, k, j, i) - 0.5*dq;
        }
    );
  }
}

// Full-grid implementations
void LinearX1(std::shared_ptr<Container<Real>>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr) {
    FLAG("Recon X1");
    auto& P = rc->Get("c.c.bulk.prims").data;
    auto pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(NPRIM, nx1);

    pmb->par_for_outer("recon_1_linear", 2 * scratch_size_in_bytes, scratch_level, ks, ke, js, je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
            ScratchPad2D<Real> ql(member.team_scratch(scratch_level), NPRIM, nx1);
            ScratchPad2D<Real> qr(member.team_scratch(scratch_level), NPRIM, nx1);
            // get reconstructed state on faces
            Reconstruction::PiecewiseLinearX1(member, k, j, is - 1, ie + 1, P, ql, qr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            PLOOP {
                parthenon::par_for_inner(member, is - 1, ie + 1, [&](const int i) {
                    Pl(p, k, j, i) = ql(p, i);
                    Pr(p, k, j, i) = qr(p, i);
                });
            }
        }
    );
}

void LinearX2(std::shared_ptr<Container<Real>>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr) {
    FLAG("Recon X2");
    auto& P = rc->Get("c.c.bulk.prims").data;
    auto pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    const int scratch_level = 1;
    size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(NPRIM, nx1);

    pmb->par_for_outer("recon_2_linear", 3 * scratch_size_in_bytes, scratch_level, ks, ke, js - 1, je + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
            // the overall algorithm/use of scratch pad here is clear inefficient and kept
            // just for demonstrating purposes. The key point is that we cannot reuse
            // reconstructed arrays for different `j` with `j` being part of the outer
            // loop given that this loop can be handled by multiple threads simultaneously.
            parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), NPRIM, nx1);
            parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), NPRIM, nx1);
            parthenon::ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), NPRIM, nx1);
            // get reconstructed state on faces
            Reconstruction::PiecewiseLinearX2(member, k, j - 1, is, ie, P, ql, q_unused);
            Reconstruction::PiecewiseLinearX2(member, k, j, is, ie, P, q_unused, qr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            PLOOP {
                parthenon::par_for_inner(member, is, ie, [&](const int i) {
                    Pl(p, k, j, i) = ql(p, i);
                    Pr(p, k, j, i) = qr(p, i);
                });
            }
        }
    );
}

void LinearX3(std::shared_ptr<Container<Real>>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr) {
    FLAG("Recon X3");
    auto& P = rc->Get("c.c.bulk.prims").data;
    auto pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    const int scratch_level = 1;
    size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(NPRIM, nx1);

    pmb->par_for_outer("recon_3_linear", 3 * scratch_size_in_bytes, scratch_level, ks - 1, ke + 1, js, je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
            // the overall algorithm/use of scratch pad here is clear inefficient and kept
            // just for demonstrating purposes. The key point is that we cannot reuse
            // reconstructed arrays for different `j` with `j` being part of the outer
            // loop given that this loop can be handled by multiple threads simultaneously.

            parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), NPRIM, nx1);
            parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), NPRIM, nx1);
            parthenon::ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), NPRIM, nx1);
            // get reconstructed state on faces
            Reconstruction::PiecewiseLinearX3(member, k - 1, j, is, ie, P, ql, q_unused);
            Reconstruction::PiecewiseLinearX3(member, k, j, is, ie, P, q_unused, qr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            PLOOP {
                parthenon::par_for_inner(member, is, ie, [&](const int i) {
                    Pl(p, k, j, i) = ql(p, i);
                    Pr(p, k, j, i) = qr(p, i);
                });
            }
        }
    );
}

// BUILD UP WENO5 RECONSTRUCTION

void WENO5X1(std::shared_ptr<Container<Real>>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr)
{
    FLAG("Recon X1");
    auto& P = rc->Get("c.c.bulk.prims").data;
    auto pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    pmb->par_for("recon_1", 0, NPRIM-1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA_VARS
        {
            weno5(P(p, k, j, i-2), P(p, k, j, i-1), P(p, k, j, i),
                  P(p, k, j, i+1), P(p, k, j, i+2),
                  Pl(p, k, j, i), Pr(p, k, j, i));
        }
    );
}
void WENO5X2(std::shared_ptr<Container<Real>>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr)
{
    FLAG("Recon X2");
    auto& P = rc->Get("c.c.bulk.prims").data;
    auto pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    pmb->par_for("recon_2", 0, NPRIM-1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA_VARS
        {
            weno5(P(p, k, j-2, i), P(p, k, j-1, i), P(p, k, j, i),
                  P(p, k, j+1, i), P(p, k, j+2, i),
                  Pl(p, k, j, i),  Pr(p, k, j, i));
        }
    );
}
void WENO5X3(std::shared_ptr<Container<Real>>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr)
{
    FLAG("Recon X3");
    auto& P = rc->Get("c.c.bulk.prims").data;
    auto pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    pmb->par_for("recon_3", 0, NPRIM-1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA_VARS
        {
            weno5(P(p, k-2, j, i), P(p, k-1, j, i), P(p, k, j, i),
                  P(p, k+1, j, i), P(p, k+2, j, i),
                  Pl(p, k, j, i),  Pr(p, k, j, i));
        }
    );
}
} // namespace Reconstruction

// WENO interpolation. See Tchekhovskoy et al. 2007 (T07), Shu 2011 (S11)
// Implemented by Monika Moscibrodzka
// Supplanted by namespaced version soon^TM
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