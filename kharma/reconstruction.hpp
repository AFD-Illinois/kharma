/**
 * Reconstruction schemes specific to KHARMA.  Currently just WENO5, a.k.a. the best scheme.
 *
 * TODO show Parthenon the light of WENO
 */
#pragma once

#include "decs.hpp"

#include "reconstruct/plm_inline.hpp"

using namespace parthenon;

#define EPS 1.e-26

// TODO faster.  Make Parthenon Linear work.  Make merged version work

namespace Reconstruction
{

// BUILD UP (a) LINEAR MC RECONSTRUCTION

// Single-item implementation
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
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real dql = q(p, k, j, i) - q(p, k, j, i - 1);
            Real dqr = q(p, k, j, i + 1) - q(p, k, j, i);
            Real dq = mc(dql, dqr)*dqr;
            ql(p, i + 1) = q(p, k, j, i) + 0.5*dq;
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
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real dql = q(p, k, j, i) - q(p, k, j - i, i);
            Real dqr = q(p, k, j + 1, i) - q(p, k, j, i);
            Real dq = mc(dql, dqr)*dqr;
            ql(p, i) = q(p, k, j, i) + 0.5*dq;
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
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real dql = q(p, k, j, i) - q(p, k - 1, j, i);
            Real dqr = q(p, k + 1, j, i) - q(p, k, j, i);
            Real dq = mc(dql, dqr)*dqr;
            ql(p, i) = q(p, k, j, i) + 0.5*dq;
            qr(p, i) = q(p, k, j, i) - 0.5*dq;
        }
    );
  }
}

// BUILD UP WENO5 RECONSTRUCTION

// Single-element implementation
// WENO interpolation. See Tchekhovskoy et al. 2007 (T07), Shu 2011 (S11)
// Implemented by Monika Moscibrodzka
KOKKOS_INLINE_FUNCTION void weno5(const Real x1, const Real x2, const Real x3, const Real x4, const Real x5,
                                Real &lout, Real &rout)
{
    // Smoothness indicators, T07 A18 or S11 8
    Real beta[3], c1, c2;
    c1 = x1 - 2.*x2 + x3; c2 = x1 - 4.*x2 + 3.*x3;
    beta[0] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x2 - 2.*x3 + x4; c2 = x4 - x2;
    beta[1] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x3 - 2.*x4 + x5; c2 = x5 - 4.*x4 + 3.*x3;
    beta[2] = (13./12.)*c1*c1 + (1./4.)*c2*c2;

    // Nonlinear weights S11 9
    Real den[3] = {EPS + beta[0], EPS + beta[1], EPS + beta[2]};
    den[0] *= den[0]; den[1] *= den[1]; den[2] *= den[2];

    Real wtr[3] = {(1./16.)/den[0], (5./8. )/den[1], (5./16.)/den[2]};
    Real Wr = wtr[0] + wtr[1] + wtr[2];

    Real wtl[3] = {(1./16.)/den[2], (5./8. )/den[1], (5./16.)/den[0]};
    Real Wl = wtl[0] + wtl[1] + wtl[2];

    // S11 1, 2, 3
    lout = ((3./8.)*x5 - (5./4.)*x4 + (15./8.)*x3)*(wtl[0] / Wl) +
            ((-1./8.)*x4 + (3./4.)*x3 + (3./8.)*x2)*(wtl[1] / Wl) +
            ((3./8.)*x3 + (3./4.)*x2 - (1./8.)*x1)*(wtl[2] / Wl);
    rout = ((3./8.)*x1 - (5./4.)*x2 + (15./8.)*x3)*(wtr[0] / Wr) +
            ((-1./8.)*x2 + (3./4.)*x3 + (3./8.)*x4)*(wtr[1] / Wr) +
            ((3./8.)*x3 + (3./4.)*x4 - (1./8.)*x5)*(wtr[2] / Wr);
}
KOKKOS_INLINE_FUNCTION void weno5l(const Real x1, const Real x2, const Real x3, const Real x4, const Real x5,
                                Real &lout)
{
    // Smoothness indicators, T07 A18 or S11 8
    Real beta[3], c1, c2;
    c1 = x1 - 2.*x2 + x3; c2 = x1 - 4.*x2 + 3.*x3;
    beta[0] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x2 - 2.*x3 + x4; c2 = x4 - x2;
    beta[1] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x3 - 2.*x4 + x5; c2 = x5 - 4.*x4 + 3.*x3;
    beta[2] = (13./12.)*c1*c1 + (1./4.)*c2*c2;

    // Nonlinear weights S11 9
    Real den[3] = {EPS + beta[0], EPS + beta[1], EPS + beta[2]};
    den[0] *= den[0]; den[1] *= den[1]; den[2] *= den[2];

    Real wtl[3] = {(1./16.)/den[2], (5./8. )/den[1], (5./16.)/den[0]};
    Real Wl = wtl[0] + wtl[1] + wtl[2];

    // S11 1, 2, 3
    lout = ((3./8.)*x5 - (5./4.)*x4 + (15./8.)*x3)*(wtl[0] / Wl) +
            ((-1./8.)*x4 + (3./4.)*x3 + (3./8.)*x2)*(wtl[1] / Wl) +
            ((3./8.)*x3 + (3./4.)*x2 - (1./8.)*x1)*(wtl[2] / Wl);
}
KOKKOS_INLINE_FUNCTION void weno5r(const Real x1, const Real x2, const Real x3, const Real x4, const Real x5,
                                Real &rout)
{
    // Smoothness indicators, T07 A18 or S11 8
    Real beta[3], c1, c2;
    c1 = x1 - 2.*x2 + x3; c2 = x1 - 4.*x2 + 3.*x3;
    beta[0] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x2 - 2.*x3 + x4; c2 = x4 - x2;
    beta[1] = (13./12.)*c1*c1 + (1./4.)*c2*c2;
    c1 = x3 - 2.*x4 + x5; c2 = x5 - 4.*x4 + 3.*x3;
    beta[2] = (13./12.)*c1*c1 + (1./4.)*c2*c2;

    // Nonlinear weights S11 9
    Real den[3] = {EPS + beta[0], EPS + beta[1], EPS + beta[2]};
    den[0] *= den[0]; den[1] *= den[1]; den[2] *= den[2];

    Real wtr[3] = {(1./16.)/den[0], (5./8. )/den[1], (5./16.)/den[2]};
    Real Wr = wtr[0] + wtr[1] + wtr[2];

    rout = ((3./8.)*x1 - (5./4.)*x2 + (15./8.)*x3)*(wtr[0] / Wr) +
            ((-1./8.)*x2 + (3./4.)*x3 + (3./8.)*x4)*(wtr[1] / Wr) +
            ((3./8.)*x3 + (3./4.)*x4 - (1./8.)*x5)*(wtr[2] / Wr);
}

// Single-row implementations
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X1(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real lout, rout;
            weno5(q(p, k, j, i - 2),
                  q(p, k, j, i - 1),
                  q(p, k, j, i),
                  q(p, k, j, i + 1),
                  q(p, k, j, i + 2), lout, rout);
            ql(p, i + 1) = lout;
            qr(p, i) = rout;
        }
    );
  }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X2(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real lout, rout;
            weno5(q(p, k, j - 2, i),
                  q(p, k, j - 1, i),
                  q(p, k, j, i),
                  q(p, k, j + 1, i),
                  q(p, k, j + 2, i), lout, rout);
            ql(p, i) = lout;
            qr(p, i) = rout;
        }
    );
  }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X2l(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql)
{
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real lout;
            weno5l(q(p, k, j - 2, i),
                  q(p, k, j - 1, i),
                  q(p, k, j, i),
                  q(p, k, j + 1, i),
                  q(p, k, j + 2, i), lout);
            ql(p, i) = lout;
        }
    );
  }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X2r(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &qr)
{
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real rout;
            weno5r(q(p, k, j - 2, i),
                  q(p, k, j - 1, i),
                  q(p, k, j, i),
                  q(p, k, j + 1, i),
                  q(p, k, j + 2, i), rout);
            qr(p, i) = rout;
        }
    );
  }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X3(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real lout, rout;
            weno5(q(p, k - 2, j, i),
                  q(p, k - 1, j, i),
                  q(p, k, j, i),
                  q(p, k + 1, j, i),
                  q(p, k + 2, j, i), lout, rout);
            ql(p, i) = lout;
            qr(p, i) = rout;
        }
    );
  }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X3l(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql)
{
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real lout;
            weno5l(q(p, k - 2, j, i),
                  q(p, k - 1, j, i),
                  q(p, k, j, i),
                  q(p, k + 1, j, i),
                  q(p, k + 2, j, i), lout);
            ql(p, i) = lout;
        }
    );
  }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X3r(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &qr)
{
  PLOOP {
    parthenon::par_for_inner(member, il, iu,
        KOKKOS_LAMBDA_1D {
            Real rout;
            weno5r(q(p, k - 2, j, i),
                  q(p, k - 1, j, i),
                  q(p, k, j, i),
                  q(p, k + 1, j, i),
                  q(p, k + 2, j, i), rout);
            qr(p, i) = rout;
        }
    );
  }
}

// FULL-GRID WRAPPER FUNCTION
void ReconstructLR(std::shared_ptr<Container<Real>>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr, int dir, ReconstructionType recon) {
    FLAG(string_format("Reconstuct X%d", dir));
    auto& P = rc->Get("c.c.bulk.prims").data;
    auto pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    // TODO take a hard look at this when using small meshes
    int is = pmb->cellbounds.is(domain)-1, ie = pmb->cellbounds.ie(domain)+1;
    int js = pmb->cellbounds.js(domain)-1, je = pmb->cellbounds.je(domain)+1;
    int ks = pmb->cellbounds.ks(domain)-1, ke = pmb->cellbounds.ke(domain)+1;

    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(NPRIM, n1);

    pmb->par_for_outer(string_format("recon_x%d", dir), 2 * scratch_size_in_bytes, scratch_level, ks, ke, js, je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
            ScratchPad2D<Real> ql(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> qr(member.team_scratch(scratch_level), NPRIM, n1);

            // get reconstructed state on faces
            // TODO switch statements are fast... right?
            switch (recon) {
            case ReconstructionType::linear_mc:
                switch (dir) {
                case X1DIR:
                    PiecewiseLinearX1(member, k, j, is, ie, P, ql, qr);
                    break;
                case X2DIR:
                    PiecewiseLinearX2(member, k, j, is, ie, P, ql, qr);
                    break;
                case X3DIR:
                    PiecewiseLinearX3(member, k, j, is, ie, P, ql, qr);
                    break;
                }
                break;
            case ReconstructionType::weno5:
                switch (dir) {
                case X1DIR:
                    WENO5X1(member, k, j, is, ie, P, ql, qr);
                    break;
                case X2DIR:
                    WENO5X2(member, k, j, is, ie, P, ql, qr);
                    break;
                case X3DIR:
                    WENO5X3(member, k, j, is, ie, P, ql, qr);
                    break;
                }
                break;
            }

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            // Correct the Parthenon X1 flux to be how HARM expects it
            if (dir == X1DIR) {
                PLOOP {
                    parthenon::par_for_inner(member, is, ie, [&](const int i) {
                        Pr(p, k, j, i) = qr(p, i);
                        Pl(p, k, j, i) = ql(p, i + 1);
                    });
                }
            } else {
                PLOOP {
                    parthenon::par_for_inner(member, is, ie, [&](const int i) {
                        Pr(p, k, j, i) = qr(p, i);
                        Pl(p, k, j, i) = ql(p, i);
                    });
                }
            }
        }
    );
}

} // namespace Reconstruction
