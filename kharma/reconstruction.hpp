/* 
 *  File: reconstruction.hpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *  
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include "decs.hpp"

#include "reconstruct/dc_inline.hpp"
#include "reconstruct/plm_inline.hpp"

using namespace parthenon;

#define EPS 1.e-26

/**
 * This namespace covers custom new reconstructions for KHARMA, and a function which
 * automatically chooses the inner loop based on an enum from decs.hpp
 */
namespace KReconstruction
{
// BUILD UP (a) LINEAR MC RECONSTRUCTION
// TODO left/right splits?

// Single-item implementation
KOKKOS_INLINE_FUNCTION Real mc(const Real dm, const Real dp)
{
    const Real r = (abs(dp) > 0. ? dm/dp : 2.0);
    return max(0.0, min(2.0, min(2*r,0.5*(1+r))));
}

// Single-row implementations
// Note that "L" and "R" refer to the sides of the *face*
// ql(1) is to the right of the first zone center, but corresponds to the face value reconstructed from the left
// qr(1) is then the value at that face reconstructed from the right
template <typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX1(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
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
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA_1D {
                Real dql = q(p, k, j, i) - q(p, k, j - 1, i);
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
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
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

// Single-element implementation: "left" and "right" here are relative to zone centers, so the combo calls will switch them later.
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

// Row-wise implementations
// Note that "L" and "R" refer to the sides of the *face*
// ql(1) is to the right of the first zone center, but corresponds to the face value reconstructed from the left
// qr(1) is then the value at that face reconstructed from the right
// This is *opposite* the single-zone convention (or rather, offset from it to the faces), so weirdly WENO5X2l calls weno5r.  Get it?
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X1(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA_1D {
                Real lout, rout;
                weno5(q(p, k, j, i - 2),
                    q(p, k, j, i - 1),
                    q(p, k, j, i),
                    q(p, k, j, i + 1),
                    q(p, k, j, i + 2), lout, rout);
                ql(p, i+1) = rout;
                qr(p, i) = lout;
            }
        );
    }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X2(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA_1D {
                Real lout, rout;
                weno5(q(p, k, j - 2, i),
                    q(p, k, j - 1, i),
                    q(p, k, j, i),
                    q(p, k, j + 1, i),
                    q(p, k, j + 2, i), lout, rout);
                ql(p, i) = rout;
                qr(p, i) = lout;
            }
        );
    }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X2l(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql)
{
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA_1D {
                Real rout;
                weno5r(q(p, k, j - 2, i),
                    q(p, k, j - 1, i),
                    q(p, k, j, i),
                    q(p, k, j + 1, i),
                    q(p, k, j + 2, i), rout);
                ql(p, i) = rout;
            }
        );
    }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X2r(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &qr)
{
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA_1D {
                Real lout;
                weno5l(q(p, k, j - 2, i),
                    q(p, k, j - 1, i),
                    q(p, k, j, i),
                    q(p, k, j + 1, i),
                    q(p, k, j + 2, i), lout);
                qr(p, i) = lout;
            }
        );
    }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X3(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA_1D {
                Real lout, rout;
                weno5(q(p, k - 2, j, i),
                    q(p, k - 1, j, i),
                    q(p, k, j, i),
                    q(p, k + 1, j, i),
                    q(p, k + 2, j, i), lout, rout);
                ql(p, i) = rout;
                qr(p, i) = lout;
            }
        );
    }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X3l(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &ql)
{
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA_1D {
                Real rout;
                weno5r(q(p, k - 2, j, i),
                    q(p, k - 1, j, i),
                    q(p, k, j, i),
                    q(p, k + 1, j, i),
                    q(p, k + 2, j, i), rout);
                ql(p, i) = rout;
            }
        );
    }
}
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X3r(parthenon::team_mbr_t const &member, const int k, const int j,
                       const int il, const int iu, const T &q, ScratchPad2D<Real> &qr)
{
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA_1D {
                Real lout;
                weno5l(q(p, k - 2, j, i),
                    q(p, k - 1, j, i),
                    q(p, k, j, i),
                    q(p, k + 1, j, i),
                    q(p, k + 2, j, i), lout);
                qr(p, i) = lout;
            }
        );
    }
}

/**
 * Call the correct reconstruction 
 */
template <typename T>
KOKKOS_INLINE_FUNCTION void reconstruct(const ReconstructionType& recon, parthenon::team_mbr_t& member, const Coordinates_t& G, const T &P,
                                        const int& n1, const int& k, const int& j, const int& is_l, const int& ie_l, const int& dir,
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    int nvar = P.GetDim(4);
    int scratch_level = 1;
    ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), nvar, n1);
    // Extra scratch space for Parthenon's VL limiter stuff
    ScratchPad2D<Real> qc(member.team_scratch(scratch_level), nvar, n1);
    ScratchPad2D<Real> dql(member.team_scratch(scratch_level), nvar, n1);
    ScratchPad2D<Real> dqr(member.team_scratch(scratch_level), nvar, n1);
    ScratchPad2D<Real> dqm(member.team_scratch(scratch_level), nvar, n1);

    // Get reconstructed state on faces
    // Note the switch statement here is in the outer loop, or it would be a performance concern
    switch (recon) {
    case ReconstructionType::donor_cell:
        switch (dir) {
        case X1DIR:
            DonorCellX1(member, k, j, is_l, ie_l, P, ql, qr);
            break;
        case X2DIR:
            DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
            DonorCellX2(member, k, j, is_l, ie_l, P, q_unused, qr);
            break;
        case X3DIR:
            DonorCellX3(member, k - 1, j, is_l, ie_l, P, ql, q_unused);
            DonorCellX3(member, k, j, is_l, ie_l, P, q_unused, qr);
            break;
        }
        break;
    case ReconstructionType::linear_vl:
        switch (dir) {
        case X1DIR:
            PiecewiseLinearX1(member, k, j, is_l, ie_l, G, P, ql, qr, qc, dql, dqr, dqm);
            break;
        case X2DIR:
            PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, G, P, ql, q_unused, qc, dql, dqr, dqm);
            PiecewiseLinearX2(member, k, j, is_l, ie_l, G, P, q_unused, qr, qc, dql, dqr, dqm);
            break;
        case X3DIR:
            PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, G, P, ql, q_unused, qc, dql, dqr, dqm);
            PiecewiseLinearX3(member, k, j, is_l, ie_l, G, P, q_unused, qr, qc, dql, dqr, dqm);
            break;
        }
        break;
    case ReconstructionType::linear_mc:
        switch (dir) {
        case X1DIR:
            KReconstruction::PiecewiseLinearX1(member, k, j, is_l, ie_l, P, ql, qr);
            break;
        case X2DIR:
            KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
            KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_unused, qr);
            break;
        case X3DIR:
            KReconstruction::PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, P, ql, q_unused);
            KReconstruction::PiecewiseLinearX3(member, k, j, is_l, ie_l, P, q_unused, qr);
            break;
        }
        break;
    case ReconstructionType::weno5:
        switch (dir) {
        case X1DIR:
            KReconstruction::WENO5X1(member, k, j, is_l, ie_l, P, ql, qr);
            break;
        case X2DIR:
            KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
            KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
            break;
        case X3DIR:
            KReconstruction::WENO5X3l(member, k - 1, j, is_l, ie_l, P, ql);
            KReconstruction::WENO5X3r(member, k, j, is_l, ie_l, P, qr);
            break;
        }
        break;
        // TODO if this is even useful, pass in the copious extra things it needs
    // case ReconstructionType::weno5_lower_poles:
    //     switch (dir) {
    //     case X1DIR:
    //         KReconstruction::WENO5X1(member, k, j, is_l, ie_l, P, ql, qr);
    //         break;
    //     case X2DIR:
    //         // This prioritizes calculating fluxes for the same *zone* with the same *algorithm*,
    //         // mostly to mirror iharm3d.  One could imagine prioritizing matching faces, instead
    //         if (is_inner_x2 && j == js + 1) {
    //             DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
    //             //DonorCellX2(member, k, j, is_l, ie_l, P, q_unused, qr);
    //             KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_unused, qr);
    //         } else if (is_inner_x2 && j == js + 2) {
    //             KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
    //             //KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_unused, qr);
    //             KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
    //         } else if (is_outer_x2 && j == je - 1) {
    //             KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
    //             //KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
    //             KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_unused, qr);
    //         } else if (is_outer_x2 && j == je) {
    //             KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
    //             //DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
    //             DonorCellX2(member, k, j, is_l, ie_l, P, q_unused, qr);
    //         } else {
    //             KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
    //             KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
    //         }
    //         break;
    //     case X3DIR:
    //         KReconstruction::WENO5X3l(member, k - 1, j, is_l, ie_l, P, ql);
    //         KReconstruction::WENO5X3r(member, k, j, is_l, ie_l, P, qr);
    //         break;
    //     }
    //     break;
    }
}

} // namespace KReconstruction
