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

// Single-item implementation
KOKKOS_INLINE_FUNCTION Real mc(const Real dm, const Real dp)
{
    const Real r = (m::abs(dp) > 0. ? dm/dp : 2.0);
    return m::max(0.0, m::min(2.0, m::min(2*r,0.5*(1+r))));
}

// Single-row implementations
// Note that "L" and "R" refer to the sides of the *face*
// ql(1) is to the right of the first zone center, but corresponds to the face value reconstructed from the left
// qr(1) is then the value at that face reconstructed from the right
template <typename T>
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX1(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
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
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX2(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
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
KOKKOS_INLINE_FUNCTION void PiecewiseLinearX3(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T& q, ScratchPad2D<Real> &ql,
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

#pragma omp declare simd
KOKKOS_FORCEINLINE_FUNCTION
Real mc(const Real dm, const Real dp, const Real alpha) {
  const Real dc = (dm * dp > 0.0) * 0.5 * (dm + dp);
  return std::copysign(
      std::min(std::fabs(dc), alpha * std::min(std::fabs(dm), std::fabs(dp))), dc);
}

// Single-element implementation: "left" and "right" here are relative to zone centers, so the combo calls will switch them later.
// WENO interpolation. See Tchekhovskoy et al. 2007 (T07), Shu 2011 (S11)
// Implemented by Monika Moscibrodzka
KOKKOS_FORCEINLINE_FUNCTION void weno5(const Real& q0, const Real& q1, const Real& q2, const Real& q3, const Real& q4,
                                Real &qr, Real &ql)
{
  constexpr Real w5alpha[3][3] = {{1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0},
                                  {-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0},
                                  {1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0}};
  constexpr Real w5gamma[3] = {0.1, 0.6, 0.3};
  constexpr Real eps = 1e-100;
  constexpr Real thirteen_thirds = 13.0 / 3.0;

  Real a = q0 - 2 * q1 + q2;
  Real b = q0 - 4.0 * q1 + 3.0 * q2;
  Real beta0 = thirteen_thirds * a * a + b * b + eps;
  a = q1 - 2.0 * q2 + q3;
  b = q3 - q1;
  Real beta1 = thirteen_thirds * a * a + b * b + eps;
  a = q2 - 2.0 * q3 + q4;
  b = q4 - 4.0 * q3 + 3.0 * q2;
  Real beta2 = thirteen_thirds * a * a + b * b + eps;
  const Real tau5 = std::fabs(beta2 - beta0);

  beta0 = (beta0 + tau5) / beta0;
  beta1 = (beta1 + tau5) / beta1;
  beta2 = (beta2 + tau5) / beta2;

  Real w0 = w5gamma[0] * beta0 + eps;
  Real w1 = w5gamma[1] * beta1 + eps;
  Real w2 = w5gamma[2] * beta2 + eps;
  Real wsum = 1.0 / (w0 + w1 + w2);
  ql = w0 * (w5alpha[0][0] * q0 + w5alpha[0][1] * q1 + w5alpha[0][2] * q2);
  ql += w1 * (w5alpha[1][0] * q1 + w5alpha[1][1] * q2 + w5alpha[1][2] * q3);
  ql += w2 * (w5alpha[2][0] * q2 + w5alpha[2][1] * q3 + w5alpha[2][2] * q4);
  ql *= wsum;
  const Real alpha_l =
      3.0 * wsum * w0 * w1 * w2 /
          (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
      eps;

  w0 = w5gamma[0] * beta2 + eps;
  w1 = w5gamma[1] * beta1 + eps;
  w2 = w5gamma[2] * beta0 + eps;
  wsum = 1.0 / (w0 + w1 + w2);
  qr = w0 * (w5alpha[0][0] * q4 + w5alpha[0][1] * q3 + w5alpha[0][2] * q2);
  qr += w1 * (w5alpha[1][0] * q3 + w5alpha[1][1] * q2 + w5alpha[1][2] * q1);
  qr += w2 * (w5alpha[2][0] * q2 + w5alpha[2][1] * q1 + w5alpha[2][2] * q0);
  qr *= wsum;
  const Real alpha_r =
      3.0 * wsum * w0 * w1 * w2 /
          (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
      eps;

  Real dq = q3 - q2;
  dq = mc(q2 - q1, dq, 2.0);

  const Real alpha_lin = 2.0 * alpha_l * alpha_r / (alpha_l + alpha_r);
  ql = alpha_lin * ql + (1.0 - alpha_lin) * (q2 + 0.5 * dq);
  qr = alpha_lin * qr + (1.0 - alpha_lin) * (q2 - 0.5 * dq);
}
KOKKOS_FORCEINLINE_FUNCTION void weno5l(const Real& q0, const Real& q1, const Real& q2, const Real& q3, const Real& q4,
                                Real &qr)
{
    Real ql;
    weno5(q0, q1, q2, q3, q4, qr, ql);
}
KOKKOS_FORCEINLINE_FUNCTION void weno5r(const Real& q0, const Real& q1, const Real& q2, const Real& q3, const Real& q4,
                                Real &ql)
{
    Real qr;
    weno5(q0, q1, q2, q3, q4, qr, ql);
}

// Row-wise implementations
// Note that "L" and "R" refer to the sides of the *face*
// ql(1) is to the right of the first zone center, but corresponds to the face value reconstructed from the left
// qr(1) is then the value at that face reconstructed from the right
// This is *opposite* the single-zone convention (or rather, offset from it to the faces), so weirdly WENO5X2l calls weno5r.  Get it?
template <typename T>
KOKKOS_INLINE_FUNCTION void WENO5X1(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
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
KOKKOS_INLINE_FUNCTION void WENO5X2(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
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
KOKKOS_INLINE_FUNCTION void WENO5X2l(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql)
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
KOKKOS_INLINE_FUNCTION void WENO5X2r(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &qr)
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
KOKKOS_INLINE_FUNCTION void WENO5X3(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
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
KOKKOS_INLINE_FUNCTION void WENO5X3l(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql)
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
KOKKOS_INLINE_FUNCTION void WENO5X3r(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &qr)
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
 * Templated calls to different reconstruction algorithms
 * This is basically a compile-time 'if' or 'switch' statement, where all the options get generated
 * at compile-time (see harm_driver.cpp where they're spelled out explicitly)
 * 
 * We could temlate these directly on the function if Parthenon could agree on what argument list to use
 * Better than a runtime decision per outer loop I think
 */
template <ReconstructionType Recon, int dir>
KOKKOS_INLINE_FUNCTION void reconstruct(parthenon::team_mbr_t& member, const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr) {}
// DONOR CELL
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::donor_cell, X1DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    DonorCellX1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::donor_cell, X2DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_u);
    DonorCellX2(member, k, j, is_l, ie_l, P, q_u, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::donor_cell, X3DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    DonorCellX3(member, k - 1, j, is_l, ie_l, P, ql, q_u);
    DonorCellX3(member, k, j, is_l, ie_l, P, q_u, qr);
}
// LINEAR W/VAN LEER
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_vl, X1DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // Extra scratch space for Parthenon's VL limiter stuff
    ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    PiecewiseLinearX1(member, k, j, is_l, ie_l, G, P, ql, qr, qc, dql, dqr, dqm);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_vl, X2DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // Extra scratch space for Parthenon's VL limiter stuff
    ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, G, P, ql, q_u, qc, dql, dqr, dqm);
    PiecewiseLinearX2(member, k, j, is_l, ie_l, G, P, q_u, qr, qc, dql, dqr, dqm);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_vl, X3DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // Extra scratch space for Parthenon's VL limiter stuff
    ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, G, P, ql, q_u, qc, dql, dqr, dqm);
    PiecewiseLinearX3(member, k, j, is_l, ie_l, G, P, q_u, qr, qc, dql, dqr, dqm);
}
// LINEAR WITH MC
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_mc, X1DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    KReconstruction::PiecewiseLinearX1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_mc, X2DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_u);
    KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_u, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::linear_mc, X3DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    KReconstruction::PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, P, ql, q_u);
    KReconstruction::PiecewiseLinearX3(member, k, j, is_l, ie_l, P, q_u, qr);
}
// WENO5
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::weno5, X1DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    KReconstruction::WENO5X1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::weno5, X2DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
    KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::weno5, X3DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    KReconstruction::WENO5X3l(member, k - 1, j, is_l, ie_l, P, ql);
    KReconstruction::WENO5X3r(member, k, j, is_l, ie_l, P, qr);
}

// WENO5 lowered poles:
// Linear X2 reconstruction near X2 boundaries
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::weno5_lower_poles, X1DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    reconstruct<ReconstructionType::weno5, X1DIR>(member, G, P, k, j, is_l, ie_l, ql, qr);
    // Linear X1 reconstruction near X1 boundaries (copied from lower_edges in kharma_next)
    //constexpr int o = 5; // offset
    //KReconstruction::WENO5X1(member, k, j, is_l+o, ie_l-o, P, ql, qr);
    //KReconstruction::PiecewiseLinearX1(member, k, j, is_l, is_l+o-1, P, ql, qr);
    //KReconstruction::PiecewiseLinearX1(member, k, j, ie_l-o+1, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::weno5_lower_poles, X2DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // This prioiritizes using the same fluxes on faces rather than for cells.
    // Neither is transparently wrong (afaict) but this feels nicer
    constexpr int o = 6; //5;
    if (j > o || j < P.GetDim(2) - o) {
        KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
        KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
    } else {
        ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
        KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_u);
        KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_u, qr);
    }
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<ReconstructionType::weno5_lower_poles, X3DIR>(parthenon::team_mbr_t& member,
                                        const GRCoordinates& G, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    reconstruct<ReconstructionType::weno5, X3DIR>(member, G, P, k, j, is_l, ie_l, ql, qr);
}

} // namespace KReconstruction
