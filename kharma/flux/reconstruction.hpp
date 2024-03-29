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

/**
 * This namespace covers custom new reconstructions for KHARMA, and a function which
 * automatically chooses the inner loop based on an enum from decs.hpp
 */
namespace KReconstruction
{
constexpr Real EPS = 1.e-26;

// Enum for types.
enum class Type{donor_cell=0, linear_mc, linear_vl, ppm, mp5, weno5, weno5_lower_edges, weno5_lower_poles};

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
            KOKKOS_LAMBDA (const int& i) {
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
            KOKKOS_LAMBDA (const int& i) {
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
            KOKKOS_LAMBDA (const int& i) {
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
// Adapted from implementation in iharm3d originally by Monika Moscibrodzka
// References: Tchekhovskoy et al. 2007 (T07), Shu 2011 (S11)

// Single-element implementation: "left" and "right" here are relative to zone centers, so the combo calls will switch them later.
KOKKOS_INLINE_FUNCTION void weno5(const Real& x1, const Real& x2, const Real& x3, const Real& x4, const Real& x5,
                                Real &lout, Real &rout)
{
    // // Smoothness indicators, T07 A18 or S11 8
    // const Real beta1 = (13./12.)*SQR(x1 - 2*x2 + x3)
    //                  + (1./4.)*SQR(x1 - 4*x2 + 3*x3);
    // const Real beta2 = (13./12.)*SQR(x2 - 2*x3 + x4)
    //                  + (1./4.)*SQR(x4 - x2);
    // const Real beta3 = (13./12.)*SQR(x3 - 2*x4 + x5)
    //                  + (1./4.)*SQR(x5 - 4*x4 + 3*x3);

    // // Nonlinear weights S11 9
    // const Real den_inv1 = 1./(EPS + beta1*beta1);
    // const Real den_inv2 = 1./(EPS + beta2*beta2);
    // const Real den_inv3 = 1./(EPS + beta3*beta3);

    // // S11 1, 2, 3 left
    // const Real wtl1 = 0.5 * den_inv3;
    // const Real wtl2 = 5 * den_inv2;
    // const Real wtl3 = 2.5 * den_inv1;
    // lout = ((3*x5  - 10*x4 + 15*x3)*wtl1 +
    //         (-x4 + 6*x3  + 3*x2)*wtl2 +
    //         (3*x3  + 6*x2  - x1)*wtl3)
    //         / (8*(wtl1 + wtl2 + wtl3));

    // // S11 1, 2, 3 right
    // const Real wtr1 = 0.5 * den_inv1;
    // const Real wtr2 = 5 * den_inv2;
    // const Real wtr3 = 2.5 * den_inv3;
    // rout = ((3*x1 - 10*x2 + 15*x3)*wtr1 +
    //         (-x2 + 6*x3 + 3*x4)*wtr2 +
    //         (3*x3 + 6*x4 - x5)*wtr3)
    //         / (8*(wtr1 + wtr2 + wtr3));

    // OLD RECON
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
KOKKOS_INLINE_FUNCTION void weno5l(const Real x1, const Real& x2, const Real& x3, const Real x4, const Real& x5,
                                Real &lout)
{
    // // Smoothness indicators, T07 A18 or S11 8
    // const Real beta1 = (13./12.)*SQR(x1 - 2*x2 + x3)
    //                  + (1./4.)*SQR(x1 - 4*x2 + 3*x3);
    // const Real beta2 = (13./12.)*SQR(x2 - 2*x3 + x4)
    //                  + (1./4.)*SQR(x4 - x2);
    // const Real beta3 = (13./12.)*SQR(x3 - 2*x4 + x5)
    //                  + (1./4.)*SQR(x5 - 4*x4 + 3*x3);

    // // Nonlinear weights S11 9
    // const Real den_inv1 = 1./(EPS + beta1*beta1);
    // const Real den_inv2 = 1./(EPS + beta2*beta2);
    // const Real den_inv3 = 1./(EPS + beta3*beta3);

    // // S11 1, 2, 3 left
    // const Real wtl1 = 0.5 * den_inv3;
    // const Real wtl2 = 5 * den_inv2;
    // const Real wtl3 = 2.5 * den_inv1;
    // lout = ((3*x5  - 10*x4 + 15*x3)*wtl1 +
    //         (-x4 + 6*x3  + 3*x2)*wtl2 +
    //         (3*x3  + 6*x2  - x1)*wtl3)
    //         / (8*(wtl1 + wtl2 + wtl3));


    // OLD RECON
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
KOKKOS_INLINE_FUNCTION void weno5r(const Real& x1, const Real& x2, const Real& x3, const Real x4, const Real& x5,
                                Real &rout)
{
    // // Smoothness indicators, T07 A18 or S11 8
    // const Real beta1 = (13./12.)*SQR(x1 - 2*x2 + x3)
    //                  + (1./4.)*SQR(x1 - 4*x2 + 3*x3);
    // const Real beta2 = (13./12.)*SQR(x2 - 2*x3 + x4)
    //                  + (1./4.)*SQR(x4 - x2);
    // const Real beta3 = (13./12.)*SQR(x3 - 2*x4 + x5)
    //                  + (1./4.)*SQR(x5 - 4*x4 + 3*x3);

    // // Nonlinear weights S11 9
    // const Real den_inv1 = 1./(EPS + beta1*beta1);
    // const Real den_inv2 = 1./(EPS + beta2*beta2);
    // const Real den_inv3 = 1./(EPS + beta3*beta3);

    // // S11 1, 2, 3 right
    // const Real wtr1 = 0.5 * den_inv1;
    // const Real wtr2 = 5 * den_inv2;
    // const Real wtr3 = 2.5 * den_inv3;
    // rout = ((3*x1 - 10*x2 + 15*x3)*wtr1 +
    //         (-x2 + 6*x3 + 3*x4)*wtr2 +
    //         (3*x3 + 6*x4 - x5)*wtr3)
    //         / (8*(wtr1 + wtr2 + wtr3));


    // OLD RECON
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
KOKKOS_INLINE_FUNCTION void WENO5X1(parthenon::team_mbr_t const &member, const int& k, const int& j,
                       const int& il, const int& iu, const T &q, ScratchPad2D<Real> &ql,
                       ScratchPad2D<Real> &qr)
{
    const int nu = q.GetDim(4) - 1;
    for (int p = 0; p <= nu; ++p) {
        parthenon::par_for_inner(member, il, iu,
            KOKKOS_LAMBDA (const int& i) {
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
            KOKKOS_LAMBDA (const int& i) {
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
            KOKKOS_LAMBDA (const int& i) {
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
            KOKKOS_LAMBDA (const int& i) {
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
            KOKKOS_LAMBDA (const int& i) {
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
            KOKKOS_LAMBDA (const int& i) {
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
            KOKKOS_LAMBDA (const int& i) {
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
 * Parablic reconstruction, see Collela & Woodward '84
 *  
 * Adapted from iharm2d implementation,
 * originally written by Xiaoyue Guan
 */
KOKKOS_INLINE_FUNCTION void para(const Real& x1, const Real& x2, const Real& x3, const Real& x4, const Real& x5,
          Real& lout, Real& rout)
{
    Real y[5], dq[5]; // TODO(BSP) can these be eliminated easily?

    y[0]=x1;
    y[1]=x2;
    y[2]=x3;
    y[3]=x4;
    y[4]=x5;

    // CW 1.7
    for (int i=1; i <= 3; i++) {
        const Real Dqm = 2*(y[i] - y[i-1]);
        const Real Dqp = 2*(y[i+1] - y[i]);
        if (Dqm*Dqp <= 0.) {
            dq[i] = 0.; // CW1.8
        } else {
            const Real Dqc = 0.5*(y[i+1] - y[i-1]);
            dq[i] = m::copysign(m::min(m::abs(Dqc), m::min(m::abs(Dqm), m::abs(Dqp))), Dqc);
        }
    }

    // CW 1.6
    lout = 0.5*(y[2] + y[1]) - (1./6.)*(dq[2] - dq[1]);
    rout = 0.5*(y[3] + y[2]) - (1./6.)*(dq[3] - dq[2]);

    // CW 1.10
    if (((rout - y[2]) * (y[2] - lout)) <= 0.) {
        lout = y[2];
        rout = y[2];
    }
    const Real qd = (rout - lout);
    const Real qe = 6*(y[2] - 0.5*(lout + rout));
    if (qd * (qd - qe) < 0.) {
        lout = 3*y[2] - 2*rout;
    } else if (qd * (qd + qe) < 0.) {
        rout = 3*y[2] - 2*lout;
    }
}


/**
 * Templated calls to different reconstruction algorithms
 * This is basically a compile-time 'if' or 'switch' statement, where all the options get generated
 * at compile-time (see driver.cpp for the different instantiations)
 * 
 * We could template these directly on the function if Parthenon could agree on what argument list to use
 * Better than a runtime decision per outer loop I think
 */
template <Type Recon, int dir>
KOKKOS_INLINE_FUNCTION void reconstruct(parthenon::team_mbr_t& member, const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr) {}
// DONOR CELL
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::donor_cell, X1DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    DonorCellX1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::donor_cell, X2DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_u);
    DonorCellX2(member, k, j, is_l, ie_l, P, q_u, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::donor_cell, X3DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    DonorCellX3(member, k - 1, j, is_l, ie_l, P, ql, q_u);
    DonorCellX3(member, k, j, is_l, ie_l, P, q_u, qr);
}
// LINEAR W/VAN LEER
// template <>
// KOKKOS_INLINE_FUNCTION void reconstruct<Type::linear_vl, X1DIR>(parthenon::team_mbr_t& member,
//                                         const VariablePack<Real> &P,
//                                         const int& k, const int& j, const int& is_l, const int& ie_l, 
//                                         ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
// {
//     // Extra scratch space for Parthenon's VL limiter stuff
//     ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     PiecewiseLinearX1(member, k, j, is_l, ie_l, G, P, ql, qr, qc, dql, dqr, dqm);
// }
// template <>
// KOKKOS_INLINE_FUNCTION void reconstruct<Type::linear_vl, X2DIR>(parthenon::team_mbr_t& member,
//                                         const VariablePack<Real> &P,
//                                         const int& k, const int& j, const int& is_l, const int& ie_l, 
//                                         ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
// {
//     // Extra scratch space for Parthenon's VL limiter stuff
//     ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, G, P, ql, q_u, qc, dql, dqr, dqm);
//     PiecewiseLinearX2(member, k, j, is_l, ie_l, G, P, q_u, qr, qc, dql, dqr, dqm);
// }
// template <>
// KOKKOS_INLINE_FUNCTION void reconstruct<Type::linear_vl, X3DIR>(parthenon::team_mbr_t& member,
//                                         const VariablePack<Real> &P,
//                                         const int& k, const int& j, const int& is_l, const int& ie_l, 
//                                         ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
// {
//     // Extra scratch space for Parthenon's VL limiter stuff
//     ScratchPad2D<Real>  qc(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> dql(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> dqr(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> dqm(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
//     PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, G, P, ql, q_u, qc, dql, dqr, dqm);
//     PiecewiseLinearX3(member, k, j, is_l, ie_l, G, P, q_u, qr, qc, dql, dqr, dqm);
// }
// LINEAR WITH MC
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::linear_mc, X1DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    KReconstruction::PiecewiseLinearX1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::linear_mc, X2DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_u);
    KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_u, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::linear_mc, X3DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
    KReconstruction::PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, P, ql, q_u);
    KReconstruction::PiecewiseLinearX3(member, k, j, is_l, ie_l, P, q_u, qr);
}
// WENO5
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::weno5, X1DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    KReconstruction::WENO5X1(member, k, j, is_l, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::weno5, X2DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
    KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::weno5, X3DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    KReconstruction::WENO5X3l(member, k - 1, j, is_l, ie_l, P, ql);
    KReconstruction::WENO5X3r(member, k, j, is_l, ie_l, P, qr);
}
// WENO5 lowered edges:
// Linear X1 reconstruction near X1 boundaries
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::weno5_lower_edges, X1DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // This prioiritizes using the same-order fluxes on faces rather than for cells.
    // Neither is transparently wrong (afaict) but this feels nicer
    constexpr int o = 5; // offset
    KReconstruction::WENO5X1(member, k, j, is_l+o, ie_l-o, P, ql, qr);
    KReconstruction::PiecewiseLinearX1(member, k, j, is_l, is_l+o-1, P, ql, qr);
    KReconstruction::PiecewiseLinearX1(member, k, j, ie_l-o+1, ie_l, P, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::weno5_lower_edges, X2DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    reconstruct<Type::weno5, X2DIR>(member, P, k, j, is_l, ie_l, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::weno5_lower_edges, X3DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    reconstruct<Type::weno5, X3DIR>(member, P, k, j, is_l, ie_l, ql, qr);
}
// WENO5 lowered poles:
// Linear X2 reconstruction near X2 boundaries
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::weno5_lower_poles, X1DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    reconstruct<Type::weno5, X1DIR>(member, P, k, j, is_l, ie_l, ql, qr);
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::weno5_lower_poles, X2DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    // This prioiritizes using the same fluxes on faces rather than for cells.
    // Neither is transparently wrong (afaict) but this feels nicer
    constexpr int o = 5;
    if (j > o || j < P.GetDim(1) - o) {
        KReconstruction::WENO5X2l(member, k, j - 1, is_l+o, ie_l-o, P, ql);
        KReconstruction::WENO5X2r(member, k, j, is_l+o, ie_l-o, P, qr);
    } else {
        ScratchPad2D<Real> q_u(member.team_scratch(1), P.GetDim(4), P.GetDim(1));
        KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, is_l+o-1, P, ql, q_u);
        KReconstruction::PiecewiseLinearX2(member, k, j, is_l, is_l+o-1, P, q_u, qr);
    }
}
template <>
KOKKOS_INLINE_FUNCTION void reconstruct<Type::weno5_lower_poles, X3DIR>(parthenon::team_mbr_t& member,
                                        const VariablePack<Real> &P,
                                        const int& k, const int& j, const int& is_l, const int& ie_l, 
                                        ScratchPad2D<Real> ql, ScratchPad2D<Real> qr)
{
    reconstruct<Type::weno5, X3DIR>(member, P, k, j, is_l, ie_l, ql, qr);
}

/**
 * Versions computing just the (limited) slope, for linear reconstructions.
 * Used for gradient calculations needed to implement Extended GRMHD.
 */
template <Type Recon>
KOKKOS_INLINE_FUNCTION Real slope_limit(Real x1, Real x2, Real x3, Real dx);
// Linear MC slope limiter
template <>
KOKKOS_INLINE_FUNCTION Real slope_limit<Type::linear_mc>(Real x1, Real x2, Real x3, Real dx)
{
    const Real Dqm = 2 * (x2 - x1) / dx;
    const Real Dqp = 2 * (x3 - x2) / dx;
    const Real Dqc = 0.5 * (x3 - x1) / dx;

    if (Dqm * Dqp <= 0) {
        return 0;
    } else {
        if ((m::abs(Dqm) < m::abs(Dqp)) && (m::abs(Dqm) < m::abs(Dqc))) {
            return Dqm;
        } else if (m::abs(Dqp) < m::abs(Dqc)) {
            return Dqp;
        } else {
            return Dqc;
        }
    }
}
// Linear Van Leer slope limiter
template <>
KOKKOS_INLINE_FUNCTION Real slope_limit<Type::linear_vl>(Real x1, Real x2, Real x3, Real dx)
{
    const Real Dqm = (x2 - x1) / dx;
    const Real Dqp = (x3 - x2) / dx;

    const Real extrema = Dqm * Dqp;

    if (extrema <= 0) {
        return 0;
    } else {
        return (2 * extrema / (Dqm + Dqp)); 
    }
}

/**
 * Run slope_limit in direction 'dir' using limiter 'recon'
 */
template <Type recon, int dir>
KOKKOS_INLINE_FUNCTION Real slope_calc(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i);
// And six implementations.  Why can't you partial-specialize functions?  Why?
template <>
KOKKOS_INLINE_FUNCTION Real slope_calc<Type::linear_mc, X1DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_mc>(P(p, k, j, i-1), P(p, k, j, i), P(p, k, j, i+1), G.Dxc<X1DIR>(i));
}
template <>
KOKKOS_INLINE_FUNCTION Real slope_calc<Type::linear_mc, X2DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_mc>(P(p, k, j-1, i), P(p, k, j, i), P(p, k, j+1, i), G.Dxc<X2DIR>(j));
}
template <>
KOKKOS_INLINE_FUNCTION Real slope_calc<Type::linear_mc, X3DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_mc>(P(p, k-1, j, i), P(p, k, j, i), P(p, k+1, j, i), G.Dxc<X3DIR>(k));
}
template <>
KOKKOS_INLINE_FUNCTION Real slope_calc<Type::linear_vl, X1DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_vl>(P(p, k, j, i-1), P(p, k, j, i), P(p, k, j, i+1), G.Dxc<X1DIR>(i));
}
template <>
KOKKOS_INLINE_FUNCTION Real slope_calc<Type::linear_vl, X2DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_vl>(P(p, k, j-1, i), P(p, k, j, i), P(p, k, j+1, i), G.Dxc<X2DIR>(j));
}
template <>
KOKKOS_INLINE_FUNCTION Real slope_calc<Type::linear_vl, X3DIR>(const GRCoordinates& G, const VariablePack<Real>& P,
                                              const int& p, const int& k, const int& j, const int& i)
{
    return slope_limit<Type::linear_vl>(P(p, k-1, j, i), P(p, k, j, i), P(p, k+1, j, i), G.Dxc<X3DIR>(k));
}

} // namespace KReconstruction
