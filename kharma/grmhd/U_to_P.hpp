/* 
 *  File: U_to_P.hpp
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

#include "kharma_utils.hpp"

// Accuracy required for U to P
#define UTOP_ERRTOL 1.e-8
// Maximum iterations when doing U to P inversion
#define UTOP_ITER_MAX 8
// Heuristic step size
#define DELTA 1e-5

namespace GRMHD {

KOKKOS_INLINE_FUNCTION Real err_eqn(const Real& gam, const Real& Bsq, const Real& D, const Real& Ep, const Real& QdB,
                                    const Real& Qtsq, const Real& Wp, InversionStatus& eflag);
KOKKOS_INLINE_FUNCTION Real lorentz_calc_w(const Real& Bsq, const Real& D, const Real& QdB,
                                        const Real& Qtsq, const Real& Wp);

/**
 * Recover local primitive variables, with a one-dimensional Newton-Raphson iterative solver.
 * Iteration starts from the current primitive values, and otherwse may *fail to converge*
 * 
 * Returns a code indicating whether the solver converged (success), failed (max_iter), or
 * indicating that the converged solution was unphysical (bad_ut, neg_rhou, neg_rho, neg_u)
 * 
 * On error, will not write replacement values, leaving the previous step's values in place
 * These are fixed later, in FixUtoP
 */
KOKKOS_INLINE_FUNCTION InversionStatus u_to_p(const GRCoordinates &G, const VariablePack<Real>& U, const VarMap& m_u,
                                              const Real& gam, const int& k, const int& j, const int& i, const Loci loc,
                                              const VariablePack<Real>& P, const VarMap& m_p)
{
    // Catch negative density
    if (U(m_u.RHO, k, j, i) <= 0.) {
        return InversionStatus::neg_input;
    }

    // Convert from conserved variables to four-vectors
    const Real alpha = 1./m::sqrt(-G.gcon(loc, j, i, 0, 0));
    const Real gdet = G.gdet(loc, j, i);
    const Real a_over_g = alpha / gdet;
    const Real D = U(m_u.RHO, k, j, i) * a_over_g;

    Real Bcon[GR_DIM] = {0};
    if (m_u.B1 >= 0) {
        Bcon[1] = U(m_u.B1, k, j, i) * a_over_g;
        Bcon[2] = U(m_u.B2, k, j, i) * a_over_g;
        Bcon[3] = U(m_u.B3, k, j, i) * a_over_g;
    }

    const Real Qcov[GR_DIM] =
        {(U(m_u.UU, k, j, i) - U(m_u.RHO, k, j, i)) * a_over_g,
          U(m_u.U1, k, j, i) * a_over_g,
          U(m_u.U2, k, j, i) * a_over_g,
          U(m_u.U3, k, j, i) * a_over_g};

    const Real ncov[GR_DIM] = {(Real) -alpha, 0., 0., 0.};

    // TODO faster with on-the-fly gcon/cov?
    Real Bcov[GR_DIM], Qcon[GR_DIM], ncon[GR_DIM];
    G.lower(Bcon, Bcov, k, j, i, loc);
    G.raise(Qcov, Qcon, k, j, i, loc);
    G.raise(ncov, ncon, k, j, i, loc);

    const Real Bsq = dot(Bcon, Bcov);
    const Real QdB = dot(Bcon, Qcov);
    const Real Qdotn = dot(Qcon, ncov);

    Real Qtcon[GR_DIM];
    DLOOP1 Qtcon[mu] = Qcon[mu] + ncon[mu] * Qdotn;
    const Real Qtsq = dot(Qcon, Qcov) + m::pow(Qdotn, 2);

    // Set up eqtn for W'; this is the energy density
    const Real Ep = -Qdotn - D;

    // Numerical rootfinding

    // Accumulator for errors in err_eqn
    InversionStatus eflag = InversionStatus::success;

    // Initial guess from primitives:
    Real Wp, err;
    {
        const Real gamma = GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc);
        if (gamma < 1) return InversionStatus::bad_ut;
        const Real rho = P(m_p.RHO, k, j, i), u = P(m_p.UU, k, j, i);

        Wp = (rho + u + (gam - 1) * u) * gamma * gamma - rho * gamma;
        err = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);
    }

    Real dW;
    {
        // Step around the guess & evaluate errors
        const Real Wpm = (1. - DELTA) * Wp; //heuristic
        const Real h = Wp - Wpm;
        const Real Wpp = Wp + h;
        const Real errm = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wpm, eflag);
        const Real errp = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wpp, eflag);

        // Attempt a Halley/Muller/Bailey/Press step
        const Real dedW = (errp - errm) / (Wpp - Wpm);
        const Real dedW2 = (errp - 2. * err + errm) / m::pow(h,2);
        // TODO look at this clip & the next vs iteration convergence %s
        const Real f = clip(0.5 * err * dedW2 / m::pow(dedW,2), -0.3, 0.3);

        dW = clip(-err / dedW / (1. - f), -0.5*Wp, 2.0*Wp);
    }

    // Take the first step
    Real Wp1 = Wp;
    Real err1 = err;
    Wp += dW;
    err = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

    // Not good enough?  apply secant method
    int iter = 0;
    for (iter = 0; iter < UTOP_ITER_MAX; iter++)
    {
        dW = clip((Wp1 - Wp) * err / (err - err1), (Real) -0.5*Wp, (Real) 2.0*Wp);

        Wp1 = Wp;
        err1 = err;

        Wp += dW;

        if (m::abs(dW / Wp) < UTOP_ERRTOL) break;

        err = err_eqn(gam, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

        if (m::abs(err / Wp) < UTOP_ERRTOL) break;
    }
    // If there was a bad gamma calculation, do not set primitives other than B
    // Uncomment to error on any bad velocity.  iharm2d/3d do not do this.
    //if (eflag) return eflag;
    // Return failure to converge
    if (iter == UTOP_ITER_MAX) return InversionStatus::max_iter;

    // Find utsq, gamma, rho from Wp
    const Real gamma = lorentz_calc_w(Bsq, D, QdB, Qtsq, Wp);
    if (gamma < 1) return InversionStatus::bad_ut;

    const Real rho = D / gamma;
    const Real W = Wp + D;
    const Real w = W / (gamma*gamma);
    const Real p = (w - rho) * (gam - 1) / gam;
    const Real u = w - (rho + p);

    // Return without updating non-B primitives
    if (rho < 0 && u < 0) return InversionStatus::neg_rhou;
    else if (rho < 0) return InversionStatus::neg_rho;
    else if (u < 0) return InversionStatus::neg_u;

    // Set primitives
    P(m_p.RHO, k, j, i) = rho;
    P(m_p.UU, k, j, i) = u;

    // Find u(tilde); Eqn. 31 of Noble et al.
    const Real pre = (gamma / (W + Bsq));
    P(m_p.U1, k, j, i) = pre * (Qtcon[1] + QdB * Bcon[1] / W);
    P(m_p.U2, k, j, i) = pre * (Qtcon[2] + QdB * Bcon[2] / W);
    P(m_p.U3, k, j, i) = pre * (Qtcon[3] + QdB * Bcon[3] / W);

    return InversionStatus::success;
}

// Document this
KOKKOS_INLINE_FUNCTION Real err_eqn(const Real& gam, const Real& Bsq, const Real& D, const Real& Ep, const Real& QdB,
                                    const Real& Qtsq, const Real& Wp, InversionStatus& eflag)
{
    const Real W = Wp + D;
    const Real gamma = lorentz_calc_w(Bsq, D, QdB, Qtsq, Wp);
    if (gamma < 1) eflag = InversionStatus::bad_ut;
    const Real w = W / m::pow(gamma,2);
    const Real rho = D / gamma;
    const Real p = (w - rho) * (gam - 1) / gam;

    return -Ep + Wp - p + 0.5 * Bsq + 0.5 * (Bsq * Qtsq - QdB * QdB) / m::pow((Bsq + W), 2);

}

/**
 * Fluid relativistic factor gamma in terms of inversion state variables
 */
KOKKOS_INLINE_FUNCTION Real lorentz_calc_w(const Real& Bsq, const Real& D, const Real& QdB,
                                           const Real& Qtsq, const Real& Wp)
{
    const Real QdBsq = QdB * QdB;
    const Real W = Wp + D;
    const Real W2 = W * W;
    const Real WB = W + Bsq;

    // This is basically inversion of eq. A7 of Mignone & McKinney
    const Real utsq = -((W + WB) * QdBsq + W2 * Qtsq) / (QdBsq * (W + WB) + W2 * (Qtsq - WB * WB));

    // Catch utsq < 0 and YELL
    // TODO latter number should be ~1e3*GAMMAMAX^2
    if (utsq < -1.e-15 || utsq > 1.e7) {
        return -1.; // This will trigger caller to return an error immediately
    } else {
        return m::sqrt(1. + m::abs(utsq));
    }
}

} // namespace GRMHD
