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

#include "utils.hpp"

#define ERRTOL 1.e-8

namespace GRMHD {

KOKKOS_INLINE_FUNCTION Real err_eqn(const EOS* eos, const Real& Bsq, const Real& D, const Real& Ep, const Real& QdB,
                                    const Real& Qtsq, const Real& Wp, InversionStatus& eflag);
KOKKOS_INLINE_FUNCTION Real lorentz_calc_w(const Real& Bsq, const Real& D, const Real& QdB,
                                        const Real& Qtsq, const Real& Wp);

/**
 * Recover local primitives
 * Like in *_functions.hpp, this function is overloaded between passing references to global arrays, and "immediates"
 * The difference is this one is probably coded pretty slowly
 */
KOKKOS_INLINE_FUNCTION InversionStatus u_to_p(const GRCoordinates &G, const Real U[NPRIM],
                                    const Real B_U[NVEC], const EOS* eos,
                                    const int& k, const int& j, const int& i, const Loci loc, Real P[NPRIM])
{
    Real alpha = 1./sqrt(-G.gcon(loc, j, i, 0, 0));
    Real gdet = G.gdet(loc, j, i);
    Real a_over_g = alpha / gdet;

    // Catch negative density
    if (U[prims::rho] <= 0.) {
        return InversionStatus::neg_input;
    }

    // Convert from conserved variables to four-vectors
    Real D = U[prims::rho] * a_over_g;

    Real Bcon[GR_DIM];
    Bcon[0] = 0.;
    Bcon[1] = B_U[0] * a_over_g;
    Bcon[2] = B_U[1] * a_over_g;
    Bcon[3] = B_U[2] * a_over_g;

    Real Qcov[GR_DIM];
    Qcov[0] = (U[prims::u] - U[prims::rho]) * a_over_g;
    Qcov[1] = U[prims::u1] * a_over_g;
    Qcov[2] = U[prims::u2] * a_over_g;
    Qcov[3] = U[prims::u3] * a_over_g;

    Real ncov[GR_DIM] = {(Real) -alpha, 0., 0., 0.};

    // TODO faster with on-the-fly gcon/cov?
    Real Bcov[GR_DIM], Qcon[GR_DIM], ncon[GR_DIM];
    G.lower(Bcon, Bcov, k, j, i, loc);
    G.raise(Qcov, Qcon, k, j, i, loc);
    G.raise(ncov, ncon, k, j, i, loc);

    Real Bsq = dot(Bcon, Bcov);
    Real QdB = dot(Bcon, Qcov);
    Real Qdotn = dot(Qcon, ncov);

    Real Qtcon[GR_DIM];
    DLOOP1 Qtcon[mu] = Qcon[mu] + ncon[mu] * Qdotn;
    Real Qtsq = dot(Qcon, Qcov) + pow(Qdotn, 2);

    // Set up eqtn for W'; this is the energy density
    Real Ep = -Qdotn - D;

    // Numerical rootfinding

    // Initial guess from primitives:
    // Fetch the current values, or guess defaults if they're uninitialized
    Real gamma, rho0, u;
    int iter_max;
    if (P[prims::rho] != 0. || P[prims::u] != 0.) {
        gamma = lorentz_calc(G, &(P[prims::u1]), j, i, loc);
        rho0 = P[prims::rho];
        u = P[prims::u];
        iter_max = 8;
    } else {
        //printf("Guessing P\n");
        rho0 = 0.1;
        u = 0.01;
        gamma = 1.1;
        iter_max = 20;
    }
    if (gamma < 1) return InversionStatus::bad_ut;
    // Calculate an initial guess for Wp
    Real Wp = (rho0 + u + eos->p(rho0, u)) * gamma * gamma - rho0 * gamma;

    // Gather any errors during iteration with a single flag to return afterward
    InversionStatus eflag = InversionStatus::success;

    // Step around the guess & evaluate errors
    Real Wpm = (1. - DELTA) * Wp; //heuristic
    Real h = Wp - Wpm;
    Real Wpp = Wp + h;
    Real errp = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wpp, eflag);
    Real err = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);
    Real errm = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wpm, eflag);

    // Attempt a Halley/Muller/Bailey/Press step
    Real dedW = (errp - errm) / (Wpp - Wpm);
    Real dedW2 = (errp - 2. * err + errm) / pow(h,2);
    // TODO look at this clip & the next vs iteration convergence %s
    Real f = clip(0.5 * err * dedW2 / pow(dedW,2), -0.3, 0.3);

    Real dW = clip(-err / dedW / (1. - f), -0.5*Wp, 2.0*Wp);

    // Wp, dW, err
    Real Wp1 = Wp;
    Real err1 = err;

    Wp += dW;
    err = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

    // Not good enough?  apply secant method
    int iter = 0;
    for (iter = 0; iter < iter_max; iter++)
    {
        dW = clip((Wp1 - Wp) * err / (err - err1), (Real) -0.5*Wp, (Real) 2.0*Wp);

        Wp1 = Wp;
        err1 = err;

        Wp += dW;

        if (fabs(dW / Wp) < UTOP_ERRTOL) break;

        err = err_eqn(eos, Bsq, D, Ep, QdB, Qtsq, Wp, eflag);

        if (fabs(err / Wp) < UTOP_ERRTOL) break;
    }
    // If there was a bad gamma calculation, do not set primitives other than B
    // Uncomment to error on any bad velocity.  iharm2d/3d do not do this.
    //if (eflag) return eflag;
    // Return failure to converge
    if (iter == iter_max) return InversionStatus::max_iter;

    // Find utsq, gamma, rho0 from Wp
    gamma = lorentz_calc_w(Bsq, D, QdB, Qtsq, Wp);
    if (gamma < 1) return InversionStatus::bad_ut;

    rho0 = D / gamma;
    Real W = Wp + D;
    Real w = W / (gamma*gamma);
    Real p = eos->p_w(rho0, w);
    u = w - (rho0 + p);

    // Return without updating non-B primitives
    if (rho0 < 0 && u < 0) return InversionStatus::neg_rhou;
    else if (rho0 < 0) return InversionStatus::neg_rho;
    else if (u < 0) return InversionStatus::neg_u;

    // Set primitives
    P[prims::rho] = rho0;
    P[prims::u] = u;

    // Find u(tilde); Eqn. 31 of Noble et al.
    Real pre = (gamma / (W + Bsq));
    P[prims::u1] = pre * (Qtcon[1] + QdB * Bcon[1] / W);
    P[prims::u2] = pre * (Qtcon[2] + QdB * Bcon[2] / W);
    P[prims::u3] = pre * (Qtcon[3] + QdB * Bcon[3] / W);

    return InversionStatus::success;
}
KOKKOS_INLINE_FUNCTION InversionStatus u_to_p(const GRCoordinates &G, const GridVars U, const GridVector B_U, const EOS* eos,
                                  const int& k, const int& j, const int& i, const Loci loc, GridVars P)
{
    Real Pl[NPRIM], Ul[NPRIM], B_Ul[NVEC];
    PLOOP {
        Ul[p] = U(p, k, j, i);
        Pl[p] = P(p, k, j, i);
    }
    VLOOP B_Ul[v] = B_U(v, k, j, i);
    InversionStatus ret = u_to_p(G, Ul, B_Ul, eos, k, j, i, loc, Pl);
    PLOOP P(p, k, j, i) = Pl[p];
    return ret;
}
KOKKOS_INLINE_FUNCTION InversionStatus u_to_p(const GRCoordinates &G, const GridVars U, const Real B_U[NVEC], const EOS* eos,
                                  const int& k, const int& j, const int& i, const Loci loc, GridVars P)
{
    Real Pl[NPRIM], Ul[NPRIM];
    PLOOP {
        Ul[p] = U(p, k, j, i);
        Pl[p] = P(p, k, j, i);
    }
    InversionStatus ret = u_to_p(G, Ul, B_U, eos, k, j, i, loc, Pl);
    PLOOP P(p, k, j, i) = Pl[p];
    return ret;
}

// Document this
KOKKOS_INLINE_FUNCTION Real err_eqn(const EOS* eos, const Real& Bsq, const Real& D, const Real& Ep, const Real& QdB,
                                    const Real& Qtsq, const Real& Wp, InversionStatus& eflag)
{
    Real W = Wp + D;
    Real gamma = lorentz_calc_w(Bsq, D, QdB, Qtsq, Wp);
    if (gamma < 1) eflag = InversionStatus::bad_ut;
    Real w = W / pow(gamma,2);
    Real rho0 = D / gamma;
    Real p = eos->p_w(rho0, w);

    return -Ep + Wp - p + 0.5 * Bsq + 0.5 * (Bsq * Qtsq - QdB * QdB) / pow((Bsq + W), 2);

}

/**
 * Fluid relativistic factor gamma in terms of inversion state variables
 */
KOKKOS_INLINE_FUNCTION Real lorentz_calc_w(const Real& Bsq, const Real& D, const Real& QdB,
                                        const Real& Qtsq, const Real& Wp)
{
    Real QdBsq = QdB * QdB;
    Real W = Wp + D;
    Real W2 = W * W;
    Real WB = W + Bsq;

    // This is basically inversion of eq. A7 of Mignone & McKinney
    Real utsq = -((W + WB) * QdBsq + W2 * Qtsq) / (QdBsq * (W + WB) + W2 * (Qtsq - WB * WB));

    // Catch utsq < 0 and YELL
    // TODO latter number should be ~1e3*GAMMAMAX^2
    if (utsq < -1.e-15 || utsq > 1.e7) {
        return -1.;
    } else {
        return sqrt(1. + fabs(utsq));
    }
}

} // namespace GRMHD
