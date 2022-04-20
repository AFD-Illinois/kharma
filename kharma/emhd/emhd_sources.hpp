/* 
 *  File: emhd_sources.hpp
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

#include "emhd.hpp"
#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"

/**
 * The various implicit/solved source terms for EGRMHD evolution.
 * Explicit terms are added in emhd.cpp
 */

namespace EMHD {

/**
 * Implicit source terms for EMHD, evaluated during implicit step calculation
 */
template<typename Local>
KOKKOS_FORCEINLINE_FUNCTION void implicit_sources(const GRCoordinates& G, const Local& P, const VarMap& m_p,
                                             const Real& gam, const int& j, const int& i,
                                             const EMHD_parameters& emhd_params,
                                             Real& dUq, Real& dUdP)
{
    // These are intentionally the tilde versions!
    Real tau, chi_e, nu_e;
    EMHD::set_parameters(G, P, m_p, emhd_params, gam, tau, chi_e, nu_e);
    dUq  = -G.gdet(Loci::center, j, i) * (P(m_p.Q) / tau);
    dUdP = -G.gdet(Loci::center, j, i) * (P(m_p.DP) / tau);
}

/**
 * EMHD source terms requiring time derivatives, used to evaluate residual
 * gamma, j, i,
 */
template<typename Local>
KOKKOS_FORCEINLINE_FUNCTION void time_derivative_sources(const GRCoordinates& G, const Local& P_new,
                                                    const Local& P_old, const Local& P,
                                                    const VarMap& m_p, const EMHD_parameters& emhd_params,
                                                    const Real& gam, const Real& dt, const int& j, const int& i,
                                                    Real& dUq, Real& dUdP)
{
    // Parameters
    Real tau, chi_e, nu_e;
    EMHD::set_parameters(G, P, m_p, emhd_params, gam, tau, chi_e, nu_e);

    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, j, i, Loci::center, Dtmp);
    double bsq = max(dot(Dtmp.bcon, Dtmp.bcov), SMALL);

    // TIME DERIVATIVES
    Real ucon[GR_DIM], ucov_new[GR_DIM], ucov_old[GR_DIM];
    GRMHD::calc_ucon(G, P_old, m_p, j, i, Loci::center, ucon);
    G.lower(ucon, ucov_old, 0, j, i, Loci::center);
    GRMHD::calc_ucon(G, P_new, m_p, j, i, Loci::center, ucon);
    G.lower(ucon, ucov_new, 0, j, i, Loci::center);
    Real dt_ucov[GR_DIM];
    DLOOP1 dt_ucov[mu] = (ucov_new[mu] - ucov_old[mu]) / dt;

    // Compute div of ucon (only the temporal part is nonzero)
    Real div_ucon = 0;
    DLOOP1 div_ucon += G.gcon(Loci::center, j, i, 0, mu) * dt_ucov[mu];
    // dTheta/dt
    const Real Theta_new = max((gam-1) * P_new(m_p.UU) / P_new(m_p.RHO), SMALL);
    const Real Theta_old = max((gam-1) * P_old(m_p.UU) / P_old(m_p.RHO), SMALL);
    const Real dt_Theta = (Theta_new - Theta_old) / dt;

    // TEMPORAL SOURCE TERMS
    const Real& rho = P(m_p.RHO);
    const Real& Theta = (gam-1) * P(m_p.UU) / P(m_p.RHO);
    Real q0 = -rho * chi_e * (Dtmp.bcon[0] / sqrt(bsq)) * dt_Theta;
    DLOOP1 q0 -= rho * chi_e * (Dtmp.bcon[mu] / sqrt(bsq)) * Theta * Dtmp.ucon[0] * dt_ucov[mu];

    Real dP0 = -rho * nu_e * div_ucon;
    DLOOP1 dP0 += 3. * rho * nu_e * (Dtmp.bcon[0] * Dtmp.bcon[mu] / bsq) * dt_ucov[mu];

    Real q0_tilde = 0., dP0_tilde = 0;
    EMHD::convert_q_dP_to_prims(q0, dP0, rho, Theta, tau, chi_e, nu_e, 
                                emhd_params, q0_tilde, dP0_tilde);

    dUq  = G.gdet(Loci::center, j, i) * (q0_tilde / tau);
    dUdP = G.gdet(Loci::center, j, i) * (dP0_tilde / tau);

    if (emhd_params.higher_order_terms) {
        dUq  += G.gdet(Loci::center, j, i) * (q0_tilde / 2.) * div_ucon;
        dUdP += G.gdet(Loci::center, j, i) * (dP0_tilde / 2.) * div_ucon;
    }
}

} // namespace EMHD
