/* 
 *  File: emhd.hpp
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

#include <parthenon/parthenon.hpp>

#include "grmhd_functions.hpp"

using namespace parthenon;

/**
 * This physics package implements the Extended GRMHD "EGRMHD" scheme of Chandra et al. 2015,
 * First implemented in GRIM, of Chandra et al. 2017.
 * 
 * It adds variables representing viscosity and heat conduction, with a combination of explicit
 * and implicit source terms; thus it requires a semi-implicit scheme for evolution,
 * implemented in KHARMA as ImexDriver.
 */
namespace EMHD {

enum ClosureType{constant=0, soundspeed, torus};

class EMHD_parameters {
    public:

        bool higher_order_terms;
        ClosureType type;
        Real tau;
        Real conduction_alpha;
        Real viscosity_alpha;

};

/**
 * Initialization: handle parameters, 
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);

/**
 * Add EGRMHD explicit source terms: anything which can be calculated once
 * and added to the general dU/dt term along with e.g. GRMHD source, wind, etc
 */
TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt);

/**
 * Set chi, nu, tau. Problem dependent
 * 
 * TODO Local & Global, when we're sure
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const Local& P, const VarMap& m_p,
                                           const EMHD_parameters& emhd_params, const Real& gam,
                                           Real& tau, Real& chi_e, Real& nu_e)
{
    if (emhd_params.type == ClosureType::constant) {
        // Set tau, nu, chi to constants
        tau   = emhd_params.tau;
        chi_e = emhd_params.conduction_alpha;
        nu_e  = emhd_params.viscosity_alpha;
    } else if (emhd_params.type == ClosureType::soundspeed) {
        // Set tau=const, chi/nu prop. to sound speed squared
        Real cs2 = (gam * (gam - 1.) * P(m_p.UU)) / (P(m_p.RHO) + (gam * P(m_p.UU)));

        tau   = emhd_params.tau;
        chi_e = emhd_params.conduction_alpha * cs2 * tau;
        nu_e  = emhd_params.viscosity_alpha * cs2 * tau;
    } else if (emhd_params.type == ClosureType::torus) {
        // Something complicated
    } // else yell?
}
template<typename Global>
KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const Global& P, const VarMap& m_p,
                                           const EMHD_parameters& emhd_params, const Real& gam,
                                           const int& k, const int& j, const int& i,
                                           Real& tau, Real& chi_e, Real& nu_e)
{
    if (emhd_params.type == ClosureType::constant) {
        // Set tau, nu, chi to constants
        tau   = emhd_params.tau;
        chi_e = emhd_params.conduction_alpha;
        nu_e  = emhd_params.viscosity_alpha;
    } else if (emhd_params.type == ClosureType::soundspeed) {
        // Set tau=const, chi/nu prop. to sound speed squared
        const Real cs2 = (gam * (gam - 1.) * P(m_p.UU, k, j, i)) /
                            (P(m_p.RHO, k, j, i) + (gam * P(m_p.UU, k, j, i)));

        tau   = emhd_params.tau;
        chi_e = emhd_params.conduction_alpha * cs2 * tau;
        nu_e  = emhd_params.viscosity_alpha * cs2 * tau;
    } else if (emhd_params.type == ClosureType::torus) {
        // Something complicated
    } // else yell?
}
// Local version for use in initialization, as q/dP need to be converted to prim tilde forms
KOKKOS_INLINE_FUNCTION void set_parameters(const GRCoordinates& G, const Real& rho, const Real& u,
                                           const EMHD_parameters& emhd_params, const Real& gam,
                                           const int& k, const int& j, const int& i,
                                           Real& tau, Real& chi_e, Real& nu_e)
{
    if (emhd_params.type == ClosureType::constant) {
        // Set tau, nu, chi to constants
        tau   = emhd_params.tau;
        chi_e = emhd_params.conduction_alpha;
        nu_e  = emhd_params.viscosity_alpha;
    } else if (emhd_params.type == ClosureType::soundspeed) {
        // Set tau=const, chi/nu prop. to sound speed squared
        const Real cs2 = (gam * (gam - 1.) * u) / (rho + (gam * u));
        tau   = emhd_params.tau;
        chi_e = emhd_params.conduction_alpha * cs2 * tau;
        nu_e  = emhd_params.viscosity_alpha * cs2 * tau;
    } // else yell?
}

/**
 * Get a row of the EMHD stress-energy tensor with first index up, second index down.
 * A factor of sqrt(4 pi) is absorbed into the definition of b.
 * Note this must be passed the full q, dP, not the primitive prims.q, usually denote qtilde
 *
 * Entirely local!
 */
KOKKOS_INLINE_FUNCTION void calc_tensor(const Real& rho, const Real& u, const Real& pgas,
                                        const Real& q, const Real& dP,
                                        const FourVectors& D, const int& dir,
                                        Real emhd[GR_DIM])
{
    const Real bsq = max(dot(D.bcon, D.bcov), SMALL);
    const Real eta = pgas + rho + u + bsq;
    const Real ptot = pgas + 0.5 * bsq;

    DLOOP1 {
        emhd[mu] = eta * D.ucon[dir] * D.ucov[mu]
                  + ptot * (dir == mu)
                  - D.bcon[dir] * D.bcov[mu]
                  + (q / sqrt(bsq)) * ((D.ucon[dir] * D.bcov[mu]) +
                                       (D.bcon[dir] * D.ucov[mu]))
                  - dP * ((D.bcon[dir] * D.bcov[mu] / bsq)
                          - (1./3) * ((dir == mu) + D.ucon[dir] * D.ucov[mu]));
    }
}

// Convert q_tilde and dP_tilde (which are primitives) to q and dP
// This is required because the stress-energy tensor depends on q and dP
KOKKOS_INLINE_FUNCTION void convert_prims_to_q_dP(const Real& q_tilde, const Real& dP_tilde,
                                        const Real& rho, const Real& Theta, 
                                        const Real& tau, const Real& chi_e, const Real& nu_e,
                                        const EMHD_parameters& emhd_params, Real& q, Real& dP)
{
    q  = q_tilde;
    dP = dP_tilde;

    if (emhd_params.higher_order_terms) {
        q  *= sqrt(chi_e * rho * pow(Theta, 2) /tau);
        dP *= sqrt(nu_e * rho * Theta /tau);
    }
}

// Convert q and dP to q_tilde and dP_tilde (which are primitives)
// This is required because,
//          1. The source terms contain q0_tilde and dP0_tilde
//          2. Initializations MAY require converting q and dP to q_tilde and dP_tilde
KOKKOS_INLINE_FUNCTION void convert_q_dP_to_prims(const Real& q, const Real& dP,
                                        const Real& rho, const Real& Theta, 
                                        const Real& tau, const Real& chi_e, const Real& nu_e,
                                        const EMHD_parameters& emhd_params, Real& q_tilde, Real& dP_tilde)
{
    q_tilde  = q;
    dP_tilde = dP;

    if (emhd_params.higher_order_terms) {
        q_tilde  *= sqrt(tau / (chi_e * rho * pow(Theta, 2)) );
        dP_tilde *= sqrt(tau / (nu_e * rho * Theta) );
    }
}

} // namespace EMHD
