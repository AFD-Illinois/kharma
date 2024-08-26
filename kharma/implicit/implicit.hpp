/* 
 *  File: implicit.hpp
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
#include "types.hpp"

#include "emhd_sources.hpp"
#include "emhd.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"


// This class calls EMHD stuff a bunch,
// since that's the only package with specific
// implicit solver stuff
using namespace EMHD;

// And an odd but useful loop for ex-iharm3d code
// This requires nvar to be defined in caller!
// It is not a const/global anymore.  So, use this loop carefully
#define PLOOP for(int ip=0; ip < nvar; ++ip)

// Version of PLOOP for just implicit ("fluid") variables
#define FLOOP for(int ip=0; ip < nfvar; ++ip)

namespace Implicit
{

// Denote implicit solver failures (solve_fail). 
// Thrown from Implicit::Step
// Status values:
// `converged`: solver converged to prescribed tolerance
// `fail`: manual backtracking wasn't good enough. FixSolve will be called
// `beyond_tol`: solver didn't converge to prescribed tolerance but didn't fail
// `backtrack`: step length of 1 gave negative rho/uu, but manual backtracking (0.1) sufficed
enum class SolverStatus{converged=0, fail, beyond_tol, backtrack};

static const std::map<int, std::string> status_names = {
    {(int) SolverStatus::fail, "failed"},
    {(int) SolverStatus::beyond_tol, "beyond tolerance"},
    {(int) SolverStatus::backtrack, "backtrack"}
};

template <typename T>
KOKKOS_INLINE_FUNCTION bool failed(T status_flag)
{
    // Return only zones which outright failed
    return static_cast<int>(status_flag) == static_cast<int>(SolverStatus::fail);
}

/**
 * Initialization.  Set parameters.
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * @brief take the per-zone implicit portion of a semi-implicit scheme
 * 
 * @param md_full_step_init the fluid state at the beginning of the step
 * @param md_sub_step_init the initial fluid state for this substep
 * @param md_flux_src the negative flux divergence plus explicit source terms
 * @param md_solver should contain initial guess on call, contains result on return
 * @param md_linesearch should contain solver prims at start, updated in the linesearch
 * @param dt the timestep (current substep)
 */
TaskStatus Step(MeshData<Real> *md_full_step_init, MeshData<Real> *md_sub_step_init, MeshData<Real> *md_flux_src,
                MeshData<Real> *md_linesearch, MeshData<Real> *md_solver, const Real& dt);

/**
 * Get the names of all variables matching 'flag' in a deterministic order, placing implicitly-evolved variables first.
 */
std::vector<std::string> GetOrderedNames(MeshBlockData<Real> *rc, const MetadataFlag& flag, bool only_implicit=false);


/**
 * @brief Fix bad zones that the implicit solver couldn't integrate. Similar to GRMHD::FixUtoP
 * 
 * @param mbd relevant fluid state
 * @return TaskStatus 
 */
TaskStatus FixSolve(MeshBlockData<Real> *mbd);
inline TaskStatus MeshFixSolve(MeshData<Real> *md) {
    Flag("MeshFixSolve");
    for (int i=0; i < md->NumBlocks(); ++i)
        FixSolve(md->GetBlockData(i).get());
    EndFlag();
    return TaskStatus::complete;
}

/**
 * Count up all nonzero solver flags on md.  Used for history file reductions.
 */
int CountSolverFails(MeshData<Real> *md);

/**
 * Print diagnostics about number of failed solves
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md);

/**
 * Calculate the residual generated by the trial primitives P_test
 * 
 * "Global" here are read-only input arrays addressed var(ip, k, j, i)
 * "Local" here is anything sliced (usually Scratch) addressable var(ip)
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_residual(const GRCoordinates& G, const Local& P_test,
                                          const Local& Pi, const Local& Ui, const Local& Ps,
                                          const Local& dudt_explicit, const Local& dUi, const Local& tmp, 
                                          const VarMap& m_p, const VarMap& m_u, const EMHD_parameters& emhd_params,
                                          const EMHD_parameters& emhd_params_s,const int& nfvar, 
                                          const int& k, const int& j, const int& i, 
                                          const Real& gam, const double& dt, Local& residual)
{
    // These lines calculate res = (U_test - Ui)/dt - dudt_explicit - 0.5*(dU_new(ip) + dUi(ip)) - dU_time(ip) )
    // Start with conserved vars corresponding to test P, U_test
    // Note this uses the Flux:: call, it needs *all* conserved vars!
    Flux::p_to_u(G, P_test, m_p, emhd_params, gam, j, i, tmp, m_u); // U_test
    // (U_test - Ui)/dt - dudt_explicit ...
    FLOOP residual(ip) = (tmp(ip) - Ui(ip)) / dt - dudt_explicit(ip);

    if (m_p.Q >= 0 || m_p.DP >= 0) {
        // Compute new implicit source terms and time derivative source terms
        Real dUq, dUdP; // Don't need full array for these
        EMHD::implicit_sources(G, P_test, Ps, m_p, gam, k, j, i, emhd_params_s, dUq, dUdP); // dU_new
        // ... - 0.5*(dU_new(ip) + dUi(ip)) ...
        if (emhd_params.conduction)
            residual(m_u.Q) -= 0.5*(dUq + dUi(m_u.Q));
        if (emhd_params.viscosity)
            residual(m_u.DP) -= 0.5*(dUdP + dUi(m_u.DP));

        EMHD::time_derivative_sources(G, P_test, Pi, Ps, m_p, emhd_params_s, gam, dt, k, j, i, dUq, dUdP); // dU_time
        // ... - dU_time(ip)
        if (emhd_params.conduction)
            residual(m_u.Q) -= dUq;
        if (emhd_params.viscosity)
            residual(m_u.DP) -= dUdP;

        // Normalize
        Real tau, chi_e, nu_e;
        EMHD::set_parameters(G, Ps, m_p, emhd_params_s, gam, j, i, tau, chi_e, nu_e);
        if (emhd_params.conduction)
            residual(m_u.Q) *= tau;
        if (emhd_params.viscosity)
            residual(m_u.DP) *= tau;
        if (emhd_params.higher_order_terms) {
            Real rho   = Ps(m_p.RHO);
            Real uu    = Ps(m_p.UU);
            Real Theta = (gam - 1.) * uu / rho;

            if (emhd_params.conduction)
                residual(m_u.Q) *= (chi_e != 0) ? m::sqrt(rho * chi_e * tau * Theta * Theta) / tau : 1.;
            if (emhd_params.viscosity)
                residual(m_u.DP) *= (nu_e != 0) ? m::sqrt(rho * nu_e * tau * Theta) / tau : 1.;
        }
    }

}

/**
 * Evaluate the jacobian for the implicit iteration, in one zone
 * 
 * Local is anything addressable by (0:nvar-1), Local2 is the same for 2D (0:nvar-1, 0:nvar-1)
 * Usually these are Kokkos subviews
 */
template<typename Local, typename Local2>
KOKKOS_INLINE_FUNCTION void calc_jacobian(const GRCoordinates& G, const Local& P_solver,
                                          const Local& P_full_step_init, const Local& U_full_step_init, const Local& P_sub_step_init,
                                          const Local& flux_src, const Local& dU_implicit, Local& tmp1, Local& tmp2, Local& tmp3,
                                          const VarMap& m_p, const VarMap& m_u, const EMHD_parameters& emhd_params_solver,
                                          const EMHD_parameters& emhd_params_sub_step_init, const int& nvar, const int& nfvar,
                                          const int& k, const int& j, const int& i,
                                          const Real& jac_delta, const Real& gam, const double& dt,
                                          Local2& jacobian, Local& residual)
{
    // Calculate residual of P
    calc_residual(G, P_solver, P_full_step_init, U_full_step_init, P_sub_step_init, flux_src, dU_implicit, tmp3,
                    m_p, m_u, emhd_params_solver, emhd_params_sub_step_init, nfvar, k, j, i, gam, dt, residual);

    // Use one scratchpad as the incremented prims P_delta,
    // one as the new residual residual_delta
    auto& P_delta        = tmp1;
    auto& residual_delta = tmp2;
    // set P_delta to P to begin with
    PLOOP P_delta(ip)    = P_solver(ip);

    // Numerically evaluate the Jacobian
    for (int col = 0; col < nfvar; col++) {
        // Compute P_delta, differently depending on whether the prims are small compared to eps
        if (m::abs(P_solver(col)) < (0.5 * jac_delta)) {
            P_delta(col) = P_solver(col) + jac_delta;
        } else {
            P_delta(col) = (1 + jac_delta) * P_solver(col);
        }

        // Compute the residual for P_delta, residual_delta
        calc_residual(G, P_delta, P_full_step_init, U_full_step_init, P_sub_step_init, flux_src, dU_implicit, tmp3, 
                    m_p, m_u, emhd_params_solver, emhd_params_sub_step_init, nfvar, k, j, i, gam, dt, residual_delta);

        // Compute forward derivatives of each residual vs the primitive col
        for (int row = 0; row < nfvar; row++) {
            jacobian(row, col) = (residual_delta(row) - residual(row)) / (P_delta(col) - P_solver(col) + SMALL_NUM);
        }

        // Reset P_delta in this col
        P_delta(col) = P_solver(col);

    }
}   

} // namespace Implicit
