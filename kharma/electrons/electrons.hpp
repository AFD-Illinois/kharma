/* 
 *  File: electrons.hpp
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

#include <memory>

#include <parthenon/parthenon.hpp>

#include "mhd_functions.hpp"

using namespace parthenon;

/**
 * This physics package implements self-consistent Electron heating and transport
 * as in Ressler+ 2015.
 * Tracks fluid total entropy in order to calculate stepwise changes, and electron
 * entropy fed by some fraction of dissipation.
 * Supports several heating models (dissipation fractions) via runtime parameters.
 */
namespace Electrons {
/**
 * Declare fields, initialize (few) parameters
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);

/**
 * Initialize electron temperatures when setting up the problem. Trivial.
 */
TaskStatus InitElectrons(MeshBlockData<Real> *rc, ParameterInput *pin);

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived".
 * Also applies floors to the calculated primitives, and fixes up any inversion errors
 */
void UtoP(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerived(MeshBlockData<Real> *rc) { UtoP(rc); }

/**
 * Floors for electron transport, mimics iharm3d:
 * fix NaN values (!) to Tp/Te maximum (KEL minimum)
 * Enforce Tp/Te minimum & maximum from parameters
 * 
 * TODO this should record floor hits to fflag!
 */
void PostUtoP(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void PostFillDerived(MeshBlockData<Real> *rc) { PostUtoP(rc); }

/**
 * Add the source term, heating the electrons based on entropy change/advection and updating the local entropy
 */
TaskStatus ApplyHeatingModels(MeshBlockData<Real> *rc_old, MeshBlockData<Real> *rc);

/**
 * Diagnostics printed/computed after each step
 * Currently nothing
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc);

/**
 * Fill fields which are calculated only for output to file
 * Currently nothing
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);

/**
 * Divide or multiply by local density to get entropy/particle
 */
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const ScratchPad2D<Real>& P, const VarMap& m_p, const FourVectors D,
                                         const int& k, const int& j, const int& i, const int dir,
                                         ScratchPad2D<Real>& flux, const VarMap m_u, const Loci loc=Loci::center)
{
    // Take the factor from the primitives, in case we need to reorder this to happen before GRMHD::prim_to_flux later
    Real rho_ut = P(m_p.RHO, i) * D.ucon[dir] * G.gdet(loc, j, i);
    flux(m_u.KTOT, i) = rho_ut * P(m_p.KTOT, i);
    if (m_p.K_HOWES >= 0)
        flux(m_u.K_HOWES, i) = rho_ut * P(m_p.K_HOWES, i);
    if (m_p.K_KAWAZURA >= 0)
        flux(m_u.K_KAWAZURA, i) = rho_ut * P(m_p.K_KAWAZURA, i);
    if (m_p.K_WERNER >= 0)
        flux(m_u.K_WERNER, i) = rho_ut * P(m_p.K_WERNER, i);
    if (m_p.K_ROWAN >= 0)
        flux(m_u.K_ROWAN, i) = rho_ut * P(m_p.K_ROWAN, i);
    if (m_p.K_SHARMA >= 0)
        flux(m_u.K_SHARMA, i) = rho_ut * P(m_p.K_SHARMA, i);
}
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                         const int& k, const int& j, const int& i,
                                         const VariablePack<Real>& flux, const VarMap m_u, const Loci loc=Loci::center)
{
    // Take the factor from the primitives, in case we need to reorder this to happen before GRMHD::prim_to_flux later
    Real ut = GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc) * sqrt(-G.gcon(loc, j, i, 0, 0));
    Real rho_ut = P(m_p.RHO, k, j, i) * ut * G.gdet(loc, j, i);

    flux(m_u.KTOT, k, j, i) = rho_ut * P(m_p.KTOT, k, j, i);
    if (m_p.K_HOWES >= 0)
        flux(m_u.K_HOWES, k, j, i) = rho_ut * P(m_p.K_HOWES, k, j, i);
    if (m_p.K_KAWAZURA >= 0)
        flux(m_u.K_KAWAZURA, k, j, i) = rho_ut * P(m_p.K_KAWAZURA, k, j, i);
    if (m_p.K_WERNER >= 0)
        flux(m_u.K_WERNER, k, j, i) = rho_ut * P(m_p.K_WERNER, k, j, i);
    if (m_p.K_ROWAN >= 0)
        flux(m_u.K_ROWAN, k, j, i) = rho_ut * P(m_p.K_ROWAN, k, j, i);
    if (m_p.K_SHARMA >= 0)
        flux(m_u.K_SHARMA, k, j, i) = rho_ut * P(m_p.K_SHARMA, k, j, i);
}

}
