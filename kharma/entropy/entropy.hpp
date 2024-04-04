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

#include "grmhd_functions.hpp"

using namespace parthenon;

/**
 * Small package to transport total entropy, providing it to other packages.
 * Currently, e- and force-free evolution use it.
 * 
 * Doubles as a simple example of adding an evolved field
 */
namespace Entropy {
/**
 * Initialization: declare any fields this package will evolve, initialize any parameters
 * 
 * This add a total entropy variable  prims.Ktot/cons.Ktot
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * In addition to the standard callback functions, a package can include arbitrary other functions.
 * This is called from InitElectrons and InitForceFree, where an entropy estimate is needed
 * *before* the first step.  Entropy, e-, and FF values are thereafter updated at step *end*
 */
TaskStatus InitEntropy(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin);

/**
 * Any implementation of UtoP needs to take an IndexDomain enum and boundary "coarse" boolean.
 * This allows KHARMA to call it over the whole domain (IndexDomain::entire) or just on a boundary
 * after conserved variables have been updated.
 * 
 * Function in this package: Get the specific entropy primitive value, by dividing the total entropy K/(rho*u^0)
 */
void BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse=false);

/**
 * Reverse of the above, recover conserved total K
 */
void BlockPtoU(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse=false);

/**
 * Update the entropy variable by computing the entropy based on GRMHD variables.
 * Done at the end of each sub-step to 
 */
TaskStatus UpdateEntropy(MeshBlockData<Real> *rc);

/**
 * Apply adjustments to KTOT.
 */
void ApplyFloors(MeshBlockData<Real> *mbd, IndexDomain domain);

/**
 * Diagnostics printed/computed after each step, called from kharma.cpp
 * 
 * Function in this package: Currently nothing
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc);

/**
 * Fill fields which are calculated only for output to dump files, called from kharma.cpp
 * 
 * Function in this package: Currently nothing
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);

/**
 * KHARMA requires some method for getting conserved variables from primitives, as well.
 * 
 * However, unlike UtoP, p_to_u is implemented device-side. That means that any
 * package defining new primitive/conserved vars must not only provide a prim_to_flux here,
 * but add it to the list in Flux::prim_to_flux.
 */
KOKKOS_FORCEINLINE_FUNCTION void p_to_u(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                         const int& k, const int& j, const int& i,
                                         const VariablePack<Real>& flux, const VarMap m_u, const Loci loc=Loci::center)
{
    // Take the factor from the primitives, in case we need to reorder this to happen before GRMHD::prim_to_flux later
    const Real ut = GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc) * m::sqrt(-G.gcon(loc, j, i, 0, 0));
    flux(m_u.KTOT, k, j, i) = P(m_p.RHO, k, j, i) * ut * G.gdet(loc, j, i) * P(m_p.KTOT, k, j, i);
}

} // namespace Entropy
