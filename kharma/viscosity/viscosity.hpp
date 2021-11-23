/* 
 *  File: viscosity.hpp
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

#include "mhd_functions.hpp"

using namespace parthenon;

/**
 * This physics package may someday implement viscosity.  It doesn't yet!
 */
namespace Viscosity {
/**
 * Initialization: declare any fields this package will evolve, initialize any parameters
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);

/**
 * In addition to the standard functions, packages can include extras.  This is called manually
 * at the end of problem initialization in problem.cpp
 */
TaskStatus InitElectrons(MeshBlockData<Real> *rc, ParameterInput *pin);

/**
 * Determine the primitive variable values, given conserved forms
 * This is where the implicit kernel will likely be placed, as each solve is per-cell after fluxes
 * and boundaries.
 * 
 * TODO make this replace GRMHD::UtoP or make it step out of the way
 */
void UtoP(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerived(MeshBlockData<Real> *rc) { UtoP(rc); }

/**
 * Floors, fixes, or other cleaning up after determining primitives.
 */
void PostUtoP(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void PostFillDerived(MeshBlockData<Real> *rc) { PostUtoP(rc); }

/**
 * Diagnostics printed/computed after each step, called from kharma.cpp
 * 
 * Function in this package: Currently nothing
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc);

/**
 * Fill fields which are calculated only for output to dump files
 * 
 * Function in this package: Currently nothing
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);

/**
 * KHARMA requires two forms of the function for obtaining conserved variables from primitives.
 * However, these are very different from UtoP/FillDerived in that they are called exclusively on the
 * device side, operating on a single zone rather than the whole fluid state.
 * 
 * Each should have roughly the signature used here, accepting scratchpads of size NVARxN1, and index
 * maps (see types.hpp) indicating which index corresponds to which variable in the packed array, as well
 * as indications of the desired zone location and flux direction (dir==0 for just the conserved variable forms).
 * As used extensively here, any variables not present in a pack will have index -1 in the map.
 *  
 * The two functions differ in two ways:
 * 1. The caller precalculate the four-vectors (u^mu, b^mu) and pass them in the struct D to prim_to_flux (see fluxes.hpp for call)
 * 2. p_to_u will only ever be called to obtain the conserved variables U, not fluxes (i.e. dir == 0 in calls)
 * 
 * Function in this package: primitive to flux/conserved transformation of conduction term q, pressure anisotropy dP
 */
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const ScratchPad2D<Real>& P, const VarMap& m_p, const FourVectors D,
                                         const int& k, const int& j, const int& i, const int dir,
                                         ScratchPad2D<Real>& flux, const VarMap m_u, const Loci loc=Loci::center)
{
    // Calculate flux through a face from primitives
}
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                         const int& k, const int& j, const int& i,
                                         const VariablePack<Real>& flux, const VarMap m_u, const Loci loc=Loci::center)
{
    // Calculate conserved variables from primitives
}

}
