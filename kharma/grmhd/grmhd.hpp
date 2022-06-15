/* 
 *  File: grmhd.hpp
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

using namespace parthenon;

/**
 * This physics package implements General-Relativistic Magnetohydrodynamics
 *
 * Anything specific to GRMHD (but not relating to the particular *order* of operations)
 * is implemented in this namespace, in the files grmhd.cpp, source.cpp, and fixup.cpp.
 * 
 * 
 */
namespace GRMHD {
// For declaring meshes, as well as the full intermediates we need (right & left fluxes etc)
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);

/**
 * Get the primitive variables
 * This just computes P, and only for the fluid varaibles.
 * Other packages must convert P->U by registering their version as "FillDerived"
 *
 * Defaults to entire domain, as the KHARMA algorithm relies on applying UtoP over ghost zones.
 * 
 * input: U, whatever form
 * output: U and P match down to inversion errors
 */
// void UtoP(MeshData<Real> *md, IndexDomain domain=IndexDomain::entire, bool coarse=false);
// inline void FillDerivedMesh(MeshData<Real> *md) { UtoP(md); }
void UtoP(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerivedBlock(MeshBlockData<Real> *rc) { UtoP(rc); }
inline TaskStatus FillDerivedBlockTask(MeshBlockData<Real> *rc) { UtoP(rc); return TaskStatus::complete; }

/**
 * Smooth over inversion failures by averaging values from each neighboring zone
 * a.k.a. Diffusion?  What diffusion?  There is no diffusion here.
 * 
 * LOCKSTEP: this function expects and should preserve P<->U
 */
TaskStatus FixUtoP(MeshBlockData<Real> *rc);
/**
 * Fix the primitive variables
 * Applies floors to the calculated primitives, and fixes up any failed inversions
 *
 * input: U & P, "matching"
 * output: U and P match with inversion errors corrected, and obey floors
 */
void PostUtoP(MeshBlockData<Real> *rc);

/**
 * Function to apply the GRMHD source term over the entire grid.
 * 
 * Note Flux::ApplyFluxes = parthenon::FluxDivergence + GRMHD::AddSource
 */
TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt);

/**
 * Returns the minimum CFL timestep among all zones in the block,
 * multiplied by a proportion "cfl" for safety.
 *
 * This is just for a particular MeshBlock/package, so don't rely on it
 * Parthenon will take the minimum and put it in pmy_mesh->dt
 */
Real EstimateTimestep(MeshBlockData<Real> *rc);

// Internal version for the light phase speed crossing time of smallest zone
Real EstimateRadiativeTimestep(MeshBlockData<Real> *rc);

/**
 * Return a tag per-block indicating whether to refine it
 * 
 * Criteria are very WIP
 */
AmrTag CheckRefinement(MeshBlockData<Real> *rc);

/**
 * Fill fields which are calculated only for output to file
 * Currently just the current jcon
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);

/**
 * Diagnostics performed after each step.
 * Currently finds any negative flags or 0/NaN values in ctop
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc);
}
