/* 
 *  File: b_cd_glm.hpp
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

#include "b_cd_glm_functions.hpp"

using namespace parthenon;

/**
 * This physics package implements B field transport with Flux-CT (Toth 2000)
 *
 * This requires only the values at cell centers
 * 
 * This implementation includes conversion from "primitive" to "conserved" B and back
 */
namespace B_CD_GLM {
/**
 * Declare fields, initialize (few) parameters
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived".
 * Also applies floors to the calculated primitives, and fixes up any inversion errors
 *
 * input: Conserved B = sqrt(-gdet) * B^i
 * output: Primitive B = B^i
 */
void UtoP(MeshBlockData<Real> *rc);

/**
 * Add the source term.  This has to be performed *after* the fluxes dUdt are applied to U
 */
TaskStatus AddSource(MeshBlockData<Real> *rc, const Real& dt);

/**
 * Calculate 2o approximation of max divB via central finite differencing
 */
Real MaxDivB(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::interior);

/**
 * Calculate max "divB" as reconstructed from psi
 */
Real MaxDivB_psi(MeshBlockData<Real> *rc0, MeshBlockData<Real> *rc1, const Real& dt, IndexDomain domain=IndexDomain::interior);

/**
 * Diagnostics printed/computed after each step
 * Currently just max divB
 */
TaskStatus PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime& tm);

/**
 * Fill fields which are calculated only for output to file
 * Currently nothing, soon the corner-centered divB values
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);
}