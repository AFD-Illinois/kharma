/* 
 *  File: kharma.hpp
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

#include "decs.hpp"
#include "types.hpp"

/**
 * General preferences for KHARMA.  Anything semi-driver-independent, like loading packages, etc.
 */
namespace KHARMA {
/**
 * This function messes with all Parthenon's parameters in-place before we hand them to the Mesh,
 * so that KHARMA decks can omit/infer some things parthenon needs.
 * This includes boundaries in spherical coordinates, coordinate system translations, etc.
 * This function also handles setting parameters from restart files
 */
void FixParameters(std::unique_ptr<ParameterInput>& pin);

/**
 * Load any packages specified in the input parameters
 */
Packages_t ProcessPackages(std::unique_ptr<ParameterInput>& pin);

/**
 * Initialize a "package" (StateDescriptor) of global variables, quantities needed randomly in several places, like:
 * dt_last, last step time
 * ctop_max, maximum speed on the grid
 * in_loop, whether one step has been completed (for e.g. EstimateTimestep)
 */
std::shared_ptr<StateDescriptor> InitializeGlobals(ParameterInput *pin);
// Version for restarts, called in PostInitialize if we're restarting from a Parthenon restart file
void ResetGlobals(ParameterInput *pin, Mesh *pmesh);

/**
 * Imitate Parthenon's FillDerived call, but on only a subset of zones defined by 'domain'
 * Used for boundary calls, see boundaries.cpp
 */
void FillDerivedDomain(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, int coarse);

/**
 * Code-wide work before each step in the fluid evolution.  Currently just updates globals.
 */
void PreStepMeshUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm);

/**
 * Code-wide work after each step in the fluid evolution.  Currently just updates globals.
 */
void PostStepMeshUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm);

/**
 * Calculate and print diagnostics after each step. Currently:
 * GRMHD: pflags & fflags, negative values in rho,u, ctop of 0 or NaN
 * B fields: MaxDivB
 */
void PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime &tm);

/**
 * Fill any arrays that are calculated only for output, e.g. divB, jcon, etc.
 * This calls the FillOutput function of each package
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);
}
