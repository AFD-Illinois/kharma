/* 
 *  File: kharma_package.hpp
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

#include <parthenon/parthenon.hpp>

using namespace parthenon;

/**
 * Adds a number of useful callbacks which KHARMA packages might want to take advantage of,
 * which may not make sense to add to the Parthenon StateDescriptor struct upstream
 * (or which simply haven't been added yet, for whatever reason)
 * 
 * KHARMA packages which handle variables evolved with Flux:: must additionally provide
 * some device-side functions.
 * 1. Package::prim_to_flux -- various calling conventions, see grmhd_functions.hpp
 * 
 */
class KHARMAPackage : public StateDescriptor {
    public:
        KHARMAPackage(std::string name) : StateDescriptor(name) {}

        // PHYSICS
        // Recovery of primitive variables from conserved.
        // These can be host-side functions because they are not called from the Uberkernel --
        // rather, they are called on zone center values once per step only.
        // Called by various Flux::*UtoP*
        std::function<void(MeshBlockData<Real>*, IndexDomain, bool)> BlockUtoP = nullptr;
        std::function<void(MeshData<Real>*, IndexDomain, bool)> MeshUtoP = nullptr;

        // Maybe at some point we'll have 
        // Since Flux::prim_to_flux must cover everything, it's not worth splitting now
        //std::function<void(MeshBlockData<Real>*, IndexDomain, bool)> BlockPtoU = nullptr;

        // Source term to add to the conserved variables during each step
        std::function<void(MeshData<Real>*, MeshData<Real>*)> AddSource = nullptr;

        // Source term to apply to primitive variables, needed for some problems in order
        // to control dissipation (Hubble, turbulence).
        // Must be applied over entire domain!
        std::function<void(MeshBlockData<Real>*)> BlockApplyPrimSource = nullptr;

        // Apply any fixes after the initial fluxes are calculated
        std::function<void(MeshData<Real>*)> FixFlux = nullptr;

        // Apply any floors or limiters specific to the package (that is, on the package's variables)
        // Called by Floors::*ApplyFloors
        std::function<void(MeshBlockData<Real>*, IndexDomain)> BlockApplyFloors = nullptr;
        std::function<void(MeshData<Real>*, IndexDomain)> MeshApplyFloors = nullptr;

        // CONVENIENCE
        // Anything to be done before each step begins -- currently just updating global "in_loop"
        std::function<void(Mesh*, ParameterInput*, const SimTime&)> MeshPreStepUserWorkInLoop = nullptr;
        // Anything to be done after every step is fully complete -- usually reductions or preservation of variables
        std::function<void(Mesh*, ParameterInput*, const SimTime&)> MeshPostStepUserWorkInLoop = nullptr;

        // Anything to be done just before any outputs (dump files, restarts, history files) are made
        // Usually for filling output-only variables
        // TODO Add MeshUserWorkBeforeOutput to Parthenon
        std::function<void(MeshBlock*, ParameterInput*)> BlockUserWorkBeforeOutput = nullptr;

        // BOUNDARIES
        // Currently only used by the "boundaries" package, or overridden during problem initialization
        // Note these functions take the boundary domain as an argument, so you can assign the same function to multiple boundaries.
        std::function<void(std::shared_ptr<MeshBlockData<Real>>&, IndexDomain, bool)> KHARMAInnerX1Boundary = nullptr;
        std::function<void(std::shared_ptr<MeshBlockData<Real>>&, IndexDomain, bool)> KHARMAOuterX1Boundary = nullptr;
        std::function<void(std::shared_ptr<MeshBlockData<Real>>&, IndexDomain, bool)> KHARMAInnerX2Boundary = nullptr;
        std::function<void(std::shared_ptr<MeshBlockData<Real>>&, IndexDomain, bool)> KHARMAOuterX2Boundary = nullptr;
};

/**
 * Implement the above callbacks
 */
namespace Packages {

/**
 * Any "fixes" to the fluxes through zone faces calculated by GetFlux.
 * These are all package-defined, with boundary fluxes and magnetic field transport
 * being the big cases.
 */
TaskStatus FixFlux(MeshData<Real> *md);

/**
 * 
 */
TaskStatus BlockUtoP(MeshBlockData<Real> *mbd, IndexDomain domain, bool coarse=false);
TaskStatus MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse=false);

/**
 * Fill the primitive variables P using the conserved U, for every package except "GRMHD."
 * That is, currently, B fields and electrons.
 * This is used for KHARMA's boundaries, which act on (e.g., reflect or outflow) the
 * conserved variables where available (and thus must recover primitives),
 * but act on primitive rho,u,uvec and must leave those alone.
 */
TaskStatus BlockUtoPExceptMHD(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse=false);
TaskStatus MeshUtoPExceptMHD(MeshData<Real> *md, IndexDomain domain, bool coarse=false);

/**
 * Fill all conserved variables (U) from primitive variables (P), over a whole block
 */
// TaskStatus BlockPtoU(MeshBlockData<Real> *mbd, IndexDomain domain, bool coarse=false);

/**
 * Add any source terms to the conserved variables.  Applied over the interior/physical zones only, as these
 * are the only ones well-defined in the only place this function is called.
 */
TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt);

/**
 * Add any source terms to the primitive variables.  Applied directly rather than adding to a derivative.
 */
TaskStatus BlockApplyPrimSource(MeshBlockData<Real> *rc);

/**
 * Apply all floors, including any package-specific limiters.
 * This function respects "disable_floors".
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
TaskStatus BlockApplyFloors(MeshBlockData<Real> *mbd, IndexDomain domain);
TaskStatus MeshApplyFloors(MeshData<Real> *md, IndexDomain domain);

// These are already Parthenon global callbacks -- see their documentation
// I define them here so I can pass them on to packages
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin);
void PreStepUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm);
void PostStepUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm);
void PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime &tm);
}