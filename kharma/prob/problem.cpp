/* 
 *  File: problem.cpp
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

#include "problem.hpp"

#include "b_field_tools.hpp"
#include "boundaries.hpp"
#include "debug.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "types.hpp"

// Problem initialization headers
#include "bondi.hpp"
#include "driven_turbulence.hpp"
#include "explosion.hpp"
#include "fm_torus.hpp"
#include "resize_restart.hpp"
#include "kelvin_helmholtz.hpp"
#include "bz_monopole.hpp"
#include "mhdmodes.hpp"
#include "orszag_tang.hpp"
#include "shock_tube.hpp"
#include "hubble.hpp"
#include "noh.hpp"
// EMHD problem headers
#include "emhd/anisotropic_conduction.hpp"
#include "emhd/emhdmodes.hpp"
#include "emhd/emhdshock.hpp"
#include "emhd/conducting_atmosphere.hpp"
#include "emhd/bondi_viscous.hpp"
// TODO electron problem headers?

using namespace parthenon;

void KHARMA::ProblemGenerator(MeshBlock *pmb, ParameterInput *pin)
{
    auto rc = pmb->meshblock_data.Get();
    Flag(rc.get(), "Initializing Block");

    // Breakout to call the appropriate initialization function,
    // defined in accompanying headers.

    auto prob = pin->GetString("parthenon/job", "problem_id"); // Required parameter
    
    if (MPIRank0()) {
        std::cout << "Initializing problem: " << prob << std::endl;
    }
    TaskStatus status = TaskStatus::fail;
    // GRMHD
    if (prob == "mhdmodes") {
        status = InitializeMHDModes(rc.get(), pin);
    } else if (prob == "orszag_tang") {
        status = InitializeOrszagTang(rc.get(), pin);
    } else if (prob == "explosion") {
        status = InitializeExplosion(rc.get(), pin);
    } else if (prob == "kelvin_helmholtz") {
        status = InitializeKelvinHelmholtz(rc.get(), pin);
    } else if (prob == "shock") {
        status = InitializeShockTube(rc.get(), pin);
    } else if (prob == "bondi") {
        status = InitializeBondi(rc.get(), pin);
    } else if (prob == "bz_monopole") {
        status = InitializeBZMonopole(rc.get(), pin);
    // Electrons
    } else if (prob == "noh") {
        status = InitializeNoh(rc.get(), pin);
    } else if (prob == "hubble") {
        status = InitializeHubble(rc.get(), pin);
    } else if (prob == "driven_turbulence") {
        status = InitializeDrivenTurbulence(rc.get(), pin);
    // Extended GRMHD
    } else if (prob == "emhdmodes") {
        status = InitializeEMHDModes(rc.get(), pin);
    } else if (prob == "anisotropic_conduction") {
        status = InitializeAnisotropicConduction(rc.get(), pin);
    } else if (prob == "emhdshock") {
        status = InitializeEMHDShock(rc.get(), pin);
    } else if (prob == "conducting_atmosphere") {
        status = InitializeAtmosphere(rc.get(), pin);
    } else if (prob == "bondi_viscous") {
        status = InitializeBondiViscous(rc.get(), pin);
    // Everything
    } else if (prob == "torus") {
        status = InitializeFMTorus(rc.get(), pin);
    } else if (prob == "resize_restart") {
        status = ReadIharmRestart(rc.get(), pin);
    }

    // If we didn't initialize a problem, yell
    if (status != TaskStatus::complete) {
        throw std::invalid_argument("Invalid or incomplete problem: "+prob);
    }

    // If we're not restarting, do any grooming of the initial conditions
    if (prob != "resize_restart") {
        // Perturb the internal energy a bit to encourage accretion
        // Note this defaults to zero & is basically turned on only for torii
        if (pin->GetOrAddReal("perturbation", "u_jitter", 0.0) > 0.0) {
            PerturbU(rc.get(), pin);
        }

        // Initialize electron entropies to defaults if enabled
        if (pmb->packages.AllPackages().count("Electrons")) {
            Electrons::InitElectrons(rc.get(), pin);
        }
    }

    // Fill the conserved variables U,
    // which we'll treat as the independent/fundamental state.
    // P is filled again from this later on
    // Note this is needed *after* P is finalized, but
    // *before* the floor call: normal-observer floors need U populated
    Flux::PtoU(rc.get(), IndexDomain::interior);

    // If we're not restarting, apply the floors
    if (prob != "resize_restart") {
        // This is purposefully done even if floors are disabled,
        // as it is required for consistent initialization
        // Note however we do *not* preserve any inversion flags in this call.
        // There will be subsequent renormalization and re-inversion that will
        // initialize those flags.
        Floors::ApplyFloors(rc.get(), IndexDomain::interior);
    }

    Flag(rc.get(), "Initialized Block");
}
