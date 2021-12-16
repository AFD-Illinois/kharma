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

#include "boundaries.hpp"
#include "debug.hpp"
#include "fixup.hpp"
#include "floors.hpp"
#include "fluxes.hpp"
#include "gr_coordinates.hpp"
#include "b_field_tools.hpp"

// Problem initialization headers
#include "bondi.hpp"
#include "explosion.hpp"
#include "fm_torus.hpp"
#include "iharm_restart.hpp"
#include "kelvin_helmholtz.hpp"
#include "bz_monopole.hpp"
#include "mhdmodes.hpp"
#include "orszag_tang.hpp"
#include "b_field_tools.hpp"

// Package headers
#include "mhd_functions.hpp"

#include "bvals/boundary_conditions.hpp"
#include "mesh/mesh.hpp"

using namespace parthenon;

void KHARMA::ProblemGenerator(MeshBlock *pmb, ParameterInput *pin)
{
    FLAG("Initializing Block");
    auto rc = pmb->meshblock_data.Get();

    // Breakout to call the appropriate initialization function,
    // defined in accompanying headers.

    auto prob = pin->GetString("parthenon/job", "problem_id"); // Required parameter
    if (prob == "mhdmodes") {
        InitializeMHDModes(rc.get(), pin);
    } else if (prob == "orszag_tang") {
        InitializeOrszagTang(rc.get(), pin);
    } else if (prob == "explosion") {
        InitializeExplosion(rc.get(), pin);
    } else if (prob == "kelvin_helmholtz") {
        InitializeKelvinHelmholtz(rc.get(), pin);
    } else if (prob == "bondi") {
        InitializeBondi(rc.get(), pin);
    } else if (prob == "torus") {
        InitializeFMTorus(rc.get(), pin);
    } else if (prob == "bz_monopole") {
        InitializeBZMonopole(rc.get(), pin);
    } else if (prob == "iharm_restart") {
        ReadIharmRestart(rc.get(), pin);
    }

    // Pertub the internal energy a bit to encourage accretion
    // option in perturbation->u_jitter
    // TODO evaluate determinism here. How are MeshBlock gids assigned?
    if (prob != "bz_monopole") {
        // TODO how should this work, especially with iharm_restarts ?
        PerturbU(rc.get(), pin);
    }

    // Apply any floors
    GRMHD::ApplyFloors(rc.get());

    // Fill the conserved variables U,
    // which we'll treat as the independent/fundamental state.
    // P is filled again from this later on
    Flux::PrimToFlux(rc.get(), IndexDomain::entire);

    FLAG("Initialized Block");
}
