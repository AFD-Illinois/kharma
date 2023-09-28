/* 
 *  File: post_initialize.cpp
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

#include "post_initialize.hpp"

#include "b_cd.hpp"
#include "b_cleanup.hpp"
#include "b_ct.hpp"
#include "b_flux_ct.hpp"
#include "blob.hpp"
#include "boundaries.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "kharma_driver.hpp"
#include "reductions.hpp"
#include "seed_B.hpp"
#include "types.hpp"

void KHARMA::PostInitialize(ParameterInput *pin, Mesh *pmesh, bool is_restart)
{
    // This call:
    // 1. Initializes any magnetic fields which are "seeded," i.e., defined with a magnetic field implementation
    //    rather than assuming an implementation and setting the field with problem initialization.
    // 2. Renormalizes magnetic fields based on a desired ratio of maximum magnetic/gas pressures
    // 3. Adds any extra material which might be superimposed when restarting, e.g. "hotspot" regions a.k.a. "blobs"
    // 4. Resets a couple of incidental flags, if Parthenon read them from a restart file
    // 5. If necessary, cleans up any magnetic field divergence present on the grid

    // Coming into this function, at least the *interior* regions should be initialized with a problem:
    // that is, at least rho, u, uvec on each physical zone.
    // If you need Dirichlet boundary conditions, the domain-edge *ghost* zones should also be initialized,
    // as they will be "frozen in" during this function and applied thereafter.

    auto &md = pmesh->mesh_data.Get();

    auto& pkgs = pmesh->packages.AllPackages();

    // Magnetic field operations
    if (pin->GetString("b_field", "solver") != "none") {
        // If we need to seed a field based on the problem's fluid initialization...
        if (pin->GetOrAddString("b_field", "type", "none") != "none" && !is_restart) {
            // B field init is not stencil-1, needs boundaries sync'd.
            // FreezeDirichlet ensures any Dirichlet conditions aren't overwritten by zeros
            KBoundaries::FreezeDirichlet(md);
            KHARMADriver::SyncAllBounds(md);

            // Then init B field on each block...
            SeedBField(md.get(), pin);

            // If we're doing a torus problem or explicitly ask for it,
            // normalize the magnetic field according to the density
            bool is_torus = pin->GetString("parthenon/job", "problem_id") == "torus";
            if (pin->GetOrAddBoolean("b_field", "norm", is_torus)) {
                NormalizeBField(md.get(), pin);
            }
        }

        // Regardless, if evolving a field we should print max(divB)
        // divB is not stencil-1 and we may not have run the above.
        // If we did, we still need another sync, so it works out
        KBoundaries::FreezeDirichlet(md);
        KHARMADriver::SyncAllBounds(md);

        if (pkgs.count("B_FluxCT")) {
            B_FluxCT::PrintGlobalMaxDivB(md.get());
        } else if (pkgs.count("B_CT")) {
            B_CT::PrintGlobalMaxDivB(md.get());
        } else if (pkgs.count("B_CD")) {
            //B_CD::PrintGlobalMaxDivB(md.get());
        }
    }

    // Add any hotspots.
    // Note any other modifications made when restarting should be made around here
    if (pin->GetOrAddBoolean("blob", "add_blob", false)) {
        for (auto &pmb : pmesh->block_list) {
            auto rc = pmb->meshblock_data.Get();
            // This inserts only in vicinity of some global r,th,phi
            InsertBlob(rc.get(), pin);
        }
    }

    // Any extra cleanup & init especially when restarting
    if (is_restart) {
        // Parthenon restores all parameters (global vars) when restarting,
        // but KHARMA needs a few (currently one) reset instead
        KHARMA::ResetGlobals(pin, pmesh);
    }

    // Clean the B field if we've introduced a divergence somewhere
    // We call this function any time the package is loaded:
    // if we decided to load it in kharma.cpp, we need to clean.
    if (pkgs.count("B_Cleanup")) {
        if (pin->GetOrAddBoolean("b_cleanup", "output_before_cleanup", false)) {
            auto tm = SimTime(0., 0., 0, 0, 0, 0, 0.);
            auto pouts = std::make_unique<Outputs>(pmesh, pin, &tm);
            pouts->MakeOutputs(pmesh, pin, &tm, SignalHandler::OutputSignal::now);
        }

        // This does its own MPI syncs
        B_Cleanup::CleanupDivergence(md);
    }

    // Finally, synchronize boundary values.
    // Freeze any Dirichlet physical boundaries as they are now, after cleanup/sync/etc.
    KBoundaries::FreezeDirichlet(md);
    // This is the first sync if there is no B field
    KHARMADriver::SyncAllBounds(md);
    // And make sure the trivial primitive values are up-to-date
    //Packages::MeshUtoPExceptMHD(md.get(), IndexDomain::entire, false);
}
