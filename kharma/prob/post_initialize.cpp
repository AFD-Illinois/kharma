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

#include "b_field_tools.hpp"
#include "b_cleanup.hpp"
#include "blob.hpp"
#include "boundaries.hpp"
#include "debug.hpp"
#include "fixup.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "kharma.hpp"
#include "types.hpp"

#include "seed_B_ct.hpp"
#include "seed_B_cd.hpp"

void SyncAllBounds(Mesh *pmesh)
{
    // Honestly, the easiest way through this sync is:
    // 1. PtoU everywhere
    // 2. Sync like a normal step, incl. physical bounds
    // 3. UtoP everywhere
    // Luckily we're amortized over the whole sim, so we can
    // take our time.

    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();
        Flux::PtoU(rc.get(), IndexDomain::entire);
    }

    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();
        rc->ClearBoundary(BoundaryCommSubset::mesh_init);
        rc->StartReceiving(BoundaryCommSubset::mesh_init);
        rc->SendBoundaryBuffers();
    }

    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();
        rc->ReceiveAndSetBoundariesWithWait();
        rc->ClearBoundary(BoundaryCommSubset::mesh_init);
        //pmb->pbval->ProlongateBoundaries();

        // Fill P again, including ghost zones
        parthenon::Update::FillDerived(rc.get());

        // Physical boundary conditions
        parthenon::ApplyBoundaryConditions(rc);
    }
}

void KHARMA::SeedAndNormalizeB(ParameterInput *pin, Mesh *pmesh)
{
    // Add the field for torus problems as a second pass
    // Preserves P==U and ends with all physical zones fully defined
    if (pin->GetOrAddString("b_field", "type", "none") != "none") {
        // Calculating B has a stencil outside physical zones
        Flag("Extra boundary sync for B");
        SyncAllBounds(pmesh);

        // "Legacy" is the much more common normalization:
        // It's the ratio of max values over the domain i.e. max(P) / max(P_B),
        // not necessarily a local min(beta)
        Real beta_calc_legacy = pin->GetOrAddBoolean("b_field", "legacy", true);

        // Use the correct seed function based on field constraint solver
        const bool use_b_flux_ct = pmesh->packages.AllPackages().count("B_FluxCT");
        const bool use_b_cd = pmesh->packages.AllPackages().count("B_CD");

        Flag("Seeding magnetic field");
        // Seed the magnetic field and find the minimum beta
        Real beta_min = 1.e100, p_max = 0., bsq_max = 0.;
        for (auto &pmb : pmesh->block_list) {
            auto& rc = pmb->meshblock_data.Get();

            // This initializes B_P & B_U
            if (use_b_flux_ct) {
                B_FluxCT::SeedBField(rc.get(), pin);
            } else if (use_b_cd) {
                B_CD::SeedBField(rc.get(), pin);
            }

            // TODO should this be added after normalization?
            // TODO option to add flux slowly during the run?
            // Real BHflux = pin->GetOrAddReal("b_field", "bhflux", 0.0);
            // if (BHflux > 0.) {
            //     if (use_b_flux_ct) {
            //         B_FluxCT::SeedBHFlux(rc.get(), pin);
            //     } else if (use_b_cd) {
            //         B_CD::SeedBHFlux(rc.get(), pin);
            //     }
            // }

            if (beta_calc_legacy) {
                Real bsq_local = GetLocalBsqMax(rc.get());
                if(bsq_local > bsq_max) bsq_max = bsq_local;
                Real p_local = GetLocalPMax(rc.get());
                if(p_local > p_max) p_max = p_local;
            } else {
                Real beta_local = GetLocalBetaMin(rc.get());
                if(beta_local < beta_min) beta_min = beta_local;
            }
        }

        // Then, if we're in a torus problem or explicitly ask for it,
        // normalize the magnetic field according to the density
        auto prob = pin->GetString("parthenon/job", "problem_id");
        if (pin->GetOrAddBoolean("b_field", "norm", (prob == "torus"))) {
            // Default to the general literature beta_min of 100.
            // As noted above, by default this uses the definition max(P)/max(P_B)!
            Real desired_beta_min = pin->GetOrAddReal("b_field", "beta_min", 100.);

            // Calculate current beta_min value
            if (beta_calc_legacy) {
                bsq_max = MPIMax(bsq_max);
                p_max = MPIMax(p_max);
                beta_min = p_max / (0.5 * bsq_max);
            } else {
                beta_min = MPIMin(beta_min);
            }

            if (pin->GetInteger("debug", "verbose") > 0) {
                if (MPIRank0())
                    cerr << "Beta min pre-norm: " << beta_min << endl;
            }

            // Then normalize B by sqrt(beta/beta_min)
            Flag("Normalizing magnetic field");
            if (beta_min > 0) {
                Real norm = sqrt(beta_min/desired_beta_min);
                for (auto &pmb : pmesh->block_list) {
                    auto& rc = pmb->meshblock_data.Get();
                    NormalizeBField(rc.get(), norm);
                }
            }
        }

        if (pin->GetInteger("debug", "verbose") > 0) {
            // Measure again to check, and add divB for good measure
            beta_min = 1e100; p_max = 0.; bsq_max = 0.;
            for (auto &pmb : pmesh->block_list) {
                auto& rc = pmb->meshblock_data.Get();

                if (beta_calc_legacy) {
                    Real bsq_local = GetLocalBsqMax(rc.get());
                    if(bsq_local > bsq_max) bsq_max = bsq_local;
                    Real p_local = GetLocalPMax(rc.get());
                    if(p_local > p_max) p_max = p_local;
                } else {
                    Real beta_local = GetLocalBetaMin(rc.get());
                    if(beta_local < beta_min) beta_min = beta_local;
                }
            }
            if (beta_calc_legacy) {
                bsq_max = MPIMax(bsq_max);
                p_max = MPIMax(p_max);
                beta_min = p_max / (0.5 * bsq_max);
            } else {
                beta_min = MPIMin(beta_min);
            }
            // divB is implemented over a MeshBlockPack because it is fancy
            auto md = pmesh->mesh_data.GetOrAdd("base", 0).get();
            Real divb_max = 0.;
            if (use_b_flux_ct) {
                divb_max = B_FluxCT::MaxDivB(md);
            } else if (use_b_cd) {
                divb_max = B_CD::MaxDivB(md);
            }
            divb_max = MPIMax(divb_max);
            if (MPIRank0()) {
                cerr << "Beta min post-norm: " << beta_min << endl;
                cerr << "Max divB post-norm: " << divb_max << endl;
            }
        }

    }
    Flag("Added B Field");
}

void KHARMA::PostInitialize(ParameterInput *pin, Mesh *pmesh, bool is_restart, bool is_resize)
{
    Flag("Post-initialization started");
    if (!is_restart)
        KHARMA::SeedAndNormalizeB(pin, pmesh);

    if (pin->GetOrAddBoolean("blob", "add_blob", false)) {
        for (auto &pmb : pmesh->block_list) {
            auto rc = pmb->meshblock_data.Get();
            // This inserts only in vicinity of some global r,th,phi
            InsertBlob(rc.get(), pin);
        }
    }

    // Sync to fill the ghost zones
    Flag("Boundary sync");
    SyncAllBounds(pmesh);

    // TODO when (restart/non) do we need this for setting ctop?
    if (is_restart) {

        // Parthenon restored our global data for us, but we don't always want that
        KHARMA::ResetGlobals(pin, pmesh);

        // If we resized the array, cleanup any field divergence we created
        if (is_resize) {
            // Cleanup operates on full single MeshData as there are MPI syncs
            auto &mbase = pmesh->mesh_data.GetOrAdd("base", 0);
            // Clean field divergence across the whole grid
            B_Cleanup::CleanupDivergence(mbase);
            // Sync to make sure periodic boundaries are set
            Flag("Boundary sync");
            SyncAllBounds(pmesh);
        }

        // TODO anything special for imex driver here?
        // TODO there was a reconstruction here for filling ctop, but
        // it should definitely not be necessary as first dt is set with dt_first
    }

    Flag("Post-initialization finished");
}
