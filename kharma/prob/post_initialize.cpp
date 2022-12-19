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
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "mpi.hpp"
#include "types.hpp"

#include "seed_B_ct.hpp"
#include "seed_B_cd.hpp"

void KHARMA::SeedAndNormalizeB(ParameterInput *pin, std::shared_ptr<MeshData<Real>> md)
{
    // Check which solver we'll be using
    auto pmesh = md->GetMeshPointer();
    const bool use_b_flux_ct = pmesh->packages.AllPackages().count("B_FluxCT");
    const bool use_b_cd = pmesh->packages.AllPackages().count("B_CD");

    // Add the field for torus problems as a second pass
    // Preserves P==U and ends with all physical zones fully defined
    if (pin->GetOrAddString("b_field", "type", "none") != "none") {
        // Calculating B has a stencil outside physical zones
        Flag("Extra boundary sync for B");
        KBoundaries::SyncAllBounds(md);

        // "Legacy" is the much more common normalization:
        // It's the ratio of max values over the domain i.e. max(P) / max(P_B),
        // not necessarily a local min(beta)
        Real beta_calc_legacy = pin->GetOrAddBoolean("b_field", "legacy", true);

        Flag("Seeding magnetic field");
        // Seed the magnetic field and find the minimum beta
        Real beta_min = 1.e100, p_max = 0., bsq_max = 0., bsq_min = 0.;
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
                bsq_local = GetLocalBsqMin(rc.get());
                if(bsq_local < bsq_min) bsq_min = bsq_local;
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
                bsq_max = MPIReduce_once(bsq_max, MPI_MAX);
                bsq_min = MPIReduce_once(bsq_min, MPI_MIN);
                p_max = MPIReduce_once(p_max, MPI_MAX);
                beta_min = p_max / (0.5 * bsq_max);
            } else {
                beta_min = MPIReduce_once(beta_min, MPI_MIN);
            }

            if (pin->GetInteger("debug", "verbose") > 0) {
                if (MPIRank0()) {
                    std::cerr << "bsq_max pre-norm: " << bsq_max << std::endl;
                    std::cerr << "bsq_min pre-norm: " << bsq_min << std::endl;
                    std::cerr << "Beta min pre-norm: " << beta_min << std::endl;
                }
            }

            // Then normalize B by sqrt(beta/beta_min)
            Flag("Normalizing magnetic field");
            if (beta_min > 0) {
                Real norm = m::sqrt(beta_min/desired_beta_min);
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
                bsq_max = MPIReduce_once(bsq_max, MPI_MAX);
                p_max = MPIReduce_once(p_max, MPI_MAX);
                beta_min = p_max / (0.5 * bsq_max);
            } else {
                beta_min = MPIReduce_once(beta_min, MPI_MIN);
            }
            if (MPIRank0()) {
                std::cerr << "bsq_max post-norm: " << bsq_max << std::endl;
                std::cerr << "Beta min post-norm: " << beta_min << std::endl;
            }
        }
    }

    Flag("Added B Field");
}

void KHARMA::PostInitialize(ParameterInput *pin, Mesh *pmesh, bool is_restart, bool is_resize)
{
    Flag("Post-initialization started");

    // Make sure we've built the MeshData object we'll be synchronizing/updating
    auto &md = pmesh->mesh_data.GetOrAdd("base", 0);

    if (!is_restart)
        KHARMA::SeedAndNormalizeB(pin, md);

    if (pin->GetString("b_field", "solver") != "none") {
        // Synchronize our seeded or initialized field (incl. primitives) before we print out what divB it has
        KBoundaries::SyncAllBounds(md);

        const bool use_b_flux_ct = pmesh->packages.AllPackages().count("B_FluxCT");
        const bool use_b_cd = pmesh->packages.AllPackages().count("B_CD");

        // Still print divB, even if we're not initializing/normalizing field here
        if (use_b_flux_ct) {
            B_FluxCT::PrintGlobalMaxDivB(md.get());
        } // TODO B_CD version
    }

    if (pin->GetOrAddBoolean("blob", "add_blob", false)) {
        for (auto &pmb : pmesh->block_list) {
            auto rc = pmb->meshblock_data.Get();
            // This inserts only in vicinity of some global r,th,phi
            InsertBlob(rc.get(), pin);
        }
    }

    // Sync to fill the ghost zones: prims for ImExDriver, everything for HARMDriver
    Flag("Boundary sync");
    KBoundaries::SyncAllBounds(md);

    // Extra cleanup & init to do if restarting
    if (is_restart) {
        // Parthenon restored our global data for us, but we don't always want that
        KHARMA::ResetGlobals(pin, pmesh);
    }

    // If we resized the array, cleanup any field divergence we created
    // Let the user specify to do this, too
    if ((is_restart && is_resize && !pin->GetOrAddBoolean("resize_restart", "skip_b_cleanup", false))
        || pin->GetBoolean("b_field", "initial_cleanup")) {
        // Clean field divergence across the whole grid
        // Includes boundary syncs
        B_Cleanup::CleanupDivergence(md);
    }

    if (MPIRank0()) {
        std::cout << "Packages in use: " << std::endl;
        for (auto pkg : pmesh->packages.AllPackages()) {
            std::cout << pkg.first << std::endl;
        }
        std::cout << std::endl;
    }

    Flag("Post-initialization finished");
}
