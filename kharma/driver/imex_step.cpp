/* 
 *  File: imex_step.cpp
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
#include "kharma_driver.hpp"

#include "decs.hpp"

//Packages
#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "b_cleanup.hpp"
#include "electrons.hpp"
#include "grmhd.hpp"
#include "wind.hpp"
// Other headers
#include "boundaries.hpp"
#include "debug.hpp"
#include "flux.hpp"
#include "resize_restart.hpp"
#include "implicit.hpp"

#include <parthenon/parthenon.hpp>
#include <interface/update.hpp>
#include <amr_criteria/refinement_package.hpp>

TaskCollection KHARMADriver::MakeImExTaskCollection(BlockList_t &blocks, int stage)
{
    Flag("Generating default task collection");
    // Reminder that this list is created BEFORE any of the list contents are run!
    // Prints or function calls here will likely not do what you want: instead, add to the list by calling tl.AddTask()

    TaskCollection tc;
    TaskID t_none(0);

    // Which packages we've loaded affects which tasks we'll add to the list
    auto& pkgs         = blocks[0]->packages.AllPackages();
    auto& driver_pkg   = pkgs.at("Driver")->AllParams();
    const bool use_electrons = pkgs.count("Electrons");
    const bool use_b_cleanup = pkgs.count("B_Cleanup");
    const bool use_implicit = pkgs.count("Implicit");
    const bool use_jcon = pkgs.count("Current");
    const bool use_linesearch = (use_implicit) ? pkgs.at("Implicit")->Param<bool>("linesearch") : false;

    // Allocate the fluid states ("containers") we need for each block
    for (auto& pmb : blocks) {
        // first make other useful containers
        auto &base = pmb->meshblock_data.Get();
        if (stage == 1) {
            pmb->meshblock_data.Add("dUdt", base);
            for (int i = 1; i < integrator->nstages; i++)
                pmb->meshblock_data.Add(integrator->stage_name[i], base);
            
            if (use_jcon) {
                // At the end of the step, updating "mbd_sub_step_final" updates the base
                // So we have to keep a copy at the beginning to calculate jcon
                pmb->meshblock_data.Add("preserve", base);
            }

            if (use_implicit) {
                // When solving, we need a temporary copy with any explicit updates,
                // but not overwriting the beginning- or mid-step values
                pmb->meshblock_data.Add("solver", base);
                if (use_linesearch) {
                    // Need an additional state for linesearch
                    pmb->meshblock_data.Add("linesearch", base);
                }
            }
        }
    }

    //auto t_heating_test = tl.AddTask(t_none, Electrons::ApplyHeating, base.get());

    // Big synchronous region: get & apply fluxes to advance the fluid state
    // num_partitions is nearly always 1
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = single_tasklist_per_pack_region[i];
        // Container names: 
        // '_full_step_init' refers to the fluid state at the start of the full time step (Si in iharm3d)
        // '_sub_step_init' refers to the fluid state at the start of the sub step (Ss in iharm3d)
        // '_sub_step_final' refers to the fluid state at the end of the sub step (Sf in iharm3d)
        // '_flux_src' refers to the mesh object corresponding to -divF + S
        // '_solver' refers to the fluid state passed to the Implicit solver. At the end of the solve
        // '_linesearch' refers to the fluid state updated while performing a linesearch in the solver
        // copy P and U from solver state to sub_step_final state.
        auto &md_full_step_init = pmesh->mesh_data.GetOrAdd("base", i);
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage - 1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_flux_src       = pmesh->mesh_data.GetOrAdd("dUdt", i);
        // Normally we put explicit update in md_solver, then add implicitly-evolved variables and copy back.
        // If we're not doing an implicit solve at all, just write straight to sub_step_final
        std::shared_ptr<MeshData<Real>> &md_solver = (use_implicit) ? pmesh->mesh_data.GetOrAdd("solver", i) : md_sub_step_final;

        // Start receiving flux corrections and ghost cells
        namespace cb = parthenon::cell_centered_bvars;
        auto t_start_recv_bound = tl.AddTask(t_none, cb::StartReceiveBoundBufs<parthenon::BoundaryType::any>, md_sub_step_final);
        auto t_start_recv_flux = t_start_recv_bound;
        if (pmesh->multilevel)
            t_start_recv_flux = tl.AddTask(t_none, cb::StartReceiveFluxCorrections, md_sub_step_init);
        
        // Calculate the flux of each variable through each face
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U) through each face.
        const KReconstruction::Type& recon = driver_pkg.Get<KReconstruction::Type>("recon");
        auto t_fluxes = KHARMADriver::AddFluxCalculations(t_start_recv_bound, tl, recon, md_sub_step_init.get());

        // If we're in AMR, correct fluxes from neighbors
        auto t_flux_bounds = t_fluxes;
        if (pmesh->multilevel) {
            tl.AddTask(t_fluxes, cb::LoadAndSendFluxCorrections, md_sub_step_init);
            auto t_recv_flux = tl.AddTask(t_fluxes, cb::ReceiveFluxCorrections, md_sub_step_init);
            t_flux_bounds = tl.AddTask(t_recv_flux, cb::SetFluxCorrections, md_sub_step_init);
        }

        // Any package modifications to the fluxes.  e.g.:
        // 1. CT calculations for B field transport
        // 2. Zero fluxes through poles
        // etc 
        auto t_fix_flux = tl.AddTask(t_flux_bounds, Packages::FixFlux, md_sub_step_init.get());

        // Apply the fluxes to calculate a change in cell-centered values "md_flux_src"
        auto t_flux_div = tl.AddTask(t_fix_flux, Update::FluxDivergence<MeshData<Real>>, md_sub_step_init.get(), md_flux_src.get());

        // Add any source terms: geometric \Gamma * T, wind, damping, etc etc
        auto t_sources = tl.AddTask(t_flux_div, Packages::AddSource, md_sub_step_init.get(), md_flux_src.get());

        // UPDATE VARIABLES
        // This block is designed to intelligently update a set of variables partially marked "Implicit"
        // and partially "Explicit," by first doing any explicit updates, then using them as elements
        // of the "guess" for the implicit solve

        // Update the explicitly-evolved variables using the source term
        // Add any proportion of the step start required by the integrator (e.g., RK2)
        auto t_avg_data = tl.AddTask(t_sources, Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
                                    std::vector<MetadataFlag>({Metadata::GetUserFlag("Explicit"), Metadata::Independent}),
                                    md_sub_step_init.get(), md_full_step_init.get(),
                                    integrator->gam0[stage-1], integrator->gam1[stage-1],
                                    md_solver.get());
        // apply du/dt to the result
        auto t_update = tl.AddTask(t_sources, Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
                                    std::vector<MetadataFlag>({Metadata::GetUserFlag("Explicit"), Metadata::Independent}),
                                    md_solver.get(), md_flux_src.get(),
                                    1.0, integrator->beta[stage-1] * integrator->dt,
                                    md_solver.get());

        // If evolving GRMHD explicitly, UtoP needs a guess in order to converge, so we copy in md_sub_step_init
        auto t_copy_prims = t_none;
        if (!pkgs.at("GRMHD")->Param<bool>("implicit")) {
            t_copy_prims        = tl.AddTask(t_none, Copy, std::vector<MetadataFlag>({Metadata::GetUserFlag("HD"), Metadata::GetUserFlag("Primitive")}),
                                             md_sub_step_init.get(), md_solver.get());
        }

        // Make sure the primitive values of *explicitly-evolved* variables are updated.
        // Each package should have a guard which makes UtoP a no-op if it's implicitly evolved
        auto t_explicit_UtoP = tl.AddTask(t_copy_prims, Packages::MeshUtoP, md_solver.get(), IndexDomain::interior, false);

        // Done with explicit update
        auto t_explicit = t_explicit_UtoP;

        auto t_implicit = t_explicit;
        if (use_implicit) {
            // Extra containers for implicit solve
            std::shared_ptr<MeshData<Real>> &md_linesearch = (use_linesearch) ? pmesh->mesh_data.GetOrAdd("linesearch", i) : md_solver;

            // Copy the current state of any implicitly-evolved vars (at least the prims) in as a guess.
            // This sets md_solver = md_sub_step_init
            auto t_copy_guess = tl.AddTask(t_sources, Copy, std::vector<MetadataFlag>({Metadata::GetUserFlag("Implicit")}),
                                        md_sub_step_init.get(), md_solver.get());

            auto t_guess_ready = t_explicit | t_copy_guess;

            // The `solver` MeshData object now has the implicit primitives corresponding to initial/half step and
            // explicit variables have been updated to match the current step.
            // Copy the primitives to the `linesearch` MeshData object if linesearch was enabled.
            auto t_copy_linesearch = t_guess_ready;
            if (use_linesearch) {
                t_copy_linesearch = tl.AddTask(t_guess_ready, Copy, std::vector<MetadataFlag>({Metadata::GetUserFlag("Primitive")}),
                                                md_solver.get(), md_linesearch.get());
            }


            // Time-step implicit variables by root-finding the residual.
            // This calculates the primitive values after the substep for all "isImplicit" variables --
            // no need for separately adding the flux divergence or calling UtoP
            auto t_implicit_step = tl.AddTask(t_copy_linesearch, Implicit::Step, md_full_step_init.get(), md_sub_step_init.get(), 
                                         md_flux_src.get(), md_linesearch.get(), md_solver.get(), integrator->beta[stage-1] * integrator->dt);

            // Copy the entire solver state (everything defined on the grid, i.e. 'Cell') into the final state md_sub_step_final
            // If we're entirely explicit, we just declare these equal
            t_implicit = tl.AddTask(t_implicit_step, Copy, std::vector<MetadataFlag>({Metadata::Cell}),
                                    md_solver.get(), md_sub_step_final.get());

        }

        // Apply all floors & limits (GRMHD,EMHD,etc), but do *not* immediately correct UtoP failures with FixUtoP --
        // rather, we will synchronize (including pflags!) first.
        // With an extra ghost zone, this *should* still allow binary-similar evolution between numbers of mesh blocks,
        // but hasn't been tested to do so yet.
        auto t_floors = tl.AddTask(t_implicit, Packages::MeshApplyFloors, md_sub_step_final.get(), IndexDomain::interior);

        KHARMADriver::AddMPIBoundarySync(t_floors, tl, md_sub_step_final);
    }

    // Async Region: Any post-sync tasks.  Fixups, timestep & AMR tagging.
    TaskRegion &async_region2 = tc.AddRegion(blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl  = async_region2[i];
        auto &mbd_sub_step_init  = pmb->meshblock_data.Get(integrator->stage_name[stage-1]);
        auto &mbd_sub_step_final = pmb->meshblock_data.Get(integrator->stage_name[stage]);

        // If we're evolving the GRMHD variables explicitly, we need to fix UtoP variable inversion failures
        // Syncing bounds before calling this, and then running it over the whole domain, will make
        // behavior for different mesh breakdowns much more similar (identical?), since bad zones in
        // relevant ghost zone ranks will get to use all the same neighbors as if they were in the bulk
        auto t_fix_p = tl.AddTask(t_none, Inverter::FixUtoP, mbd_sub_step_final.get());

        auto t_set_bc = tl.AddTask(t_fix_p, parthenon::ApplyBoundaryConditions, mbd_sub_step_final);

        // Any package- (likely, problem-) specific source terms which must be applied to primitive variables
        // Apply these only after the final step so they're operator-split
        auto t_prim_source = t_set_bc;
        if (stage == integrator->nstages) {
            t_prim_source = tl.AddTask(t_set_bc, Packages::BlockApplyPrimSource, mbd_sub_step_final.get());
        }
        // Electron heating goes where it does in the KHARMA Driver, for the same reasons
        auto t_heat_electrons = t_prim_source;
        if (use_electrons) {
            t_heat_electrons = tl.AddTask(t_prim_source, Electrons::ApplyElectronHeating,
                                          mbd_sub_step_init.get(), mbd_sub_step_final.get());
        }

        // Make sure *all* conserved vars are synchronized at step end
        auto t_ptou = tl.AddTask(t_heat_electrons, Flux::BlockPtoU, mbd_sub_step_final.get(), IndexDomain::entire, false);

        auto t_step_done = t_ptou;

        // Estimate next time step based on ctop
        if (stage == integrator->nstages) {
            auto t_new_dt =
                tl.AddTask(t_step_done, Update::EstimateTimestep<MeshBlockData<Real>>, mbd_sub_step_final.get());

            // Update refinement
            if (pmesh->adaptive) {
                auto tag_refine = tl.AddTask(
                    t_step_done, parthenon::Refinement::Tag<MeshBlockData<Real>>, mbd_sub_step_final.get());
            }
        }
    }

    // Second boundary sync:
    // ensure that primitive variables in ghost zones are *exactly*
    // identical to their physical counterparts, now that they have been
    // modified on each rank.
    const auto &two_sync = pkgs.at("Driver")->Param<bool>("two_sync");
    if (two_sync) KHARMADriver::AddFullSyncRegion(pmesh, tc, stage);


    return tc;
}

