/* 
 *  File: multizone_step.cpp
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

// TODO CLEAN
//Packages
#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "b_cleanup.hpp"
#include "b_ct.hpp"
#include "electrons.hpp"
#include "grmhd.hpp"
#include "inverter.hpp"
#include "ismr.hpp"
#include "multizone.hpp"
#include "wind.hpp"
// Other headers
#include "boundaries.hpp"
#include "flux.hpp"
#include "kharma.hpp"
#include "implicit.hpp"
#include "resize_restart.hpp"

#include <parthenon/parthenon.hpp>
#include <interface/update.hpp>
#include <amr_criteria/refinement_package.hpp>

TaskCollection KHARMADriver::MakeMultizoneTaskCollection(BlockList_t &blocks, int stage)
{
    // Reminder that this list is created BEFORE any of the list contents are run!
    // Prints or function calls here will likely not do what you want: instead, add to the list by calling tl.AddTask()
    const int num_partitions = pmesh->DefaultNumPartitions();
    const int num_blocks = pmesh->block_list.size();
    if (num_partitions != num_blocks)
        throw std::runtime_error("Multizone operation requires one block per MeshData!");

    // We know num_blocks == num_partitions, but I'll distinguish out of habit
    bool is_active[num_blocks]; // = {false, false, true};
    bool apply_boundary_condition[num_blocks][BOUNDARY_NFACES];

    if (stage == 1)
        Multizone::DecideToSwitch(pmesh, tm);
    Multizone::DecideActiveBlocksAndBoundaryConditions(pmesh, tm, is_active, apply_boundary_condition, stage == 1);

    // TaskCollections are a collection of TaskRegions.
    // Each TaskRegion can operate on eash meshblock separately, i.e. one MeshBlockData object (slower),
    // or on a collection of MeshBlock objects called the MeshData
    TaskCollection tc;
    const TaskID t_none(0);

    //Flag("MakeTaskCollection::timestep");

    // Timestep region: calculate timestep based on the newly updated active zones
    //TaskRegion &timestep_region = tc.AddRegion(num_partitions);
    // Estimate next time step based on ctop
    if (stage == 1) {
        for (int i = 0; i < num_partitions; i++) {
            std::cout << "iblock " << i << ": is active? " << is_active[i] << ", boundary applied? " << apply_boundary_condition[i][BoundaryFace::inner_x1] << apply_boundary_condition[i][BoundaryFace::outer_x1] << std::endl;
            auto &base = pmesh->mesh_data.GetOrAdd("base", i);
            //auto &tl = timestep_region[i];
            if (is_active[i]) {
                Update::EstimateTimestep<MeshData<Real>>(base.get());
              //auto t_new_dt =
              //    tl.AddTask(t_none, Update::EstimateTimestep<MeshData<Real>>, base.get());
            } else base->GetBlockData(0)->SetAllowedDt(std::numeric_limits<Real>::max()); // just ensuring that the inactive zones are not participating
        }
        SetGlobalTimeStep();
        std::cout << "HYERIN: actual dt is " << tm.dt << std::endl;
    }

    //EndFlag();

    // Which packages we load affects which tasks we'll add to the list
    auto& pkgs = pmesh->packages.AllPackages();
    auto& flux_pkg   = pkgs.at("Flux")->AllParams();
    const bool use_b_cleanup = pkgs.count("B_Cleanup");
    const bool use_b_ct = pkgs.count("B_CT");
    const bool use_electrons = pkgs.count("Electrons");
    const bool use_fofc = flux_pkg.Get<bool>("use_fofc");
    const bool use_jcon = pkgs.count("Current");

    // Allocate/copy the things we need
    // TODO these can now be reduced by including the var lists/flags which actually need to be allocated
    // TODO except the Copy they can be run on step 1 only
    if (stage == 1) {
        auto &base = pmesh->mesh_data.Get();
        // Fluxes
        pmesh->mesh_data.Add("dUdt");
        for (int i = 1; i < integrator->nstages; i++)
            pmesh->mesh_data.Add(integrator->stage_name[i]);
        // Preserve state for time derivatives if we need to output current
        if (use_jcon) {
            pmesh->mesh_data.Add("preserve");
            // Above only copies on allocate -- ensure we copy every step
            Copy<MeshData<Real>>({Metadata::Cell}, base.get(), pmesh->mesh_data.Get("preserve").get());
        }
        // FOFC needs to determine whether the "real" U-divF will violate floors, and needs a safe place to do it.
        // We populate it later, with each *sub-step*'s initial state
        if (use_fofc) {
            pmesh->mesh_data.Add("fofc_source");
            pmesh->mesh_data.Add("fofc_guess");
        }
    }

    Flag("MakeTaskCollection::fluxes");

    static std::vector<std::string> sync_vars;
    if (sync_vars.size() == 0) {
        // Build the universe of variables to let Parthenon see when exchanging boundaries.
        // This is built to exclude incidental variables like B field initialization stuff, EMFs, etc.
        // "Boundaries" packs in buffers e.g. Dirichlet boundaries
        using FC = Metadata::FlagCollection;
        auto sync_flags = FC({Metadata::GetUserFlag("Primitive"), Metadata::Conserved,
                              Metadata::Face, Metadata::GetUserFlag("Boundaries")}, true);
        sync_vars = KHARMA::GetVariableNames(&(pmesh->packages), sync_flags);
    }

    // Flux region: calculate and apply fluxes to update conserved values
    TaskRegion &flux_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = flux_region[i];
        auto &md_full_step_init = pmesh->mesh_data.GetOrAdd("base", i);
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage - 1], i);

        if (is_active[i]) {
            // Calculate the flux of each variable through each face
            // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
            // of the conserved variables (U) through each face.
            auto t_flux_calc = KHARMADriver::AddFluxCalculations(t_none, tl, md_sub_step_init.get());
            auto t_fluxes = t_flux_calc;
            if (use_fofc) {
                auto &guess_src = pmesh->mesh_data.GetOrAdd("fofc_source", i);
                auto &guess = pmesh->mesh_data.GetOrAdd("fofc_guess", i);
                auto t_fluxes = KHARMADriver::AddFOFC(t_flux_calc, tl, md_sub_step_init.get(), md_full_step_init.get(),
                                                    md_sub_step_init.get(), guess_src.get(), guess.get(), stage);
            }

            // Any package modifications to the fluxes.  e.g.:
            // 1. Flux-CT calculations for B field transport
            // 2. Zero fluxes through poles
            // etc
            auto t_fix_flux = tl.AddTask(t_fluxes, Packages::FixFlux, md_sub_step_init.get());

            // Calculate EMFs on active blocks
            if (use_b_ct) {
                 tl.AddTask(t_fix_flux, B_CT::CalculateEMF, md_sub_step_init.get());
            }
        }
    }

    // If we're in AMR or B_CT, sync EMFs and correct fluxes for ALL blocks
    if (pmesh->multilevel || use_b_ct) {
        TaskRegion &flux_sync_region = tc.AddRegion(1);
        auto &tl = flux_sync_region[0];
        auto &md_sub_step_init  = pmesh->mesh_data.Add(integrator->stage_name[stage - 1]);
        auto &md_emf_only = pmesh->mesh_data.AddShallow("EMF", md_sub_step_init, std::vector<std::string>{"B_CT.emf"});
        // Start receiving flux corrections and ghost cells
        // auto t_start_recv_bound = tl.AddTask(t_none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, md_sync);
        auto t_start_recv_flux = t_none;
        // t_start_recv_flux = tl.AddTask(t_none, parthenon::StartReceiveFluxCorrections, md_sub_step_init);
        // auto t_emf = t_start_recv_flux;
        if (use_b_ct) {
            auto t_emf_bounds = KHARMADriver::AddBoundarySync(t_start_recv_flux, tl, md_emf_only);
            auto t_emf = t_emf_bounds;
            // for (int i=0; i < num_blocks; i++) {
            //     auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage - 1], i);
            //     auto &md_emf_only = pmesh->mesh_data.AddShallow("EMF_"+std::to_string(i), md_sub_step_init, std::vector<std::string>{"B_CT.emf"});
            //     t_emf = tl.AddTask(t_emf, Multizone::AverageEMFSeams, md_emf_only.get(), apply_boundary_condition[i]);
            // }
        }
        // auto t_load_send_flux = tl.AddTask(t_emf, parthenon::LoadAndSendFluxCorrections, md_sub_step_init);
        // auto t_recv_flux = tl.AddTask(t_load_send_flux, parthenon::ReceiveFluxCorrections, md_sub_step_init);
        // tl.AddTask(t_recv_flux, parthenon::SetFluxCorrections, md_sub_step_init);
    }

    // Then continue/finish out the flux calculation
    TaskRegion &flux_postsync_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = flux_postsync_region[i];
        auto &md_full_step_init = pmesh->mesh_data.GetOrAdd("base", i);
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage - 1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_flux_src       = pmesh->mesh_data.GetOrAdd("dUdt", i);
        auto &md_emf_only = pmesh->mesh_data.AddShallow("EMF_"+std::to_string(i), md_sub_step_init, std::vector<std::string>{"B_CT.emf"});
        if (is_active[i]) {
            auto t_emf_seams = t_none;
            if (use_b_ct) {
                // Correct the EMFs of active zones
                t_emf_seams = tl.AddTask(t_none, Multizone::AverageEMFSeams, md_emf_only.get(), apply_boundary_condition[i]);
            }

            // Apply the fluxes to calculate a change in cell-centered values "md_flux_src"
            auto t_flux_div = tl.AddTask(t_emf_seams, FluxDivergence, md_sub_step_init.get(), md_flux_src.get(),
                                        std::vector<MetadataFlag>{Metadata::Independent, Metadata::Cell, Metadata::WithFluxes}, 0);

            // Add any source terms: geometric \Gamma * T, wind, damping, etc etc
            // Also where CT sets the change in face fields
            auto t_sources = tl.AddTask(t_flux_div, Packages::AddSource, md_sub_step_init.get(), md_flux_src.get(), IndexDomain::interior);

            KHARMADriver::AddStateUpdate(t_sources, tl, md_full_step_init.get(), md_sub_step_init.get(),
                                        md_flux_src.get(), md_sub_step_final.get(),
                                        std::vector<MetadataFlag>{Metadata::GetUserFlag("Explicit"), Metadata::Independent},
                                        use_b_ct, stage);
        } else {
            auto t_copy_cell = tl.AddTask(t_none, Copy<MeshData<Real>>, std::vector<MetadataFlag>{Metadata::Cell},
                                                  md_full_step_init.get(), md_sub_step_final.get());
            auto t_copy_face = tl.AddTask(t_none, CopyFace, std::vector<MetadataFlag>{Metadata::Face},
                                                  md_full_step_init.get(), md_sub_step_final.get());
        }
    }

    // Then a full-mesh sync
    auto &md_sub_step_final = pmesh->mesh_data.Add(integrator->stage_name[stage]);
    auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage], md_sub_step_final, sync_vars);
    KHARMADriver::AddFullSyncRegion(tc, md_sync);

    EndFlag();
    Flag("MakeTaskCollection::fixes");

    // Fix Region: prims/cons sync, floors, fixes, boundary conditions which need primitives
    TaskRegion &fix_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        if (!is_active[i])
            continue;

        auto &tl = fix_region[i];
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage-1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage]+std::to_string(i), md_sub_step_final, sync_vars);

        auto t_utop = tl.AddTask(t_none, Packages::MeshUtoP, md_sub_step_final.get(), IndexDomain::entire, false);

        auto t_floors = tl.AddTask(t_utop, Packages::MeshApplyFloors, md_sub_step_final.get(), IndexDomain::entire);

        auto t_fix_p = tl.AddTask(t_floors, Inverter::MeshFixUtoP, md_sub_step_final.get());

        auto t_set_bc = tl.AddTask(t_fix_p, parthenon::ApplyBoundaryConditionsOnCoarseOrFineMD, md_sync, false);

        auto t_prim_source = t_set_bc; //t_fix_p;
        if (stage == integrator->nstages) {
            t_prim_source = tl.AddTask(t_set_bc, Packages::MeshApplyPrimSource, md_sub_step_final.get()); //t_fix_p
        }
        // Electron heating goes where it does in HARMDriver, for the same reasons
        auto t_heat_electrons = t_prim_source;
        if (use_electrons) {
            t_heat_electrons = tl.AddTask(t_prim_source, Electrons::MeshApplyElectronHeating,
                                          md_sub_step_init.get(), md_sub_step_final.get(), stage == 1); // bool is generate_grf
        }

        // Make sure *all* conserved vars are synchronized at step end
        auto t_ptou = tl.AddTask(t_heat_electrons, Flux::MeshPtoU, md_sub_step_final.get(), IndexDomain::entire, false);

        auto t_step_done = t_ptou;
        if (pkgs.count("ISMR") && pkgs.at("ISMR")->Param<uint>("nlevels") > 0) {
            auto t_derefine_b = t_ptou;
            if (pkgs.count("B_CT"))
                t_derefine_b = tl.AddTask(t_ptou, B_CT::DerefinePoles, md_sub_step_final.get());
            auto t_derefine_f = tl.AddTask(t_derefine_b, ISMR::DerefinePoles, md_sub_step_final.get());
            auto t_floors_2 = tl.AddTask(t_derefine_f, Packages::MeshApplyFloors, md_sub_step_final.get(), IndexDomain::entire);
            t_step_done = tl.AddTask(t_floors_2, Inverter::MeshFixUtoP, md_sub_step_final.get());
        }
    }

    EndFlag();
    Flag("MakeTaskCollection::extras");

    // Second boundary sync:
    // ensure that primitive variables in ghost zones are *exactly*
    // identical to their physical counterparts, now that they have been
    // modified on each rank.
    const auto &two_sync = pkgs.at("Driver")->Param<bool>("two_sync");
    if (two_sync) {
        TaskRegion &bound_sync = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
            auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage]+std::to_string(i), md_sub_step_final, sync_vars);
            //KHARMADriver::AddFullSyncRegion(tc, md_sync);
            AddBoundarySync(t_none, bound_sync[i], md_sync);
        }
    }

    EndFlag();
    
    //Flag("MakeTaskCollection::timestep");
    //// HYERIN (06/20/24) splitting this part to the end for now
    //// Switch region: decide if we want to switch zones
    //TaskRegion &switch_region = tc.AddRegion(1);
    //auto &tl = switch_region[0];
    //auto &md_temp = pmesh->mesh_data.Add(integrator->stage_name[stage]);
    //bool switch_zone = false;
    //auto t_switch = t_none;
    //if (integrator->nstages == stage) {
    //    t_switch = tl.AddTask(t_none, Multizone::DecideToSwitch, md_temp.get(), tm, switch_zone);
    //}

    //// Timestep region: calculate timestep based on the newly updated active zones
    ////TaskRegion &timestep_region = tc.AddRegion(num_partitions);
    ////auto t_new_active = t_none;
    ////// Estimate next time step based on ctop
    ////for (int i = 0; i < num_partitions; i++) {
    ////    auto &tl = timestep_region[i];
    ////    //if (switch_zone) { // take a next step and re-evaluate the active block
    ////        t_new_active = tl.AddTask(t_none, Multizone::DecideNextActiveBlocks, md_temp.get(), tm, i, is_active[i], switch_zone);
    ////    //}
    ////    if (is_active[i]) {
    ////        auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    ////    //    Update::EstimateTimestep<MeshData<Real>>(base.get());
    ////        auto t_new_dt =
    ////            tl.AddTask(t_new_active, Update::EstimateTimestep<MeshData<Real>>, base.get());
    ////    }
    ////}
    

    //EndFlag();

    return tc;
}
