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
//#include "multizone.hpp"
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
    // pmesh->DefaultNumPartitions()
    bool is_active[pmesh->block_list.size()] = {0, 0, 1};
    bool apply_boundary_condition[pmesh->block_list.size()][BOUNDARY_NFACES];

    //Multizone::DecideActiveBlocks(pmesh, is_active, apply_boundary_condition);

    // TaskCollections are a collection of TaskRegions.
    // Each TaskRegion can operate on eash meshblock separately, i.e. one MeshBlockData object (slower),
    // or on a collection of MeshBlock objects called the MeshData
    TaskCollection tc;
    const TaskID t_none(0);
    const int num_partitions = pmesh->DefaultNumPartitions();
    
    Flag("MakeTaskCollection::timestep");

    // Timestep region: calculate timestep based on the newly updated active zones
    TaskRegion &timestep_region = tc.AddRegion(num_partitions);
    // Estimate next time step based on ctop
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = timestep_region[i];
        if (is_active[i]) {
            auto &base = pmesh->mesh_data.GetOrAdd("base", i);
            auto t_new_dt =
                tl.AddTask(t_none, Update::EstimateTimestep<MeshData<Real>>, base.get());
        }
    }
    SetGlobalTimeStep();
    
    EndFlag();

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
        //pmesh->mesh_data.Add(integrator->stage_name[stage]);
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
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_flux_src       = pmesh->mesh_data.GetOrAdd("dUdt", i);
        auto t_update = t_none;

        std::cerr << pmesh->block_list.size() << " " << pmesh->DefaultNumPartitions() << " " << i << std::endl;

        if (is_active[i]) {
            // Start receiving flux corrections and ghost cells
            // auto t_start_recv_bound = tl.AddTask(t_none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, md_sync);
            auto t_start_recv_flux = t_none; //t_start_recv_bound;
            if (pmesh->multilevel || use_b_ct)
                t_start_recv_flux = tl.AddTask(t_none, parthenon::StartReceiveFluxCorrections, md_sub_step_init);

            // Calculate the flux of each variable through each face
            // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
            // of the conserved variables (U) through each face.
            auto t_flux_calc = KHARMADriver::AddFluxCalculations(t_start_recv_flux, tl, md_sub_step_init.get());
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

            // If we're in AMR, correct fluxes from neighbors
            auto t_flux_bounds = t_fix_flux;
            if (pmesh->multilevel || use_b_ct) {
                auto t_emf = t_flux_bounds;
                if (use_b_ct) {
                    // Pull out a container of only EMF to synchronize
                    auto &md_emf_only = pmesh->mesh_data.AddShallow("EMF", std::vector<std::string>{"B_CT.emf"}); // TODO this gets weird if we partition
                    auto t_emf_local = tl.AddTask(t_flux_bounds, B_CT::CalculateEMF, md_sub_step_init.get());
                    t_emf = KHARMADriver::AddBoundarySync(t_emf_local, tl, md_emf_only);
                    //Multizone::AverageEMFSeams(md_emf_only, apply_boundary_condition);
                }
                auto t_load_send_flux = tl.AddTask(t_emf, parthenon::LoadAndSendFluxCorrections, md_sub_step_init);
                auto t_recv_flux = tl.AddTask(t_load_send_flux, parthenon::ReceiveFluxCorrections, md_sub_step_init);
                t_flux_bounds = tl.AddTask(t_recv_flux, parthenon::SetFluxCorrections, md_sub_step_init);
            }

            // Apply the fluxes to calculate a change in cell-centered values "md_flux_src"
            auto t_flux_div = tl.AddTask(t_flux_bounds, FluxDivergence, md_sub_step_init.get(), md_flux_src.get(),
                                        std::vector<MetadataFlag>{Metadata::Independent, Metadata::Cell, Metadata::WithFluxes}, 0);

            // Add any source terms: geometric \Gamma * T, wind, damping, etc etc
            // Also where CT sets the change in face fields
            auto t_sources = tl.AddTask(t_flux_div, Packages::AddSource, md_sub_step_init.get(), md_flux_src.get(), IndexDomain::interior);

            t_update = KHARMADriver::AddStateUpdate(t_sources, tl, md_full_step_init.get(), md_sub_step_init.get(),
                                                md_flux_src.get(), md_sub_step_final.get(),
                                                std::vector<MetadataFlag>{Metadata::GetUserFlag("Explicit"), Metadata::Independent},
                                                use_b_ct, stage);
        } else {
            std::cerr << "HERE" << std::endl;
            Copy<MeshData<Real>>({Metadata::Cell}, md_full_step_init.get(), md_sub_step_final.get());
        }
    }

    for (int i = 0; i < 1; i++) {
        auto &md_sub_step_final = pmesh->mesh_data.Add(integrator->stage_name[stage]);
        auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage], md_sub_step_final, sync_vars);
        KHARMADriver::AddFullSyncRegion(tc, md_sync);
    }

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

        auto t_prim_source = t_set_bc;
        if (stage == integrator->nstages) {
            t_prim_source = tl.AddTask(t_set_bc, Packages::MeshApplyPrimSource, md_sub_step_final.get());
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
        auto& grmhd_pkg = pkgs.at("GRMHD")->AllParams();
        if (use_b_ct && grmhd_pkg.Get<bool>("ismr_poles")) {
            uint nlevels = grmhd_pkg.Get<uint>("ismr_nlevels");
            if (nlevels > 0) {
                auto t_derefine_poles = tl.AddTask(t_ptou, B_CT::DerefinePoles, md_sub_step_final.get(), nlevels);
                t_step_done = tl.AddTask(t_derefine_poles, Packages::MeshUtoP, md_sub_step_final.get(), IndexDomain::entire, false);
            } else {
                printf("WARNING: internal SMR near the poles is requested, but the number of levels should be >= 1. Not operating internal SMR.\n");
            }
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
        for (int i = 0; i < 1; i++) {
            auto &md_sub_step_final = pmesh->mesh_data.Add(integrator->stage_name[stage]);
            auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage], md_sub_step_final, sync_vars);
            KHARMADriver::AddFullSyncRegion(tc, md_sync);
        }
    }

    EndFlag();

    return tc;
}
