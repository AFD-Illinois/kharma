/* 
 *  File: imex_driver.cpp
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
#include "imex_driver.hpp"

#include <iostream>

#include <parthenon/parthenon.hpp>
#include <interface/update.hpp>
#include <refinement/refinement.hpp>

#include "decs.hpp"

//Packages
#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "electrons.hpp"
#include "grmhd.hpp"
#include "wind.hpp"
// Other headers
#include "boundaries.hpp"
#include "debug.hpp"
#include "flux.hpp"
#include "harm_driver.hpp"
#include "resize_restart.hpp"
#include "implicit.hpp"

TaskCollection ImexDriver::MakeTaskCollection(BlockList_t &blocks, int stage)
{
    // Reminder that NOTHING YOU CALL HERE WILL GET CALLED EVERY STEP
    // this function is run *once*, and returns a list of what should be done every step.
    // No prints or direct function calls here will do what you want, only calls to tl.AddTask()

    // This is *not* likely the task list you are looking for, and is not well commented yet.
    // See harm_driver.cpp for KHARMA's main driver.
    // This driver *requires* the "Implicit" package to be loaded, in order to read some flags
    // it defines for

    // NOTE: Renamed state names to something more intuitive. 
    // '_full_step_init' refers to the fluid state at the start of the full time step (Si in iharm3d)
    // '_sub_step_init' refers to the fluid state at the start of the sub step (Ss in iharm3d)
    // '_sub_step_final' refers to the fluid state at the end of the sub step (Sf in iharm3d)
    // '_flux_src' refers to the mesh object corresponding to -divF + S
    // '_solver' refers to the fluid state passed to the Implicit solver. At the end of the solve
    // copy P and U from solver state to sub_step_final state.

    TaskCollection tc;
    TaskID t_none(0);

    Real beta       = integrator->beta[stage - 1];
    const Real dt   = integrator->dt;
    auto stage_name = integrator->stage_name;

    // Which packages we've loaded affects which tasks we'll add to the list
    auto& pkgs         = blocks[0]->packages.AllPackages();
    bool use_b_cd      = pkgs.count("B_CD");
    bool use_b_flux_ct = pkgs.count("B_FluxCT");
    bool use_electrons = pkgs.count("Electrons");
    bool use_wind      = pkgs.count("Wind");
    bool use_emhd      = pkgs.count("EMHD");

    // Allocate the fluid states ("containers") we need for each block
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        // first make other useful containers
        auto &base = pmb->meshblock_data.Get();
        if (stage == 1) {
            pmb->meshblock_data.Add("dUdt", base);
            for (int i = 1; i < integrator->nstages; i++)
                pmb->meshblock_data.Add(stage_name[i], base);
            // At the end of the step, updating "mbd_sub_step_final" updates the base
            // So we have to keep a copy at the beginning to calculate jcon
            pmb->meshblock_data.Add("preserve", base);
            // When solving, we need a temporary copy with any explicit updates,
            // but not overwriting the beginning- or mid-step values
            pmb->meshblock_data.Add("solver", base);
        }
    }

    // Big synchronous region: get & apply fluxes to advance the fluid state
    // num_partitions is usually 1
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = single_tasklist_per_pack_region[i];
        auto &md_full_step_init = pmesh->mesh_data.GetOrAdd("base", i);
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
        auto &md_flux_src       = pmesh->mesh_data.GetOrAdd("dUdt", i);
        auto &md_solver         = pmesh->mesh_data.GetOrAdd("solver", i);

        auto t_start_recv_bound = tl.AddTask(t_none, parthenon::cell_centered_bvars::StartReceiveBoundBufs<parthenon::BoundaryType::any>, md_sub_step_final);
        auto t_start_recv_flux = t_none;
        if (pmesh->multilevel)
            t_start_recv_flux = tl.AddTask(t_none, parthenon::cell_centered_bvars::StartReceiveFluxCorrections, md_sub_step_init);
        auto t_start_recv = t_start_recv_bound | t_start_recv_flux;

        // Calculate the HLL fluxes in each direction
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U)
        const ReconstructionType& recon = pkgs.at("GRMHD")->Param<ReconstructionType>("recon");
        TaskID t_calculate_flux1, t_calculate_flux2, t_calculate_flux3;
        switch (recon) {
        case ReconstructionType::donor_cell:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::donor_cell, X1DIR>, md_sub_step_init.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::donor_cell, X2DIR>, md_sub_step_init.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::donor_cell, X3DIR>, md_sub_step_init.get());
            break;
        case ReconstructionType::linear_mc:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_mc, X1DIR>, md_sub_step_init.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_mc, X2DIR>, md_sub_step_init.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_mc, X3DIR>, md_sub_step_init.get());
            break;
        case ReconstructionType::linear_vl:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_vl, X1DIR>, md_sub_step_init.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_vl, X2DIR>, md_sub_step_init.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_vl, X3DIR>, md_sub_step_init.get());
            break;
        case ReconstructionType::weno5:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::weno5, X1DIR>, md_sub_step_init.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::weno5, X2DIR>, md_sub_step_init.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::weno5, X3DIR>, md_sub_step_init.get());
            break;
        case ReconstructionType::ppm:
        case ReconstructionType::mp5:
        case ReconstructionType::weno5_lower_poles:
            std::cerr << "Reconstruction type not supported!  Supported reconstructions:" << std::endl;
            std::cerr << "donor_cell, linear_mc, linear_vl, weno5" << std::endl;
            throw std::invalid_argument("Unsupported reconstruction algorithm!");
        }
        auto t_calculate_flux = t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;

        auto t_set_flux = t_calculate_flux;
        if (pmesh->multilevel) {
                tl.AddTask(t_calculate_flux, parthenon::cell_centered_bvars::LoadAndSendFluxCorrections, md_full_step_init);
                auto t_recv_flux = tl.AddTask(t_calculate_flux, parthenon::cell_centered_bvars::ReceiveFluxCorrections, md_full_step_init);
                t_set_flux = tl.AddTask(t_recv_flux, parthenon::cell_centered_bvars::SetFluxCorrections, md_full_step_init);
        }

        // FIX FLUXES
        // Zero any fluxes through the pole or inflow from outflow boundaries
        auto t_fix_flux = tl.AddTask(t_set_flux, KBoundaries::FixFlux, md_sub_step_init.get());

        auto t_flux_ct = t_fix_flux;
        if (use_b_flux_ct) {
            // Fix the conserved fluxes (exclusively B1/2/3) so that they obey divB==0,
            // and there is no B field flux through the pole
            auto t_flux_ct = tl.AddTask(t_fix_flux, B_FluxCT::TransportB, md_sub_step_init.get());
        }
        auto t_flux_fixed = t_flux_ct;

        // APPLY FLUXES
        auto t_flux_div = tl.AddTask(t_none, Update::FluxDivergence<MeshData<Real>>, md_sub_step_init.get(), md_flux_src.get());

        // ADD EXPLICIT SOURCES TO CONSERVED VARIABLES
        // Source term for GRMHD, \Gamma * T
        // TODO take this out in Minkowski space
        auto t_grmhd_source = tl.AddTask(t_flux_div, GRMHD::AddSource, md_sub_step_init.get(), md_flux_src.get());
        // Source term for constraint-damping.  Applied only to B
        auto t_b_cd_source = t_grmhd_source;
        if (use_b_cd) {
            t_b_cd_source = tl.AddTask(t_grmhd_source, B_CD::AddSource, md_sub_step_init.get(), md_flux_src.get());
        }
        // Wind source.  Applied to conserved variables similar to GR source term
        auto t_wind_source = t_b_cd_source;
        if (use_wind) {
            t_wind_source = tl.AddTask(t_b_cd_source, Wind::AddSource, md_flux_src.get());
        }
        auto t_emhd_source = t_wind_source;
        if (use_emhd) {
            t_emhd_source = tl.AddTask(t_wind_source, EMHD::AddSource, md_sub_step_init.get(), md_flux_src.get());
        }
        // Done with source terms
        auto t_sources = t_emhd_source;

        // UPDATE VARIABLES
        // This block is designed to intelligently update a set of variables partially marked "Implicit"
        // and partially "Explicit," by first doing any explicit updates, then using them as elements
        // of the "guess" for the implicit solve

        // Indicators for Explicit/Implicit variables to evolve
        MetadataFlag isExplicit  = pkgs.at("Implicit")->Param<MetadataFlag>("ExplicitFlag");
        MetadataFlag isImplicit  = pkgs.at("Implicit")->Param<MetadataFlag>("ImplicitFlag");
        MetadataFlag isPrimitive = pkgs.at("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
        // Substep timestep
        const double beta_this = integrator->beta[stage % integrator->nstages];
        const double dt_this = dt * beta_this;

        // Update any variables for which we should take an explicit step.
        // These calls are the equivalent of what's in HARMDriver
        // auto t_average = tl.AddTask(t_sources, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
        //                             std::vector<MetadataFlag>({isExplicit, Metadata::Independent}),
        //                             md_sub_step_init.get(), md_full_step_init.get(), beta, (1.0 - beta), md_solver.get());
        // auto t_explicit_U = tl.AddTask(t_average, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
        //                             std::vector<MetadataFlag>({isExplicit, Metadata::Independent}),
        //                             md_solver.get(), md_flux_src.get(), 1.0, beta * dt, md_solver.get());
        // Version with half/whole step to match implicit solver
        auto t_explicit_U = tl.AddTask(t_sources, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
                                    std::vector<MetadataFlag>({isExplicit, Metadata::Independent}),
                                    md_full_step_init.get(), md_flux_src.get(), 1.0, dt_this, md_solver.get());

        // Make sure the primitive values of any explicit fields are filled
        auto t_explicit_UtoP_B = t_explicit_U;
        if (!pkgs.at("B_FluxCT")->Param<bool>("implicit"))
            t_explicit_UtoP_B = tl.AddTask(t_explicit_U, B_FluxCT::FillDerivedMeshTask, md_solver.get());
        // If GRMHD is not implicit, but we're still going to be taking an implicit step, call its FillDerived function
        // TODO Would be faster/more flexible if this supported MeshData. Also maybe race condition
        auto t_explicit_UtoP_G = t_explicit_UtoP_B;
        if (!pkgs.at("GRMHD")->Param<bool>("implicit") && use_b_cd) {
            // Get flux corrections from AMR neighbors
            for (auto &pmb : pmesh->block_list) {
                auto& mbd = pmb->meshblock_data.Get();
                auto t_explicit_UtoP_G = tl.AddTask(t_explicit_UtoP_B, GRMHD::FillDerivedBlockTask, mbd.get());
            }
        }
        auto t_explicit = t_explicit_UtoP_G;

        // Copy the current implicit vars in as a guess.  This needs at least the primitive vars
        auto t_copy_guess = tl.AddTask(t_sources, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
                                    std::vector<MetadataFlag>({isImplicit}),
                                    md_sub_step_init.get(), md_sub_step_init.get(), 1.0, 0.0, md_solver.get());

        // Time-step implicit variables by root-finding the residual
        // This applies the functions of both the update above and FillDerived call below for "isImplicit" variables
        // This takes dt for the *substep*, not the whole thing, so we multiply total dt by *this step's* beta
        auto t_guess_ready = t_explicit | t_copy_guess;
        auto t_implicit = tl.AddTask(t_guess_ready, Implicit::Step, md_full_step_init.get(), md_sub_step_init.get(), 
                                    md_flux_src.get(), md_solver.get(), dt_this);

        // Copy the solver state into the final state md_sub_step_final
        auto t_copy_result = tl.AddTask(t_implicit, Update::WeightedSumData<MetadataFlag, MeshData<Real>>, 
                                        std::vector<MetadataFlag>({}), md_solver.get(), md_solver.get(), 
                                        1.0, 0.0, md_sub_step_final.get());

        // If evolving GRMHD explicitly, U_to_P needs a guess in order to converge, so we copy in md_sub_step_init
        auto t_copy_prims = t_none;
        if (!pkgs.at("GRMHD")->Param<bool>("implicit")) {
            MetadataFlag isPrimitive = pkgs.at("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
            MetadataFlag isHD        = pkgs.at("GRMHD")->Param<MetadataFlag>("HDFlag");
            auto t_copy_prims        = tl.AddTask(t_none, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
                                                std::vector<MetadataFlag>({isHD, isPrimitive}),
                                                md_sub_step_init.get(), md_sub_step_init.get(), 1.0, 0.0, md_sub_step_final.get());
        }

    }

    // Even though we filled some primitive vars 
    TaskRegion &async_region1 = tc.AddRegion(blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl  = async_region1[i];
        auto &mbd_sub_step_final = pmb->meshblock_data.Get(stage_name[stage]);

        // Note that floors are applied (to all variables!) immediately after this FillDerived call.
        // However, inversion/floor inversion failures are *not* immediately corrected with FixUtoP,
        // but synchronized (including pflags!) first.
        // With an extra ghost zone, this *should* still allow binary-similar evolution between numbers of mesh blocks,
        // but hasn't been tested to do so yet.
        auto t_fill_derived = tl.AddTask(t_none, Update::FillDerived<MeshBlockData<Real>>, mbd_sub_step_final.get());
    }

    TaskRegion &sync_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = sync_region[i];
        auto &mbd_sub_step_final = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
        // MPI/MeshBlock boundary exchange.
        // Note that in this driver, this block syncs *primitive* variables, not conserved
        KBoundaries::AddBoundarySync(t_none, tl, mbd_sub_step_final);
    }

    // Async Region: Any post-sync tasks.  Fixups, timestep & AMR things.
    TaskRegion &async_region2 = tc.AddRegion(blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl  = async_region2[i];
        auto &mbd_sub_step_init  = pmb->meshblock_data.Get(stage_name[stage-1]);
        auto &mbd_sub_step_final = pmb->meshblock_data.Get(stage_name[stage]);

        // If we're evolving even the GRMHD variables explicitly, we need to fix UtoP variable inversion failures
        // Syncing bounds before calling this, and then running it over the whole domain, will make
        // behavior for different mesh breakdowns much more similar (identical?), since bad zones in
        // relevant ghost zone ranks will get to use all the same neighbors as if they were in the bulk
        auto t_fix_derived = t_none;
        if (!pkgs.at("GRMHD")->Param<bool>("implicit")) {
            t_fix_derived = tl.AddTask(t_fix_derived, GRMHD::FixUtoP, mbd_sub_step_final.get());
        }

        auto t_set_bc = tl.AddTask(t_fix_derived, parthenon::ApplyBoundaryConditions, mbd_sub_step_final);

        // Electron heating goes where it does in HARMDriver, for the same reasons
        auto t_heat_electrons = t_set_bc;
        if (use_electrons) {
            t_heat_electrons = tl.AddTask(t_set_bc, Electrons::ApplyElectronHeating, 
                                        mbd_sub_step_init.get(), mbd_sub_step_final.get());
        }

        // Make sure conserved vars are synchronized at step end
        auto t_ptou = tl.AddTask(t_heat_electrons, Flux::PtoUTask, mbd_sub_step_final.get(), IndexDomain::entire);

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
    const auto &two_sync = pkgs.at("GRMHD")->Param<bool>("two_sync");
    if (two_sync) {
        TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &tl = single_tasklist_per_pack_region[i];
            auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);

            auto t_start_recv_bound = tl.AddTask(t_none, parthenon::cell_centered_bvars::StartReceiveBoundBufs<parthenon::BoundaryType::any>, md_sub_step_final);
            auto t_bound_sync = KBoundaries::AddBoundarySync(t_start_recv_bound, tl, md_sub_step_final);
        }
    }

    return tc;
}
