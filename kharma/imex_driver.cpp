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
#include "fixup.hpp"
#include "flux.hpp"
#include "resize_restart.hpp"
#include "implicit.hpp"
#include "source.hpp"

TaskCollection ImexDriver::MakeTaskCollection(BlockList_t &blocks, int stage)
{
    // Reminder that NOTHING YOU CALL HERE WILL GET CALLED EVERY STEP
    // this function is run *once*, and returns a list of what should be done every step.
    // No prints or direct function calls here will do what you want, only calls to tl.AddTask()

    // This is *not* likely the task list you are looking for, and is not well commented yet.
    // See harm_driver.cpp for KHARMA's main driver.
    TaskCollection tc;
    TaskID t_none(0);

    Real beta = integrator->beta[stage - 1];
    const Real dt = integrator->dt;
    auto stage_name = integrator->stage_name;

    // Which packages we've loaded affects which tasks we'll add to the list
    auto& pkgs = blocks[0]->packages.AllPackages();
    bool use_b_cd = pkgs.count("B_CD");
    bool use_b_flux_ct = pkgs.count("B_FluxCT");
    bool use_electrons = pkgs.count("Electrons");
    bool use_wind = pkgs.count("Wind");

    // Allocate the fluid states ("containers") we need for each block
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        // first make other useful containers
        auto &base = pmb->meshblock_data.Get();
        if (stage == 1) {
            pmb->meshblock_data.Add("dUdt", base);
            for (int i = 1; i < integrator->nstages; i++)
                pmb->meshblock_data.Add(stage_name[i], base);
            // At the end of the step, updating "sc1" updates the base
            // So we have to keep a copy at the beginning to calculate jcon
            pmb->meshblock_data.Add("preserve", base);
        }
    }

    // Big synchronous region: get & apply fluxes to advance the fluid state
    // num_partitions is usually 1
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = single_tasklist_per_pack_region[i];
        auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
        auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
        auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
        auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);

        auto t_start_recv = tl.AddTask(t_none, &MeshData<Real>::StartReceiving, mc1.get(),
                                    BoundaryCommSubset::all);

        // Calculate the HLL fluxes in each direction
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U)
        const ReconstructionType& recon = blocks[0]->packages.Get("GRMHD")->Param<ReconstructionType>("recon");
        TaskID t_calculate_flux1, t_calculate_flux2, t_calculate_flux3;
        switch (recon) {
        case ReconstructionType::donor_cell:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::donor_cell, X1DIR>, mc0.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::donor_cell, X2DIR>, mc0.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::donor_cell, X3DIR>, mc0.get());
            break;
        case ReconstructionType::linear_mc:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_mc, X1DIR>, mc0.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_mc, X2DIR>, mc0.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_mc, X3DIR>, mc0.get());
            break;
        case ReconstructionType::linear_vl:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_vl, X1DIR>, mc0.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_vl, X2DIR>, mc0.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_vl, X3DIR>, mc0.get());
            break;
        case ReconstructionType::weno5:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::weno5, X1DIR>, mc0.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::weno5, X2DIR>, mc0.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::weno5, X3DIR>, mc0.get());
            break;
        case ReconstructionType::ppm:
        case ReconstructionType::mp5:
        case ReconstructionType::weno5_lower_poles:
            cerr << "Reconstruction type not supported!  Supported reconstructions:" << endl;
            cerr << "donor_cell, linear_mc, linear_vl, weno5" << endl;
            throw std::invalid_argument("Unsupported reconstruction algorithm!");
        }
        auto t_calculate_flux = t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;

        auto t_recv_flux = t_calculate_flux;
        // TODO this appears to be implemented *only* block-wise, split it into its own region if so
        if (pmesh->multilevel) {
            // Get flux corrections from AMR neighbors
            for (auto &pmb : pmesh->block_list) {
                auto& rc = pmb->meshblock_data.Get();
                auto t_send_flux =
                    tl.AddTask(t_calculate_flux, &MeshBlockData<Real>::SendFluxCorrection, rc.get());
                t_recv_flux =
                    tl.AddTask(t_calculate_flux, &MeshBlockData<Real>::ReceiveFluxCorrection, rc.get());
            }
        }

        // FIX FLUXES
        // Zero any fluxes through the pole or inflow from outflow boundaries
        auto t_fix_flux = tl.AddTask(t_recv_flux, KBoundaries::FixFlux, mc0.get());

        auto t_flux_fixed = t_fix_flux;
        if (use_b_flux_ct) {
            // Fix the conserved fluxes (exclusively B1/2/3) so that they obey divB==0,
            // and there is no B field flux through the pole
            auto t_flux_ct = tl.AddTask(t_fix_flux, B_FluxCT::TransportB, mc0.get());
            t_flux_fixed = t_flux_ct;
        }

        // APPLY FLUXES
        auto t_flux_div = tl.AddTask(t_none, Update::FluxDivergence<MeshData<Real>>, mc0.get(), mdudt.get());

        // ADD EXPLICIT SOURCES TO CONSERVED VARIABLES
        // Source term for GRMHD, \Gamma * T
        // TODO take this out in Minkowski space
        auto t_grmhd_source = tl.AddTask(t_flux_div, GRMHD::AddSource, mc0.get(), mdudt.get());
        // Source term for constraint-damping.  Applied only to B
        auto t_b_cd_source = t_grmhd_source;
        if (use_b_cd) {
            t_b_cd_source = tl.AddTask(t_grmhd_source, B_CD::AddSource, mc0.get(), mdudt.get());
        }
        // Wind source.  Applied to conserved variables similar to GR source term
        auto t_wind_source = t_b_cd_source;
        if (use_wind) {
            t_wind_source = tl.AddTask(t_b_cd_source, Wind::AddSource, mdudt.get());
        }
        // Done with source terms
        auto t_sources = t_wind_source;
    }

    // This region is where GRIM and classic HARM split.
    // Classic HARM applies the fluxes to calculate a new state of conserved variables,
    // then solves for the primitive variables with UtoP (here "FillDerived")
    const auto &driver_step =
        blocks[0]->packages.Get("GRMHD")->Param<std::string>("driver_step");
    if (driver_step == "explicit") { // Explicit step
        // Update conserved state with dUdt
        const int num_partitions = pmesh->DefaultNumPartitions();
        TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &tl = single_tasklist_per_pack_region[i];
            auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
            auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
            auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
            auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);

            // UPDATE BASE CONTAINER
            auto t_avg_data = tl.AddTask(t_none, Update::AverageIndependentData<MeshData<Real>>,
                                    mc0.get(), mbase.get(), beta);
            // apply du/dt to all independent fields in the container
            auto t_update = tl.AddTask(t_avg_data, Update::UpdateIndependentData<MeshData<Real>>, mc0.get(),
                                    mdudt.get(), beta * dt, mc1.get());
        }

        // Then solve for new primitives in the fluid interior, with the primitives at step start as a guess,
        // using UtoP.  Note that since no ghost zones are updated here, and thus FixUtoP cannot use
        // ghost zones. Thus KHARMA behavior in this mode will dependent on the breakdown of meshblocks,
        // & possibly erratic when there are many fixups.
        // Full algo should boundary sync -> FixUtoP -> boundary sync
        TaskRegion &async_region = tc.AddRegion(blocks.size());
        for (int i = 0; i < blocks.size(); i++) {
            auto &pmb = blocks[i];
            auto &tl = async_region[i];
            auto &sc0 = pmb->meshblock_data.Get(stage_name[stage-1]);
            auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

            // COPY PRIMITIVES
            // These form the guess for UtoP
            auto t_copy_prims = tl.AddTask(t_none,
                [](MeshBlockData<Real> *rc0, MeshBlockData<Real> *rc1)
                {
                    Flag(rc1, "Copying prims");
                    rc1->Get("prims.rho").data.DeepCopy(rc0->Get("prims.rho").data);
                    rc1->Get("prims.u").data.DeepCopy(rc0->Get("prims.u").data);
                    rc1->Get("prims.uvec").data.DeepCopy(rc0->Get("prims.uvec").data);
                    Flag(rc1, "Copied");
                    return TaskStatus::complete;
                }, sc0.get(), sc1.get()
            );

            auto t_fill_derived = tl.AddTask(t_copy_prims, Update::FillDerived<MeshBlockData<Real>>, sc1.get());
            // This is *not* immediately corrected with FixUtoP, but synchronized (including pflags!) first.
            // With an extra ghost zone, this *should* still allow binary-similar evolution between numbers of mesh blocks
        }
    } else { // Implicit step
        const int num_partitions = pmesh->DefaultNumPartitions();
        TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &tl = single_tasklist_per_pack_region[i];
            auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
            auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
            auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
            auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);

            // time-step by root-finding the residual
            // This applies the functions of both t_update and t_fill_derived
            auto t_implicit_solve = tl.AddTask(t_none, Implicit::Step, mbase.get(), mc0.get(), mdudt.get(), mc1.get(), dt);
        }
    }

    // MPI/MeshBlock boundary exchange.
    // Optionally "packed" to send all data in one call (num_partitions defaults to 1)
    // Note that in this driver, this block syncs *primitive* variables, not conserved
    const auto &pack_comms =
        blocks[0]->packages.Get("GRMHD")->Param<bool>("pack_comms");
    if (pack_comms) {
        TaskRegion &tr1 = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
            tr1[i].AddTask(t_none, cell_centered_bvars::SendBoundaryBuffers, mc1);
        }
        TaskRegion &tr2 = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
            tr2[i].AddTask(t_none, cell_centered_bvars::ReceiveBoundaryBuffers, mc1);
        }
        TaskRegion &tr3 = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
            tr3[i].AddTask(t_none, cell_centered_bvars::SetBoundaries, mc1);
        }
    } else {
        TaskRegion &tr1 = tc.AddRegion(blocks.size());
        for (int i = 0; i < blocks.size(); i++) {
            auto &sc1 = blocks[i]->meshblock_data.Get(stage_name[stage]);
            tr1[i].AddTask(t_none, &MeshBlockData<Real>::SendBoundaryBuffers, sc1.get());
        }
        TaskRegion &tr2 = tc.AddRegion(blocks.size());
        for (int i = 0; i < blocks.size(); i++) {
            auto &sc1 = blocks[i]->meshblock_data.Get(stage_name[stage]);
            tr2[i].AddTask(t_none, &MeshBlockData<Real>::ReceiveBoundaryBuffers, sc1.get());
        }
        TaskRegion &tr3 = tc.AddRegion(blocks.size());
        for (int i = 0; i < blocks.size(); i++) {
            auto &sc1 = blocks[i]->meshblock_data.Get(stage_name[stage]);
            tr3[i].AddTask(t_none, &MeshBlockData<Real>::SetBoundaries, sc1.get());
        }
    }

    // Async Region: Any post-sync tasks.  Timestep & AMR things.
    TaskRegion &async_region = tc.AddRegion(blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region[i];
        auto &sc0 = pmb->meshblock_data.Get(stage_name[stage-1]);
        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

        auto t_clear_comm_flags = tl.AddTask(t_none, &MeshBlockData<Real>::ClearBoundary,
                                        sc1.get(), BoundaryCommSubset::all);

        auto t_prolongBound = t_clear_comm_flags;
        if (pmesh->multilevel) {
            t_prolongBound = tl.AddTask(t_clear_comm_flags, ProlongateBoundaries, sc1);
        }

        auto t_set_bc = tl.AddTask(t_prolongBound, parthenon::ApplyBoundaryConditions, sc1);

        // Syncing bounds before fixUtoP, and thus running it over the whole domain, will make
        // behavior for different mesh breakdowns much more similar (identical?), as bad zones on boundaries
        // will get to use all the same neighbors.
        // As long as we sync pflags by setting FillGhosts when using this driver!
        auto t_fix_derived = t_set_bc;
        if (driver_step == "explicit") {
            t_fix_derived = tl.AddTask(t_set_bc, GRMHD::FixUtoP, sc1.get());
        }

        // Electron heating goes where it does in HARMDriver, for the same reasons
        auto t_heat_electrons = t_fix_derived;
        if (use_electrons) {
            t_heat_electrons = tl.AddTask(t_fix_derived, Electrons::ApplyElectronHeating, sc0.get(), sc1.get());
        }

        // Make sure conserved vars are synchronized at step end
        auto t_ptou = tl.AddTask(t_heat_electrons, Flux::PtoUTask, sc1.get());

        auto t_step_done = t_ptou;

        // Estimate next time step based on ctop
        if (stage == integrator->nstages) {
            auto t_new_dt =
                tl.AddTask(t_step_done, Update::EstimateTimestep<MeshBlockData<Real>>, sc1.get());

            // Update refinement
            if (pmesh->adaptive) {
                auto tag_refine = tl.AddTask(
                    t_step_done, parthenon::Refinement::Tag<MeshBlockData<Real>>, sc1.get());
            }
        }
    }

    return tc;
}
