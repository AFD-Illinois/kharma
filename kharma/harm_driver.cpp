/* 
 *  File: harm.cpp
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
#include "harm_driver.hpp"

#include <iostream>

#include <parthenon/parthenon.hpp>
#include <interface/update.hpp>
#include <refinement/refinement.hpp>

#include "decs.hpp"

#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "bondi.hpp"
#include "boundaries.hpp"
#include "debug.hpp"
#include "fixup.hpp"
#include "fluxes.hpp"
#include "grmhd.hpp"
#include "iharm_restart.hpp"
#include "source.hpp"

TaskCollection HARMDriver::MakeTaskCollection(BlockList_t &blocks, int stage)
{
    // Reminder that NOTHING YOU CALL HERE WILL GET CALLED EVERY STEP
    // this function is run *once*, and returns a list of what should be done every step.

    // TaskCollections are split into regions, each of which can be tackled by a specified number of independent threads.
    // We inherit our split, like most things in this function, from the advection_example in Parthenon
    using namespace Update;
    TaskCollection tc;
    TaskID t_none(0);

    const Real beta = integrator->beta[stage - 1];
    const Real dt = integrator->dt;
    auto stage_name = integrator->stage_name;

    const bool use_b_cd = blocks[0]->packages.AllPackages().count("B_CD");
    const bool use_b_flux_ct = blocks[0]->packages.AllPackages().count("B_FluxCT");

    auto num_task_lists_executed_independently = blocks.size();
    TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

    // Async Region I: calculate (and in current version, apply) fluxes
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region1[i];
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

        // pull out the container we'll use to get fluxes and/or compute RHSs
        auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
        // pull out a container we'll use to store dU/dt.
        // This is just -flux_divergence in this example
        auto &dudt = pmb->meshblock_data.Get("dUdt");
        // pull out the container that will hold the updated state
        // effectively, sc1 = sc0 + dudt*dt
        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

        auto t_start_recv = tl.AddTask(t_none, &MeshBlockData<Real>::StartReceiving, sc1.get(),
                                    BoundaryCommSubset::all);

        // Calculate the HLL fluxes in each direction
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U)
        // All subsequent operations until FillDerived are applied only to U
        // TODO these could be made mesh-wide pretty easily...
        const ReconstructionType& recon = pmb->packages.Get("GRMHD")->Param<ReconstructionType>("recon");
        TaskID t_calculate_flux1, t_calculate_flux2, t_calculate_flux3;
        switch (recon) {
        case ReconstructionType::donor_cell:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::donor_cell, X1DIR>, sc0.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::donor_cell, X2DIR>, sc0.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::donor_cell, X3DIR>, sc0.get());
            break;
        case ReconstructionType::linear_mc:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_mc, X1DIR>, sc0.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_mc, X2DIR>, sc0.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_mc, X3DIR>, sc0.get());
            break;
        case ReconstructionType::linear_vl:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_vl, X1DIR>, sc0.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_vl, X2DIR>, sc0.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::linear_vl, X3DIR>, sc0.get());
            break;
        case ReconstructionType::weno5:
            t_calculate_flux1 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::weno5, X1DIR>, sc0.get());
            t_calculate_flux2 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::weno5, X2DIR>, sc0.get());
            t_calculate_flux3 = tl.AddTask(t_start_recv, Flux::GetFlux<ReconstructionType::weno5, X3DIR>, sc0.get());
            break;
        case ReconstructionType::ppm:
        case ReconstructionType::mp5:
        case ReconstructionType::weno5_lower_poles:
            cerr << "Reconstruction type not supported!  Supported reconstructions:" << endl;
            cerr << "donor_cell, linear_mc, linear_vl, weno5" << endl;
            exit(-5);
        }
        auto t_calculate_flux = t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;

        auto t_recv_flux = t_calculate_flux;
        if (pmesh->multilevel) {
            // Get flux corrections from AMR neighbors
            auto t_send_flux =
                tl.AddTask(t_calculate_flux, &MeshBlockData<Real>::SendFluxCorrection, sc0.get());
            t_recv_flux =
                tl.AddTask(t_calculate_flux, &MeshBlockData<Real>::ReceiveFluxCorrection, sc0.get());
        }

        // Zero any fluxes through the pole or inflow from outflow boundaries
        //auto t_fix_flux = tl.AddTask(t_recv_flux, KBoundaries::FixFlux, sc0.get());
        auto t_fix_flux = t_recv_flux;

        auto t_flux_fixed = t_fix_flux;
        if (use_b_flux_ct) {
            // Fix the conserved fluxes (exclusively B1/2/3) so that they obey divB==0,
            // and there is no B field flux through the pole
            auto t_flux_ct = tl.AddTask(t_fix_flux, B_FluxCT::TransportB, sc0.get());
            t_flux_fixed = t_flux_ct;
        }

        // TODO move all these below
        auto t_flux_apply = t_flux_fixed;
        const auto &combine_flux_source =
            blocks[0]->packages.Get("GRMHD")->Param<bool>("combine_flux_source");
        if (combine_flux_source) {
            t_flux_apply = tl.AddTask(t_flux_fixed, Flux::ApplyFluxes, sc0.get(), dudt.get());
        } else {
            auto t_flux_div = tl.AddTask(t_flux_fixed, FluxDivergence<MeshBlockData<Real>>, sc0.get(), dudt.get());
            t_flux_apply = tl.AddTask(t_flux_div, GRMHD::AddSource, sc0.get(), dudt.get());
        }
        //auto t_flux_apply = tl.AddTask(t_flux_fixed, Flux::ApplyFluxes, sc0.get(), dudt.get());

        // Source term for constraint-damping.  Could be applied mesh-wide in that block,
        // when we learn to write mesh-wide functions
        auto t_b_cd_source = t_flux_apply;
        if (use_b_cd) {
            t_b_cd_source = tl.AddTask(t_flux_apply, B_CD::AddSource, sc0.get(), dudt.get());
        }
    }

    // Sync Region: add effects of current step to accumulator, sync boundaries
    const int num_partitions = pmesh->DefaultNumPartitions();
    // note that task within this region that contains one tasklist per pack
    // could still be executed in parallel
    TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = single_tasklist_per_pack_region[i];
        auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
        auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
        auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
        auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);

        // When we implement ApplyFluxes over full mesh
        //auto t_flux_apply = tl.AddTask(t_none, GRMHD::ApplyFluxes, tm, mc0.get(), mdudt.get());

        auto t_avg_data = tl.AddTask(t_none, AverageIndependentData<MeshData<Real>>,
                                mc0.get(), mbase.get(), beta);
        // apply du/dt to all independent fields in the container
        auto t_update = tl.AddTask(t_avg_data, UpdateIndependentData<MeshData<Real>>, mc0.get(),
                                mdudt.get(), beta * dt, mc1.get());
    }

    // Boundary exchange.  Optionally "packed" to send all data in one call.
    // All 3 calls are for MPI/meshblock boundaries
    const auto &buffer_send_pack =
        blocks[0]->packages.Get("GRMHD")->Param<bool>("buffer_send_pack");
    if (buffer_send_pack) {
        TaskRegion &tr = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
            tr[i].AddTask(t_none, cell_centered_bvars::SendBoundaryBuffers, mc1);
        }
    } else {
        TaskRegion &tr = tc.AddRegion(num_task_lists_executed_independently);
        for (int i = 0; i < blocks.size(); i++) {
            auto &sc1 = blocks[i]->meshblock_data.Get(stage_name[stage]);
            tr[i].AddTask(t_none, &MeshBlockData<Real>::SendBoundaryBuffers, sc1.get());
        }
    }
    const auto &buffer_recv_pack =
        blocks[0]->packages.Get("GRMHD")->Param<bool>("buffer_recv_pack");
    if (buffer_recv_pack) {
        TaskRegion &tr = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
            tr[i].AddTask(t_none, cell_centered_bvars::ReceiveBoundaryBuffers, mc1);
        }
    } else {
        TaskRegion &tr = tc.AddRegion(num_task_lists_executed_independently);
        for (int i = 0; i < blocks.size(); i++) {
            auto &sc1 = blocks[i]->meshblock_data.Get(stage_name[stage]);
            tr[i].AddTask(t_none, &MeshBlockData<Real>::ReceiveBoundaryBuffers, sc1.get());
        }
    }
    const auto &buffer_set_pack =
        blocks[0]->packages.Get("GRMHD")->Param<bool>("buffer_set_pack");
    if (buffer_set_pack) {
        TaskRegion &tr = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
            tr[i].AddTask(t_none, cell_centered_bvars::SetBoundaries, mc1);
        }
    } else {
        TaskRegion &tr = tc.AddRegion(num_task_lists_executed_independently);
        for (int i = 0; i < blocks.size(); i++) {
            auto &sc1 = blocks[i]->meshblock_data.Get(stage_name[stage]);
            tr[i].AddTask(t_none, &MeshBlockData<Real>::SetBoundaries, sc1.get());
        }
    }

    // Async Region II: Fill primitive values, apply physical boundary conditions
    TaskRegion &async_region2 = tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region2[i];
        auto &base = pmb->meshblock_data.Get();
        auto &sc0 = pmb->meshblock_data.Get(stage_name[stage-1]);
        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

        auto t_clear_comm_flags = tl.AddTask(t_none, &MeshBlockData<Real>::ClearBoundary,
                                        sc1.get(), BoundaryCommSubset::all);

        auto t_prolongBound = t_clear_comm_flags;
        if (pmesh->multilevel) {
            t_prolongBound = tl.AddTask(t_clear_comm_flags, ProlongateBoundaries, sc1);
        }
        // At this point, we've sync'd all internal boundaries using the conserved
        // variables. The physical boundaries (pole, inner/outer) are trickier,
        // since they must be applied to the primitive variables rho,u,u1,u2,u3
        // but should apply to conserved forms of everything else.

        // U_to_P needs a guess in order to converge, so we copy in sc0
        // (but only the fluid primitives!)
        // TODO option to declare primitives OneCopy if we won't need the backup
        // TODO do this mesh-at-once
        auto t_copy_prims = tl.AddTask(t_prolongBound,
            [](MeshBlockData<Real> *rc0, MeshBlockData<Real> *rc1)
            {
                FLAG("Copying prims");
                rc1->Get("prims.rho").data.DeepCopy(rc0->Get("prims.rho").data);
                rc1->Get("prims.u").data.DeepCopy(rc0->Get("prims.u").data);
                rc1->Get("prims.uvec").data.DeepCopy(rc0->Get("prims.uvec").data);
                FLAG("Copied");
                return TaskStatus::complete;
            }, sc0.get(), sc1.get());

        // This will fill the fluid primitive values in all zones except physical (outflow, reflecting) boundaries,
        // i.e. where the conserved variables are consistent.
        // we then fill those boundaries based on the primitives, and calculate conserved variables
        auto t_fill_derived = tl.AddTask(t_copy_prims, Update::FillDerived<MeshBlockData<Real>>, sc1.get());

        // This is a parthenon call, but in spherical coordinates this will call the functions in
        // boundaries.cpp, which apply to the primitive variables for GRMHD, and conserved forms for everything
        // else.  That means calling FillDerived again on any sections it's replaced!
        auto t_set_bc = tl.AddTask(t_fill_derived, ApplyBoundaryConditions, sc1);
        auto t_step_done = t_set_bc;

        // Estimate next time step based on ctop
        if (stage == integrator->nstages) {
            auto t_new_dt =
                tl.AddTask(t_step_done, EstimateTimestep<MeshBlockData<Real>>, sc1.get());

            // Update refinement
            if (pmesh->adaptive) {
                auto tag_refine = tl.AddTask(
                    t_step_done, parthenon::Refinement::Tag<MeshBlockData<Real>>, sc1.get());
            }
        }
    }

    return tc;
}
