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
#include "harm_driver.hpp"
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
    // This driver *requires* the "Implicit" package to be loaded, in order to read some flags
    // it defines for 

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
    bool use_emhd = pkgs.count("EMHD");

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
        auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
        auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
        auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
        auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);
        auto &mc_solver = pmesh->mesh_data.GetOrAdd("solver", i);

        auto t_start_recv = tl.AddTask(t_none, &MeshData<Real>::StartReceiving, mc1.get(),
                                    BoundaryCommSubset::all);

        // Calculate the HLL fluxes in each direction
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U)
        const ReconstructionType& recon = pkgs.at("GRMHD")->Param<ReconstructionType>("recon");
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
        // TODO should probably keep track of/wait on all tasks!! Might be a race condition!!
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

        auto t_flux_ct = t_fix_flux;
        if (use_b_flux_ct) {
            // Fix the conserved fluxes (exclusively B1/2/3) so that they obey divB==0,
            // and there is no B field flux through the pole
            auto t_flux_ct = tl.AddTask(t_fix_flux, B_FluxCT::TransportB, mc0.get());
        }
        auto t_flux_fixed = t_flux_ct;

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
        auto t_emhd_source = t_wind_source;
        if (use_emhd) {
            t_emhd_source = tl.AddTask(t_wind_source, EMHD::AddSource, mc0.get(), mdudt.get());
        }
        // Done with source terms
        auto t_sources = t_emhd_source;

        // UPDATE VARIABLES
        // This block is designed to intelligently update a set of variables partially marked "Implicit"
        // and partially "Explicit," by first doing any explicit updates, then using them as elements
        // of the "guess" for the implicit solve

        // Indicators for Explicit/Implicit variables to evolve
        MetadataFlag isExplicit = pkgs.at("Implicit")->Param<MetadataFlag>("ExplicitFlag");
        MetadataFlag isImplicit = pkgs.at("Implicit")->Param<MetadataFlag>("ImplicitFlag");
        MetadataFlag isPrimitive = pkgs.at("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
        // Substep timestep
        const double beta_this = integrator->beta[stage % integrator->nstages];
        const double dt_this = dt * beta_this;

        // Update any variables for which we should take an explicit step.
        // These calls are the equivalent of what's in HARMDriver
        // auto t_average = tl.AddTask(t_sources, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
        //                             std::vector<MetadataFlag>({isExplicit, Metadata::Independent}),
        //                             mc0.get(), mbase.get(), beta, (1.0 - beta), mc_solver.get());
        // auto t_explicit_U = tl.AddTask(t_average, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
        //                             std::vector<MetadataFlag>({isExplicit, Metadata::Independent}),
        //                             mc_solver.get(), mdudt.get(), 1.0, beta * dt, mc_solver.get());
        // Version with half/whole step to match implicit solver
        auto t_explicit_U = tl.AddTask(t_sources, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
                                    std::vector<MetadataFlag>({isExplicit, Metadata::Independent}),
                                    mbase.get(), mdudt.get(), 1.0, dt_this, mc_solver.get());

        // Make sure the primitive values of any explicit fields are filled
        auto t_explicit_UtoP_B = t_explicit_U;
        if (!pkgs.at("B_FluxCT")->Param<bool>("implicit"))
            t_explicit_UtoP_B = tl.AddTask(t_explicit_U, B_FluxCT::FillDerivedMeshTask, mc_solver.get());
        // If GRMHD is not implicit, but we're still going to be taking an implicit step, call its FillDerived function
        // TODO Would be faster/more flexible if this supported MeshData. Also maybe race condition
        auto t_explicit_UtoP_G = t_explicit_UtoP_B;
        if (!pkgs.at("GRMHD")->Param<bool>("implicit") && use_b_cd) {
            // Get flux corrections from AMR neighbors
            for (auto &pmb : pmesh->block_list) {
                auto& rc = pmb->meshblock_data.Get();
                auto t_explicit_UtoP_G = tl.AddTask(t_explicit_UtoP_B, GRMHD::FillDerivedBlockTask, rc.get());
            }
        }
        auto t_explicit = t_explicit_UtoP_G;

        // Copy the current implicit vars in as a guess.  This needs at least the primitive vars
        auto t_copy_guess = tl.AddTask(t_sources, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
                                    std::vector<MetadataFlag>({isImplicit}),
                                    mc0.get(), mc0.get(), 1.0, 0.0, mc_solver.get());

        // Time-step implicit variables by root-finding the residual
        // This applies the functions of both the update above and FillDerived call below for "isImplicit" variables
        // This takes dt for the *substep*, not the whole thing, so we multiply total dt by *this step's* beta
        auto t_guess_ready = t_explicit | t_copy_guess;
        auto t_implicit = tl.AddTask(t_guess_ready, Implicit::Step, mbase.get(), mc0.get(), mdudt.get(), mc_solver.get(), dt_this);

        // Copy the solver state into the final state mc1
        auto t_copy_result = tl.AddTask(t_implicit, Update::WeightedSumData<MetadataFlag, MeshData<Real>>, std::vector<MetadataFlag>({}),
                                        mc_solver.get(), mc_solver.get(), 1.0, 0.0, mc1.get());

        // If evolving GRMHD explicitly, U_to_P needs a guess in order to converge, so we copy in mc0
        auto t_copy_prims = t_none;
        if (!pkgs.at("GRMHD")->Param<bool>("implicit")) {
            MetadataFlag isPrimitive = pkgs.at("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
            MetadataFlag isHD = pkgs.at("GRMHD")->Param<MetadataFlag>("HDFlag");
            auto t_copy_prims = tl.AddTask(t_none, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
                                        std::vector<MetadataFlag>({isHD, isPrimitive}),
                                        mc0.get(), mc0.get(), 1.0, 0.0, mc1.get());
        }

    }

    // Even though we filled some primitive vars 
    TaskRegion &async_region1 = tc.AddRegion(blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region1[i];
        auto &sc0 = pmb->meshblock_data.Get(stage_name[stage-1]);
        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

        // Note that floors are applied (to all variables!) immediately after this FillDerived call.
        // However, it is *not* immediately corrected with FixUtoP, but synchronized (including pflags!) first.
        // With an extra ghost zone, this *should* still allow binary-similar evolution between numbers of mesh blocks,
        // but hasn't been tested.
        auto t_fill_derived = tl.AddTask(t_none, Update::FillDerived<MeshBlockData<Real>>, sc1.get());
    }

    // MPI/MeshBlock boundary exchange.
    // Optionally "packed" to send all data in one call (num_partitions defaults to 1)
    // Note that in this driver, this block syncs *primitive* variables, not conserved
    const auto &pack_comms = pkgs.at("GRMHD")->Param<bool>("pack_comms");
    AddBoundarySync(tc, pmesh, blocks, integrator.get(), stage, pack_comms);

    // Async Region: Any post-sync tasks.  Fixups, timestep & AMR things.
    TaskRegion &async_region2 = tc.AddRegion(blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region2[i];
        auto &sc0 = pmb->meshblock_data.Get(stage_name[stage-1]);
        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

        auto t_clear_comm_flags = tl.AddTask(t_none, &MeshBlockData<Real>::ClearBoundary,
                                        sc1.get(), BoundaryCommSubset::all);

        auto t_prolongBound = t_clear_comm_flags;
        if (pmesh->multilevel) {
            t_prolongBound = tl.AddTask(t_clear_comm_flags, ProlongateBoundaries, sc1);
        }

        auto t_set_bc = tl.AddTask(t_prolongBound, parthenon::ApplyBoundaryConditions, sc1);

        // If we're evolving even the GRMHD variables explicitly, we need to fix UtoP variable inversion failures
        // Syncing bounds before calling this, and then running it over the whole domain, will make
        // behavior for different mesh breakdowns much more similar (identical?), since bad zones in
        // relevant ghost zone ranks will get to use all the same neighbors as if they were in the bulk
        auto t_fix_derived = t_set_bc;
        if (!pkgs.at("GRMHD")->Param<bool>("implicit")) {
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

    // Second boundary sync:
    // ensure that primitive variables in ghost zones are *exactly*
    // identical to their physical counterparts, now that they have been
    // modified on each rank.
    const auto &two_sync = pkgs.at("GRMHD")->Param<bool>("two_sync");
    if (two_sync) {
        TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            auto &tl = single_tasklist_per_pack_region[i];
            auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);

            auto t_start_recv = tl.AddTask(t_none, &MeshData<Real>::StartReceiving, mc1.get(),
                                        BoundaryCommSubset::all);
        }

        AddBoundarySync(tc, pmesh, blocks, integrator.get(), stage, pack_comms);

        TaskRegion &async_region = tc.AddRegion(blocks.size());
        for (int i = 0; i < blocks.size(); i++) {
            auto &pmb = blocks[i];
            auto &tl = async_region[i];
            auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

            auto t_clear_comm_flags = tl.AddTask(t_none, &MeshBlockData<Real>::ClearBoundary,
                                            sc1.get(), BoundaryCommSubset::all);

            auto t_prolongBound = t_clear_comm_flags;
            if (pmesh->multilevel) {
                t_prolongBound = tl.AddTask(t_clear_comm_flags, ProlongateBoundaries, sc1);
            }
        }
    }

    return tc;
}
