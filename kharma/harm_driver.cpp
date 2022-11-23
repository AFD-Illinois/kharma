/* 
 *  File: harm_driver.cpp
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
#include "electrons.hpp"
#include "grmhd.hpp"
#include "wind.hpp"

#include "boundaries.hpp"
#include "debug.hpp"
#include "flux.hpp"
#include "resize_restart.hpp"

TaskCollection HARMDriver::MakeTaskCollection(BlockList_t &blocks, int stage)
{
    // Reminder that NOTHING YOU CALL HERE WILL GET CALLED EVERY STEP
    // this function is run *once*, and returns a list of what should be done every step.
    // No prints or direct function calls here will do what you want, only calls to tl.AddTask()

    // TaskCollections are split into regions, each of which can be tackled by a specified number of independent threads.
    // We take most of the splitting logic here from the advection example in Parthenon,
    // except that we calculate the fluxes in a Mesh-wide section rather than for MeshBlocks independently
    TaskCollection tc;
    TaskID t_none(0);

    Real beta = integrator->beta[stage - 1];
    const Real dt = integrator->dt;
    auto stage_name = integrator->stage_name;

    // Which packages we load affects which tasks we'll add to the list
    auto& pkgs = blocks[0]->packages.AllPackages();
    bool use_b_cd = pkgs.count("B_CD");
    bool use_b_flux_ct = pkgs.count("B_FluxCT");
    bool use_electrons = pkgs.count("Electrons");
    bool use_wind = pkgs.count("Wind");

    // Allocate the fields ("containers") we need block by block
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

        auto t_start_recv_bound = tl.AddTask(t_none, parthenon::cell_centered_bvars::StartReceiveBoundBufs<parthenon::BoundaryType::any>, mc1);
        auto t_start_recv_flux = tl.AddTask(t_none, parthenon::cell_centered_bvars::StartReceiveFluxCorrections, mc0);
        auto t_start_recv = t_start_recv_bound | t_start_recv_flux;

        // Calculate the HLL fluxes in each direction
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U)
        // All subsequent operations until FillDerived are applied only to U
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
            std::cerr << "Reconstruction type not supported!  Supported reconstructions:" << std::endl;
            std::cerr << "donor_cell, linear_mc, linear_vl, weno5" << std::endl;
            throw std::invalid_argument("Unsupported reconstruction algorithm!");
        }
        auto t_calculate_flux = t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;

        auto t_set_flux = t_calculate_flux;
        if (pmesh->multilevel) {
                tl.AddTask(t_calculate_flux, parthenon::cell_centered_bvars::LoadAndSendFluxCorrections, mc0);
                auto t_recv_flux = tl.AddTask(t_calculate_flux, parthenon::cell_centered_bvars::ReceiveFluxCorrections, mc0);
                t_set_flux = tl.AddTask(t_recv_flux, parthenon::cell_centered_bvars::SetFluxCorrections, mc0);
        }

        // FIX FLUXES
        // Zero any fluxes through the pole or inflow from outflow boundaries
        auto t_fix_flux = tl.AddTask(t_set_flux, KBoundaries::FixFlux, mc0.get());

        auto t_flux_ct = t_fix_flux;
        if (use_b_flux_ct) {
            // Fix the conserved fluxes (exclusively B1/2/3) so that they obey divB==0,
            // and there is no B field flux through the pole
            t_flux_ct = tl.AddTask(t_fix_flux, B_FluxCT::TransportB, mc0.get());
        }
        auto t_flux_fixed = t_flux_ct;

        // APPLY FLUXES
        auto t_flux_div = tl.AddTask(t_flux_fixed, Update::FluxDivergence<MeshData<Real>>, mc0.get(), mdudt.get());

        // ADD SOURCES TO CONSERVED VARIABLES
        // Source term for GRMHD, \Gamma * T
        
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

        // UPDATE BASE CONTAINER
        auto t_avg_data = tl.AddTask(t_sources, Update::AverageIndependentData<MeshData<Real>>,
                                mc0.get(), mbase.get(), beta);
        // apply du/dt to all independent fields in the container
        auto t_update = tl.AddTask(t_avg_data, Update::UpdateIndependentData<MeshData<Real>>, mc0.get(),
                                mdudt.get(), beta * dt, mc1.get());

        // U_to_P needs a guess in order to converge, so we copy in sc0
        // (but only the fluid primitives!)  Copying and syncing ensures that solves of the same zone
        // on adjacent ranks are seeded with the same value, which keeps them (more) similar
        MetadataFlag isPrimitive = pkgs.at("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
        MetadataFlag isHD = pkgs.at("GRMHD")->Param<MetadataFlag>("HDFlag");
        auto t_copy_prims = tl.AddTask(t_none, Update::WeightedSumData<MetadataFlag, MeshData<Real>>,
                                    std::vector<MetadataFlag>({isHD, isPrimitive}),
                                    mc0.get(), mc0.get(), 1.0, 0.0, mc1.get());
        
        KBoundaries::AddBoundarySync(t_copy_prims, tl, mc1);
        // if (pmesh->multilevel) {
        //     auto t_restrict = tl.AddTask(t_bound_sync, parthenon::cell_centered_refinement::RestrictPhysicalBounds, mc1.get());
        //     tl.AddTask(t_restrict, ProlongateBoundaries, mc1);
        // }
    }

    // Async Region: Fill primitive values, apply physical boundary conditions,
    // add any source terms which require the full primitives->primitives step
    // TODO this can be Meshified
    TaskRegion &async_region = tc.AddRegion(blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region[i];
        //auto &base = pmb->meshblock_data.Get();
        auto &sc0 = pmb->meshblock_data.Get(stage_name[stage-1]);
        auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

        // At this point, we've sync'd all internal boundaries using the conserved
        // variables. The physical boundaries (pole, inner/outer) are trickier,
        // since they must be applied to the primitive variables rho,u,u1,u2,u3
        // but should apply to conserved forms of everything else.

        // This call fills the fluid primitive values in all physical zones, that is, including MPI boundaries but
        // not the physical boundaries (which haven't been filled yet!)
        // This relies on the primitives being calculated identically in MPI boundaries, vs their corresponding
        // physical zones in the adjacent mesh block.  To ensure this, we seed the solver with the same values
        // in each case, by synchronizing them along with the conserved values above.
        auto t_fill_derived = tl.AddTask(t_none, Update::FillDerived<MeshBlockData<Real>>, sc1.get());
        // After this call, the floors are applied (with the hook 'PostFillDerived', see floors.cpp)

        // Immediately fix any inversions which failed.  Floors have been applied already as a part of (Post)FillDerived,
        // so fixups performed by averaging zones will return logical results.  Floors are re-applied after fixups
        // Someday this will not be necessary as guaranteed-convergent UtoP schemes exist
        auto t_fix_derived = tl.AddTask(t_fill_derived, GRMHD::FixUtoP, sc1.get());

        // This is a parthenon call, but in spherical coordinates it will call the KHARMA functions in
        // boundaries.cpp, which apply physical boundary conditions based on the primitive variables of GRHD,
        // and based on the conserved forms for everything else.  Note that because this is called *after*
        // FillDerived (since it needs bulk fluid primitives to apply GRMHD boundaries), this function
        // must call FillDerived *again* (for everything except the GRHD variables) to fill P in the ghost zones.
        // This is why KHARMA packages need to implement their "FillDerived" a.k.a. UtoP functions in the form
        // UtoP(rc, domain, coarse): so that they can be run over just the boundary domains here.
        auto t_set_bc = tl.AddTask(t_fix_derived, parthenon::ApplyBoundaryConditions, sc1);

        // ADD SOURCES TO PRIMITIVE VARIABLES
        // In order to calculate dissipation, we must know the entropy at the beginning and end of the substep,
        // and this must be calculated from the fluid primitive variables rho,u (and for stability, obey floors!).
        // We only have these just now from FillDerived (and PostFillDerived, and the boundary consistency stuff)
        // Luckily, ApplyElectronHeating does *not* need another synchronization of the ghost zones, as it is applied to
        // all zones and has a stencil of only one zone.  As with FillDerived, this trusts that evaluations 
        // on the same zone match between MeshBlocks.
        auto t_heat_electrons = t_set_bc;
        if (use_electrons) {
            auto t_heat_electrons = tl.AddTask(t_set_bc, Electrons::ApplyElectronHeating, sc0.get(), sc1.get());
        }

        auto t_step_done = t_heat_electrons;

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

            auto t_start_recv_bound = tl.AddTask(t_none, parthenon::cell_centered_bvars::StartReceiveBoundBufs<parthenon::BoundaryType::any>, mc1);
            auto t_bound_sync = KBoundaries::AddBoundarySync(t_start_recv_bound, tl, mc1);
        }
    }

    return tc;
}
