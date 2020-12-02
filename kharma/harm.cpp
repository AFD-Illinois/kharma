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

#include <iostream>

#include <parthenon/parthenon.hpp>
#include <interface/update.hpp>

#include "decs.hpp"

#include "bondi.hpp"
#include "boundaries.hpp"
#include "fixup.hpp"
#include "fluxes.hpp"
#include "grmhd.hpp"
#include "harm.hpp"

#include "iharm_restart.hpp"

/**
 * Custom block update stolen from advection_driver, again.
 * Really not sure why this isn't just a part of Parthenon...
 */
TaskStatus UpdateMeshBlockData(const int stage, Integrator *integrator,
                          std::shared_ptr<parthenon::MeshBlockData<Real>> &in,
                          std::shared_ptr<parthenon::MeshBlockData<Real>> &base,
                          std::shared_ptr<parthenon::MeshBlockData<Real>> &dudt,
                          std::shared_ptr<parthenon::MeshBlockData<Real>> &out) {
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  parthenon::Update::AverageIndependentData(in, base, beta);
  parthenon::Update::UpdateIndependentData(in, dudt, beta * dt, out);
  return TaskStatus::complete;
}

TaskCollection HARMDriver::MakeTaskCollection(BlockList_t &blocks, int stage)
{
    // TODO extra Refinement flux steps in whatever form they look like when I get around to SMR :)
    // Then same with updating refinement when maybe eventually AMR
    // TODO Figure out when in here to grab the beginning & end states & calculate jcon,
    // then only do so on output steps

    // TaskCollections are split into regions, each of which can be tackled by a specified number of independent threads.
    // TODO describe final split
    TaskCollection tc;
    TaskRegion &update_cons = tc.AddRegion(blocks.size());
    TaskID t_none(0);

    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = update_cons[i];

        // Parthenon separates out stages of higher-order integrators with "containers," er, MeshBlockData objects
        // (they're just a bundle of NDarrays capable of holding the Fields in a FluidState)
        // We use container per stage, filled and used over the course of the step to eventually update the base container
        // An accumulator dUdt is provided to temporarily store this stage's contribution to the RHS
        auto& base = pmb->meshblock_data.Get();
        if (stage == 1) {
            pmb->meshblock_data.Add("dUdt", base);
            for (int i=1; i < integrator->nstages; i++)
                pmb->meshblock_data.Add(stage_name[i], base);
        }

        // pull out the container we'll use to get fluxes and/or compute RHSs
        auto& sc0  = pmb->meshblock_data.Get(stage_name[stage-1]);
        // pull out a container we'll use to store dU/dt.
        auto& dudt = pmb->meshblock_data.Get("dUdt");
        // pull out the container that will hold the updated state
        auto& sc1  = pmb->meshblock_data.Get(stage_name[stage]);

        auto t_start_recv = tl.AddTask(t_none, &MeshBlockData<Real>::StartReceiving, sc1.get(),
                                    BoundaryCommSubset::all);

        // Calculate the HLL fluxes in each direction
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U)
        // All subsequent operations until FillDerived are applied only to U
        auto t_calculate_flux1 = tl.AddTask(t_start_recv, HLLE::GetFlux, sc0, X1DIR);
        auto t_calculate_flux2 = tl.AddTask(t_start_recv, HLLE::GetFlux, sc0, X2DIR);
        auto t_calculate_flux3 = tl.AddTask(t_start_recv, HLLE::GetFlux, sc0, X3DIR);
        auto t_calculate_flux = t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;

        // These operate only on the conserved fluxes
        //auto t_fix_flux = tl.AddTask(t_calculate_flux, FixFlux, sc0);
        auto t_flux_ct = tl.AddTask(t_calculate_flux, GRMHD::FluxCT, sc0);

        // Apply the corrected fluxes to create a single update dU/dt
        //auto t_flux_divergence = tl.AddTask(t_flux_ct, Update::FluxDivergence<MeshBlockData<Real>>, sc0, dudt);
        // Add the source term.  NOTE THIS USES P!  But U hasn't been touched yet, just fluxes
        //auto t_source_term = tl.AddTask(t_flux_divergence, GRMHD::AddSourceTerm, sc0, dudt);
        //auto t_flux_apply = t_source_term;

        // Alternative to above: combine the above and customize! TODO option?
        auto t_flux_apply = tl.AddTask(t_flux_ct, GRMHD::ApplyFluxes, sc0, dudt);

        // Apply dU/dt to the stage's initial state sc0 to obtain the stage final state sc1
        auto t_update_container = tl.AddTask(t_flux_apply, UpdateMeshBlockData, stage, integrator, sc0, base, dudt, sc1);

        // Update ghost cells.  Only performed on U of sc1
        auto t_send =
            tl.AddTask(t_update_container, &MeshBlockData<Real>::SendBoundaryBuffers, sc1.get());
        auto t_recv = tl.AddTask(t_send, &MeshBlockData<Real>::ReceiveBoundaryBuffers, sc1.get());
        auto t_fill_from_bufs = tl.AddTask(t_recv, &MeshBlockData<Real>::SetBoundaries, sc1.get());
        auto t_clear_comm_flags = tl.AddTask(t_fill_from_bufs, &MeshBlockData<Real>::ClearBoundary, sc1.get(),
                                            BoundaryCommSubset::all);

        // Set physical boundaries
        // Only respects/updates U
        //auto t_set_parthenon_bc = tl.AddTask(t_fill_from_bufs, parthenon::ApplyBoundaryConditions, sc1);

        // Fill primitives, bringing U and P back into lockstep
        auto t_fill_derived = tl.AddTask(t_clear_comm_flags, Update::FillDerived<MeshBlockData<Real>>, sc1);

        // ApplyCustomBoundaries is a catch-all for things HARM needs done:
        // Inflow checks, renormalizations, Bondi outer boundary.  All keep lockstep.
        auto t_set_custom_bc = tl.AddTask(t_fill_derived, ApplyCustomBoundaries, sc1);
        auto t_step_done = t_set_custom_bc;

        if (stage == integrator->nstages) {
            // estimate next time step
            auto new_dt = tl.AddTask(t_step_done, Update::EstimateTimestep<MeshBlockData<Real>>, sc1);
        }
    }
    return tc;
}