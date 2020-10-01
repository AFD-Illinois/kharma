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

#include "parthenon/parthenon.hpp"

#include "decs.hpp"

#include "bondi.hpp"
#include "boundaries.hpp"
#include "containers.hpp"
#include "fixup.hpp"
#include "grmhd.hpp"
#include "harm.hpp"

#include "iharm_restart.hpp"

TaskList HARMDriver::MakeTaskList(MeshBlock *pmb, int stage)
{
    TaskList tl;

    // Parthenon separates out stages of higher-order integrators with "containers"
    // (a bundle of arrays capable of holding all Fields in the FluidState)
    // One container per stage, filled and used over the course of the step to eventually update the base container
    // An accumulator dUdt is provided to temporarily store this stage's contribution to the RHS
    // TODO: Figure out when the step beginning and end are both accessible, for calculating jcon
    if (stage == 1) {
        auto& base = pmb->real_containers.Get();
        pmb->real_containers.Add("dUdt", base);
        for (int i=1; i<integrator->nstages; i++)
            pmb->real_containers.Add(stage_name[i], base);
    }

    // pull out the container we'll use to get fluxes and/or compute RHSs
    auto& sc0  = pmb->real_containers.Get(stage_name[stage-1]);
    // pull out a container we'll use to store dU/dt.
    auto& dudt = pmb->real_containers.Get("dUdt");
    // pull out the container that will hold the updated state
    auto& sc1  = pmb->real_containers.Get(stage_name[stage]);

    // TODO what does this do exactly?
    TaskID t_none(0);
    auto t_start_recv = tl.AddTask(t_none, &Container<Real>::StartReceiving, sc1.get(),
                                   BoundaryCommSubset::all);

    // Calculate the LLF fluxes in each direction
    // This uses the primitives (P) to calculate fluxes to update the conserved variables (U)
    // Hence the two should reflect *exactly* the same fluid state, hereafter "lockstep"
    // TODO actual concurrency?
    // Assumes valid P on entry, spits out valid U fluxes
    auto t_calculate_flux1 = tl.AddTask(t_start_recv, GRMHD::CalculateFlux, sc0, X1DIR);
    auto t_calculate_flux2 = tl.AddTask(t_start_recv, GRMHD::CalculateFlux, sc0, X2DIR);
    auto t_calculate_flux3 = tl.AddTask(t_start_recv, GRMHD::CalculateFlux, sc0, X3DIR);
    auto t_calculate_flux = t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;

    // TODO add these sensibly for AMR/SMR runs (below Fix and/or CT?)
    // auto t_send_flux =
    //     tl.AddTask(&Container<Real>::SendFluxCorrection, sc0.get(), t_calculate_flux);
    // auto t_recv_flux =
    //     tl.AddTask(&Container<Real>::ReceiveFluxCorrection, sc0.get(), t_calculate_flux);

    // These operate totally on fluxes
    auto t_fix_flux = tl.AddTask(t_calculate_flux, FixFlux, sc0);
    auto t_flux_ct = tl.AddTask(t_fix_flux, GRMHD::FluxCT, sc0);

    // Apply fluxes to create a single update dU/dt
    auto t_flux_divergence = tl.AddTask(t_flux_ct, Update::FluxDivergence, sc0, dudt);
    auto t_source_term = tl.AddTask(t_flux_divergence, GRMHD::AddSourceTerm, sc0, dudt);
    // Apply dU/dt to the stage's initial state sc0 to obtain the stage final state sc1
    // Note this *only fills U* of sc1, so sc1 is out of lockstep
    auto t_update_container = tl.AddTask(t_source_term, UpdateContainer, pmb, stage, stage_name, integrator);

    // Update ghost cells.  Only performed on U of sc1
    auto t_send =
        tl.AddTask(t_update_container, &Container<Real>::SendBoundaryBuffers, sc1.get());
    auto t_recv = tl.AddTask(t_send, &Container<Real>::ReceiveBoundaryBuffers, sc1.get());
    auto t_fill_from_bufs = tl.AddTask(t_recv, &Container<Real>::SetBoundaries, sc1.get());
    auto t_clear_comm_flags = tl.AddTask(t_fill_from_bufs, &Container<Real>::ClearBoundary, sc1.get(),
                                         BoundaryCommSubset::all);

    // TODO add sensibly for AMR runs
    // auto t_prolong_bound = tl.AddTask([](MeshBlock *pmb) {
    //     pmb->pbval->ProlongateBoundaries(0.0, 0.0);
    //     return TaskStatus::complete;
    // }, t_fill_from_bufs, pmb);

    // Set physical boundaries
    // ApplyCustomBoundaries is a catch-all for things HARM needs done:
    // Inflow checks, renormalizations, Bondi outer boundary
    // Only respects/updates U
    auto t_set_parthenon_bc = tl.AddTask(t_fill_from_bufs, parthenon::ApplyBoundaryConditions, sc1);

    // Fill primitives, bringing U and P back into lockstep
    auto t_fill_derived = tl.AddTask(t_set_parthenon_bc, parthenon::FillDerivedVariables::FillDerived, sc1);

    // Our boundaries require knowing P, so we separate them out.  They keep lockstep.
    auto t_set_custom_bc = tl.AddTask(t_fill_derived, ApplyCustomBoundaries, sc1);
    auto t_step_done = t_set_custom_bc;

    // estimate next time step
    if (stage == integrator->nstages) {
        auto new_dt = tl.AddTask(t_step_done,
            [](std::shared_ptr<Container<Real>> &rc) {
                auto pmb = rc->GetBlockPointer();
                pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
                return TaskStatus::complete;
            }, sc1);

        // Update refinement
        if (pmesh->adaptive) {
            auto tag_refine = tl.AddTask(t_step_done,
            [](MeshBlock *pmb) {
                pmb->pmr->CheckRefinementCondition();
                return TaskStatus::complete;
            }, pmb);
        }
    }
    return tl;
}