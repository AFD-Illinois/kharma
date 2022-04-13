/* 
 *  File: harm_driver.hpp
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
#pragma once

#include <memory>

#include <parthenon/parthenon.hpp>

#include "types.hpp"

using namespace parthenon;

/**
 * A Driver object orchestrates everything that has to be done to a mesh to constitute a step.
 * For HARM, this means the predictor-corrector steps of fluid evolution.
 * 
 * Unlike MHD, GRMHD has two independent sets of variables: the conserved variables, and a set of
 * "primitive" variables more amenable to reconstruction.  To evolve the fluid, the conserved
 * variables must be:
 * 1. Transformed to the primitives
 * 2. Reconstruct the right- and left-going components at zone faces
 * 3. Transform back to conserved quantities and calculate the fluxes at faces
 * 4. Update conserved variables using the divergence of conserved fluxes
 * 
 * (for higher-order schemes, this is more or less just repeated and added)
 *
 * iharm3d (and the ImEx driver) put step 1 at the bottom, and syncs/fixes primitive variables
 * between each step.  This driver runs through the steps as listed, applying floors after step
 * 1 as iharm3d does, but syncing the conserved variables.
 */
class HARMDriver : public MultiStageDriver {
    public:
        /**
         * Default constructor
         */
        HARMDriver(ParameterInput *pin, ApplicationInput *papp, Mesh *pm) : MultiStageDriver(pin, papp, pm) {}

        /**
         * All the tasks which constitute advancing the fluid in a mesh by one stage.
         * This includes calculation of the primitives and reconstruction of their face values,
         * calculation of conserved values and fluxes thereof at faces,
         * application of fluxes and a source term in order to update zone values,
         * and finally calculation of the next timestep based on the CFL condition.
         * 
         * The function is heavily documented since order changes can introduce subtle bugs,
         * usually w.r.t. fluid "state" being spread across the primitive and conserved quantities
         */
        TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);

    private:
        // Global solves need a reduction point
        AllReduce<Real> update_norm;
};

/**
 * Add a boundary synchronization sequence to the TaskCollection tc.
 * 
 * This sequence is used identically in several places, so it makes sense
 * to define once and use elsewhere.
 * TODO could make member of a HARMDriver/ImExDriver superclass?
 */
inline void AddBoundarySync(TaskCollection &tc, Mesh *pmesh, BlockList_t &blocks, StagedIntegrator *integrator, int stage)
{
    TaskID t_none(0);
    const int num_partitions = pmesh->DefaultNumPartitions();
    auto stage_name = integrator->stage_name;
    TaskRegion &tr1 = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
        tr1[i].AddTask(t_none,
            [](MeshData<Real> *mc1){ Flag(mc1, "Parthenon Send Buffers"); return TaskStatus::complete; }
        , mc1.get());
        tr1[i].AddTask(t_none, cell_centered_bvars::SendBoundaryBuffers, mc1);
    }
    TaskRegion &tr2 = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
        tr2[i].AddTask(t_none,
            [](MeshData<Real> *mc1){ Flag(mc1, "Parthenon Recv Buffers"); return TaskStatus::complete; }
        , mc1.get());
        tr2[i].AddTask(t_none, cell_centered_bvars::ReceiveBoundaryBuffers, mc1);
    }
    TaskRegion &tr3 = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
        tr3[i].AddTask(t_none,
            [](MeshData<Real> *mc1){ Flag(mc1, "Parthenon Set Boundaries"); return TaskStatus::complete; }
        , mc1.get());
        tr3[i].AddTask(t_none, cell_centered_bvars::SetBoundaries, mc1);
    }
}
