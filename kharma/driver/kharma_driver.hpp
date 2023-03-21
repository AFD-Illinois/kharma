/* 
 *  File: kharma_driver.hpp
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

#include "decs.hpp"
#include "types.hpp"

#include "reconstruction.hpp"

using namespace parthenon;

/**
 * This is the "Driver" class for KHARMA.
 * A Driver object orchestrates everything that has to be done to a mesh to constitute a step.
 * This means handling RK2/4/predictor-corrector stepping
 * 
 * Somewhat confusingly, but very conveniently, it is also a package; therefore, it defines
 * a static member function Initialize(), which returns a StateDescriptor.
 * Many things in that list are referenced by other packages dependent on this one.
 * 
 */
class KHARMADriver : public MultiStageDriver {
    public:
        KHARMADriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm) : MultiStageDriver(pin, app_in, pm) {}

        static std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

        /**
         * A Driver object orchestrates everything that has to be done to a mesh to take a step.
         * The function MakeTaskCollection outlines everything to be done in one sub-step,
         * so that the driver can repeat calls to create a predictor-corrector, RK2/4, etc.
         * 
         * Unlike MHD, GRMHD must keep two forms of the variables: the conserved variables, and a set of
         * "primitive" variables more amenable to reconstruction.  To evolve the fluid, the code must:
         * 1. Reconstruct the right- and left-going components at zone faces, given the primitive variables
         * 2. Calculate the fluxes of conserved quantities through the faces
         * 2a. Apply any fixes to fluxes (e.g., for the magnetic field)
         * 3. Update conserved variables using their prior values the divergence of conserved fluxes
         * 3a. Apply any source terms (e.g., the geometric term in GRMHD)
         * 4. Recover primtive variables
         * 4a. Apply any stability limits (floors)
         * 4b. Fix any errors in recovering the primitives, re-apply floors
         * 5. Apply any source terms (KEL), or calculate outputs (jcon) which require the change in primitive values
         * 
         * This is before any synchronization between different blocks, etc, etc.
         * Both task lists proceed roughly in this order, and you'll see the same broad outlines in both.
         */
        TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
        TaskCollection MakeDefaultTaskCollection(BlockList_t &blocks, int stage);

        /**
         * This "TaskCollection" (step) 
         * ImexDriver syncs primitive variables and treats them as fundamental, whereas HARMDriver syncs conserved variables.
         * This allows ImexDriver to optionally use a semi-implicit step, adding a per-zone implicit solve via the 'Implicit'
         * package, instead of just explicit RK2 time-stepping.  This driver also allows explicit-only RK2 operation
         */
        TaskCollection MakeImExTaskCollection(BlockList_t &blocks, int stage);

        /**
         * A simple step for experimentation.  Does NOT support MPI, 
         */
        TaskCollection MakeSimpleTaskCollection(BlockList_t &blocks, int stage);


        static TaskID AddFluxCalculations(TaskID& t_start, TaskList& tl, KReconstruction::Type recon, MeshData<Real> *md);

        /**
         * Add just the synchronization step to a task list tl, dependent upon taskID t_start, syncing mesh mc1
         * 
         * This sequence is used identically in several places, so it makes sense
         * to define once and use elsewhere.
         */
        void AddFullSyncRegion(Mesh* pmesh, TaskCollection& tc, int stage);

        /**
         * Add just the synchronization step to a task list tl, dependent upon taskID t_start, syncing mesh mc1
         * 
         * This sequence is used identically in several places, so it makes sense
         * to define once and use elsewhere.
         */
        static TaskID AddMPIBoundarySync(TaskID t_start, TaskList &tl, std::shared_ptr<MeshData<Real>> mc1);

        /**
         * Calculate the fluxes in each direction
         */
        static TaskID AddFluxCalculation(TaskID& start, TaskList& tl, KReconstruction::Type recon, MeshData<Real> *md);

        /**
         * Single call to sync all boundary conditions (MPI/internal and domain/physical boundaries)
         * Used anytime boundary sync is needed outside the usual loop of steps.
         */
        static void SyncAllBounds(std::shared_ptr<MeshData<Real>> md, bool apply_domain_bounds=true);

        // TODO swapped versions of these
        /**
         * Copy variables matching 'flags' from 'source' to 'dest'.
         * Mostly makes things easier to read.
         */
        static TaskStatus Copy(std::vector<MetadataFlag> flags, MeshData<Real>* source, MeshData<Real>* dest)
        {
            return Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>(flags, source, source, 1., 0., dest);
        }

        /**
         * Scale a variable by 'norm'.
         * Mostly makes things easier to read.
         */
        static TaskStatus Scale(std::vector<std::string> flags,  MeshBlockData<Real>* source, Real norm)
        {
            return Update::WeightedSumData<std::vector<std::string>, MeshBlockData<Real>>(flags, source, source, norm, 0., source);
        }

};