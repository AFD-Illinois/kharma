/* 
 *  File: main.cpp
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

// KHARMA Headers
#include "decs.hpp"

#include "boundaries.hpp"
#include "kharma_driver.hpp"
#include "kharma.hpp"
#include "post_initialize.hpp"
#include "problem.hpp"
#include "emhd/conducting_atmosphere.hpp"

// Parthenon headers
#include <parthenon/parthenon.hpp>

// Local headers
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

// Print warnings about configuration
#if DEBUG
#warning "Compiling with debug"

// Stacktrace on sigint. Amazingly useful
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
void print_backtrace(int sig) {
  void *array[100];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 100);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}
#endif

using namespace parthenon;

/**
 * Main function for KHARMA.  Basically a wrapper calling a particular driver class to
 * handle fluid evolution.
 *
 * Different driver classes can be switched out to implement different algorithms or
 * sets of physical processes, while re-using particular physics packages to mix and match
 *
 * Currently available drivers:
 * HARM: GRMHD using LLF with zone-centered fields, conserved variables are synchronized
 * Imex: same functionality HARM but primitive variables are synchronized,
 *       optionally uses per-zone implicit solve for some variables, for e.g. Extended GRMHD
 *
 * Future drivers?
 * bhlight: GRMHD with Monte Carlo particle transport
 */
int main(int argc, char *argv[])
{
    ParthenonManager pman;

    // A couple of callbacks are KHARMA-wide single functions
    pman.app_input->ProcessPackages = KHARMA::ProcessPackages;
    pman.app_input->ProblemGenerator = KHARMA::ProblemGenerator;
    // A few are passed on to be implemented by packages as they see fit
    pman.app_input->MeshBlockUserWorkBeforeOutput = Packages::UserWorkBeforeOutput;
    pman.app_input->PreStepMeshUserWorkInLoop = Packages::PreStepUserWorkInLoop;
    pman.app_input->PostStepMeshUserWorkInLoop = Packages::PostStepUserWorkInLoop;
    pman.app_input->PostStepDiagnosticsInLoop = Packages::PostStepDiagnostics;

    // Registering KHARMA's boundary functions here doesn't mean they will *always* run:
    // periodic & internal boundary conditions are handled by Parthenon.
    // KHARMA sets what will run in boundaries.cpp
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] = KBoundaries::ApplyBoundaryTemplate<IndexDomain::inner_x1>;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] = KBoundaries::ApplyBoundaryTemplate<IndexDomain::outer_x1>;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x2] = KBoundaries::ApplyBoundaryTemplate<IndexDomain::inner_x2>;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x2] = KBoundaries::ApplyBoundaryTemplate<IndexDomain::outer_x2>;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x3] = KBoundaries::ApplyBoundaryTemplate<IndexDomain::inner_x3>;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x3] = KBoundaries::ApplyBoundaryTemplate<IndexDomain::outer_x3>;

    // Parthenon init includes Kokkos, MPI, parses parameters & cmdline,
    // then calls ProcessPackages and ProcessProperties, then constructs the Mesh
    Flag("Parthenon Init");
    auto manager_status = pman.ParthenonInit(argc, argv);
    if (manager_status == ParthenonStatus::complete) {
        pman.ParthenonFinalize();
        return 0;
    }
    if (manager_status == ParthenonStatus::error) {
        pman.ParthenonFinalize();
        return 1;
    }
    EndFlag("Parthenon Init");

#if DEBUG
    // Replace Parthenon signal handlers with something that just prints a backtrace
    signal(SIGINT, print_backtrace);
    signal(SIGTERM, print_backtrace);
    signal(SIGSEGV, print_backtrace);
#endif

    {
        auto pin = pman.pinput.get(); // All parameters in the input file or command line
        auto pmesh = pman.pmesh.get(); // The mesh, with list of blocks & locations, size, etc
        auto papp = pman.app_input.get(); // The list of callback functions specified above

        if(MPIRank0()) {
            // Note reading "verbose" parameter from "Globals" instead of pin: it may change during simulation
            if (pmesh->packages.Get("Globals")->Param<int>("verbose") > 0) {
                // Print a list of all loaded packages.  Surprisingly useful for debugging init logic
                std::cout << "Packages in use: " << std::endl;
                for (auto package : pmesh->packages.AllPackages()) {
                    std::cout << package.first << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << "Running post-initialization tasks..." << std::endl;
        }

        // PostInitialize: Add magnetic field to the problem, initialize ghost zones.
        // Any init which may be run even when restarting, or requires all
        // MeshBlocks to be initialized already
        auto prob = pin->GetString("parthenon/job", "problem_id");
        bool is_restart = (prob == "resize_restart") || (prob == "resize_restart_kharma") || pman.IsRestart();
        KHARMA::PostInitialize(pin, pmesh, is_restart);
        Flag("Post-initialization completed");

        // Construct a temporary driver purely for parameter parsing
        KHARMADriver driver(pin, papp, pmesh);

        // Write parameters to console if we should be wordy
        if ((pmesh->packages.Get("Globals")->Param<int>("verbose") > 0) && MPIRank0()) {
            // This dumps the full Kokkos config, useful for double-checking
            // that the compile did what we wanted
            ShowConfig();
            pin->ParameterDump(std::cout);
        }

        // Then execute the driver. This is a Parthenon function inherited by our HARMDriver object,
        // which will call MakeTaskCollection, then execute the tasks on the mesh for each portion
        // of each step until a stop criterion is reached.
        Flag("Executing Driver");
        auto driver_status = driver.Execute();
    }

    // Parthenon cleanup includes Kokkos, MPI
    Flag("Finalizing");
    pman.ParthenonFinalize();

    return 0;
}
