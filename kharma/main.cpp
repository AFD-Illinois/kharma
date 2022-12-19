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
#include "imex_driver.hpp"
#include "harm_driver.hpp"
#include "kharma.hpp"
#include "mpi.hpp"
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

    pman.app_input->ProcessPackages = KHARMA::ProcessPackages;
    pman.app_input->ProblemGenerator = KHARMA::ProblemGenerator;
    pman.app_input->MeshBlockUserWorkBeforeOutput = KHARMA::FillOutput;
    pman.app_input->PreStepMeshUserWorkInLoop = KHARMA::PreStepMeshUserWorkInLoop;
    pman.app_input->PostStepMeshUserWorkInLoop = KHARMA::PostStepMeshUserWorkInLoop;
    pman.app_input->PostStepDiagnosticsInLoop = KHARMA::PostStepDiagnostics;

    // Registering KHARMA's boundary functions here doesn't mean they will *always* run:
    // all periodic boundary conditions are handled by Parthenon.
    // KHARMA sets the correct options automatically for spherical coordinate systems.
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] = KBoundaries::InnerX1;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] = KBoundaries::OuterX1;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x2] = KBoundaries::InnerX2;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x2] = KBoundaries::OuterX2;

    // Parthenon init includes Kokkos, MPI, parses parameters & cmdline,
    // then calls ProcessPackages and ProcessProperties, then constructs the Mesh
    Flag("Parthenon Initializing");
    auto manager_status = pman.ParthenonInit(argc, argv);
    if (manager_status == ParthenonStatus::complete) {
        pman.ParthenonFinalize();
        return 0;
    }
    if (manager_status == ParthenonStatus::error) {
        pman.ParthenonFinalize();
        return 1;
    }
    Flag("Parthenon Initialized");

#if DEBUG
    // Replace Parthenon signal handlers with something that just prints a backtrace
    signal(SIGINT, print_backtrace);
    signal(SIGTERM, print_backtrace);
    signal(SIGSEGV, print_backtrace);
#endif

    auto pin = pman.pinput.get(); // All parameters in the input file or command line
    auto pmesh = pman.pmesh.get(); // The mesh, with list of blocks & locations, size, etc
    auto papp = pman.app_input.get(); // The list of callback functions specified above

    // Add magnetic field to the problem, initialize ghost zones.
    // Implemented separately outside of MeshBlock since
    // this usually involves global reductions for normalization
    if(MPIRank0())
        std::cout << "Running post-initialization tasks..." << std::endl;

    auto prob = pin->GetString("parthenon/job", "problem_id");
    bool is_restart = (prob == "resize_restart") || pman.IsRestart();
    bool is_resize = (prob == "resize_restart") && !pman.IsRestart();
    KHARMA::PostInitialize(pin, pmesh, is_restart, is_resize);
    Flag("Post-initialization completed");

    // Construct a temporary driver purely for parameter parsing
    auto driver_type = pin->GetString("driver", "type");
    if (driver_type == "harm") {
        HARMDriver driver(pin, papp, pmesh);
    } else if (driver_type == "imex") {
        ImexDriver driver(pin, papp, pmesh);
    } else {
        throw std::invalid_argument("Expected driver type to be harm or imex!");
    }

    // We could still have set parameters during driver initialization
    // Note the order here is *extremely important* as the first statement has a
    // side effect which must occur on all MPI ranks
    if(pin->GetOrAddBoolean("debug", "archive_parameters", false) && MPIRank0()) {
        // Write *all* parameters to a parfile for posterity
        std::ostringstream ss;
        auto itt_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        ss << "kharma_parsed_parameters_" << std::put_time(std::gmtime(&itt_now), "%FT%TZ") << ".par";
        std::fstream pars;
        pars.open(ss.str(), std::fstream::out | std::fstream::trunc);
        pin->ParameterDump(pars);
        pars.close();
    }
    // Also write parameters to console if we should be wordy
    if ((pin->GetInteger("debug", "verbose") > 0) && MPIRank0()) {
        // This dumps the full Kokkos config, useful for double-checking
        // that the compile did what we wanted
        ShowConfig();
        pin->ParameterDump(std::cout);
    }

    // Then execute the driver. This is a Parthenon function inherited by our HARMDriver object,
    // which will call MakeTaskCollection, then execute the tasks on the mesh for each portion
    // of each step until a stop criterion is reached.
    Flag("Executing Driver");

    if (driver_type == "harm") {
        std::cout << "Initializing and running KHARMA driver." << std::endl;
        HARMDriver driver(pin, papp, pmesh);
        auto driver_status = driver.Execute();
    } else if (driver_type == "imex") {
        std::cout << "Initializing and running IMEX driver." << std::endl;
        ImexDriver driver(pin, papp, pmesh);
        auto driver_status = driver.Execute();
    }

#ifndef KOKKOS_ENABLE_CUDA
    // Cleanup our global NDArray
    extern ParArrayND<double> p_bound;
    p_bound.~ParArrayND<double>();
#endif
    // Parthenon cleanup includes Kokkos, MPI
    Flag("Finalizing");
    pman.ParthenonFinalize();

    return 0;
}
