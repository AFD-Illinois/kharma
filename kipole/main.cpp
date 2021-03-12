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

// KIPOLE Headers
#include "decs.hpp"

#include "image_driver.hpp"
#include "kipole.hpp"

// Parthenon headers
#include <parthenon/parthenon.hpp>

// Print warnings about configuration
#if DEBUG
#warning "Compiling with debug"
#endif

using namespace parthenon;

/**
 * Main function for KIPOLE.
 *
 * Currently available drivers:
 * ImageDriver: Single fast-light image from a GRMHD dump or analytic expression
 * 
 * Eventual drivers:
 * SlowLightDriver: Produce images by interpolating dump files or analytic functions
 */
int main(int argc, char *argv[])
{
    ParthenonManager pman;

    pman.app_input->ProcessPackages = KIPOLE::ProcessPackages;
    pman.app_input->ProcessProperties = KIPOLE::ProcessProperties;
    //pman.app_input->ProblemGenerator = KIPOLE::LoadModel;
    //pman.app_input->UserWorkBeforeOutput = KIPOLE::FillOutput;

    // Parthenon init includes Kokkos, MPI, parses parameters & cmdline,
    // then calls ProcessPackages and ProcessProperties, then constructs the Mesh
    FLAG("Parthenon Initializing");
    auto manager_status = pman.ParthenonInit(argc, argv);
    if (manager_status == ParthenonStatus::complete) {
        // TODO use this as an option to just write out the gridfile, initial mesh, etc.
        pman.ParthenonFinalize();
        return 0;
    }
    if (manager_status == ParthenonStatus::error) {
        pman.ParthenonFinalize();
        return 1;
    }
    FLAG("Parthenon Initialized");

    auto pin = pman.pinput.get();
    auto pmesh = pman.pmesh.get();
    auto papp = pman.app_input.get();

    if(MPIRank0() && pin->GetInteger("debug", "verbose") > 0) {
        // This dumps the full Kokkos config, useful for double-checking
        // that the compile did what we wanted
        ShowConfig();
        pin->ParameterDump(cout);
    }

    // Then construct & run the driver
    ImageDriver driver(pin, papp, pmesh);

    FLAG("Executing Driver");
    auto driver_status = driver.Execute();

    // Parthenon cleanup includes Kokkos, MPI
    FLAG("Finalizing");
    pman.ParthenonFinalize();

    return 0;
}
