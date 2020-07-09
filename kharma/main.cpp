// K-HARM/KHARMA open-source GRMHD code
// Based on Parthenon structured-grid AMR library
// Adapted heavily from example therein
// (c) Illinois AFD Group, released under BSD 3-clause license

// KHARMA Headers
#include "decs.hpp"

#include "harm.hpp"
#include "mpi.hpp"
#include "problem.hpp"

// Parthenon headers
#include <parthenon/parthenon.hpp>

// Print warnings about configuration
#if DEBUG
#warning "Compiling with debug"
#endif

#if NGHOST < 3
#error "HARM needs 3 ghost cells!  Configure this in Parthenon CMakeLists!"
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
 * HARM: GRMHD using LLF with zone-centered fields
 *
 * Future drivers?
 * KHARMA: GRMHD using LLF with face-centered fields
 * bhlight: GRMHD with Monte Carlo particle transport
 */
int main(int argc, char *argv[])
{
    ParthenonManager pman;

    FLAG("Parthenon Initializing");
    auto manager_status = pman.ParthenonInit(argc, argv);
    if (manager_status == ParthenonStatus::complete) {
        pman.ParthenonFinalize();
        return 0;
    }
    if (manager_status == ParthenonStatus::error) {
        pman.ParthenonFinalize();
        return 1;
    }
    FLAG("Parthenon Initialized");
    if(MPIRank0()) ShowConfig();

    auto pin = pman.pinput.get();
    auto pmesh = pman.pmesh.get();
    InitializeMesh(pin, pmesh);

    HARMDriver driver(pin, pmesh);

    FLAG("Executing Driver");
    auto driver_status = driver.Execute();

    // Parthenon cleanup includes Kokkos, MPI
    FLAG("Finalizing");
    pman.ParthenonFinalize();

    return 0;
}