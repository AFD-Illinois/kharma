// K-HARM/KHARMA open-source GRMHD code
// Based on Parthenon structured-grid AMR library
// Adapted heavily from example therein
// (c) Illinois AFD Group, released under BSD 3-clause license

// Parthenon headers
#include "parthenon_manager.hpp"

// KHARMA Headers
#include "decs.hpp"
#include "harm.hpp"

// Warn once *during compile* specifying whether CUDA is being used
// This catches a lot of configuration mistakes
#if defined( Kokkos_ENABLE_CUDA )
#warning "Compiling with CUDA"
#else
#warning "Compiling with OpenMP Only"
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
int main(int argc, char *argv[]) {
  ParthenonManager pman;

  FLAG("Initializing");
  auto manager_status = pman.ParthenonInit(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  FLAG("Initialized");
  ShowConfig();

  HARMDriver driver(pman.pinput.get(), pman.pmesh.get(), pman.pouts.get());

  // start a timer
  pman.PreDriver();

  auto driver_status = driver.Execute();

  // Make final outputs, print diagnostics
  pman.PostDriver(driver_status);

  // call MPI_Finalize if necessary
  pman.ParthenonFinalize();

  return 0;
}
