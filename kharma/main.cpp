// K-HARM/KHARMA open-source GRMHD code
// Based on Parthenon structured-grid AMR library
// Adapted heavily from example therein
// (c) Illinois AFD Group, released under BSD 3-clause license

// Parthenon headers
#include "parthenon_manager.hpp"

// KHARMA Headers
#include "decs.hpp"

#include "coordinate_systems.hpp"
#include "harm.hpp"

// Warn once *during compile* specifying whether CUDA is being used
// This catches a lot of configuration mistakes
#if defined( Kokkos_ENABLE_CUDA )
#warning "Compiling with CUDA"
#else
#warning "Compiling with OpenMP Only"
#endif

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

    FLAG("Initializing");
    auto manager_status = pman.ParthenonInit(argc, argv);
    // auto pin = pman.pinput.get();
    // bool re_init = false;

    // Sometimes we need to modify things that will affect the mesh/timeframe initialization
    // If coordinate system is KS or BL, we reverse out x1min from Rout/Rhor to put 5 zones in EH
    // auto coord_str = pin->GetOrAddString("coordinates", "base", "cartesian_minkowski");
    // if (coord_str.find("minkowski") == std::string::npos) { // If we're not in flat space...
    //     // Define the base/embedding coordinate system Kerr-Schild
    //     GReal a = pin->GetOrAddReal("coordinates", "a", 0.9375);
    //     GReal Rhor;
    //     if (coord_str.find("ks") != std::string::npos) {
    //         auto base_coords = SphKSCoords(a);
    //         Rhor = base_coords.rhor();
    //     } else if (coord_str.find("bl") != std::string::npos) {
    //         auto base_coords = SphBLCoords(a);
    //         Rhor = base_coords.rhor();
    //     } else {
    //         throw std::invalid_argument("Unsupported coordinate system for determining Rin!");
    //     }
    //     GReal Rout = pin->GetOrAddReal("coordinates", "r_out", 100.);
    //     // This will need to be revised for multi-level meshes
    //     int n1 = pin->GetInteger("mesh", "nx1");
    //     // Note this only applies to MKS/CMKS/FMKS
    //     GReal Rin = exp((n1 * log(Rhor) / 5.5 - log(Rout)) / (-1. + n1 / 5.5));
    //     std::vector<GReal> startx = {log(Rin), 0.0, 0.0};
    //     std::vector<GReal> stopx = {log(Rout), 1.0, 2*M_PI};
    //     re_init = true;
    // }

    // // Then re-initialize
    // if (re_init) {
    //     FLAG("Re-initializing");
    //     // TODO split Parthenon argument parsing from mesh initialization
    // }

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

    HARMDriver driver(pman.pinput.get(), pman.pmesh.get());

    // start a timer
    pman.PreDriver();

    auto driver_status = driver.Execute();

    // Make final outputs, print diagnostics
    pman.PostDriver(driver_status);

    // call MPI_Finalize if necessary
    pman.ParthenonFinalize();

    return 0;
}
