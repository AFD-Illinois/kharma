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
#include "mpi.hpp"
#include "problem.hpp"
#include "seed_B.hpp"

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

    auto pin = pman.pinput.get();
    // Initialize the problem on each meshblock
    // TODO this is *instead* of defining MeshBlock::ProblemGenerator,
    // which means that initial auto-refinement will *NOT* work.
    // we're a ways from AMR yet so we'll cross that bridge etc.
    MeshBlock *pmb = pman.pmesh->pblock;
    while (pmb != nullptr) {
        // Initialize the block
        InitializeProblem(pin, pmb);
        pmb = pmb->next;
    }

    // Normalize the field in a 2nd pass for torus problems
    if (pin->GetString("parthenon/job", "problem_id") == "torus" &&
        pin->GetOrAddString("torus", "b_field_type", "none") != "none") {
        // Normalize the magnetic field by first calculating the global min beta...
        Real beta_min = 1e100;
        MeshBlock *pmb = pman.pmesh->pblock;
        while (pmb != nullptr) {
            Real beta_local = GetLocalBetaMin(pmb);
            if(beta_local < beta_min) beta_min = beta_local;
            pmb = pmb->next;
        }
        beta_min = mpi_min(beta_min);

        // Then normalizing by sqrt(beta/beta_min)
        Real beta = pin->GetOrAddReal("torus", "beta_min", 100.);
        Real factor = sqrt(beta/beta_min);
        pmb = pman.pmesh->pblock;
        while (pmb != nullptr) {
            NormalizeBField(pmb, factor);
            pmb = pmb->next;
        }
    }

    HARMDriver driver(pin, pman.pmesh.get());

    auto driver_status = driver.Execute();

    // call MPI_Finalize if necessary
    pman.ParthenonFinalize();

    return 0;
}
