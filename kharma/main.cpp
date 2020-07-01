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
    // Initialize the problem on each meshblock
    // TODO this is *instead* of defining MeshBlock::ProblemGenerator,
    // which means that initial auto-refinement will *NOT* work.
    // we're a ways from AMR yet so we'll cross that bridge etc.
    MeshBlock *pmb = pman.pmesh->pblock;
    while (pmb != nullptr) {
        // Initialize the block
        InitializeProblem(pin, pmb);
        FLAG("Initialized Block");
        pmb = pmb->next;
    }
    FLAG("Initialized Mesh");

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
        beta_min = MPIMin(beta_min);

        if(MPIRank0()) cerr << "Min beta of " << beta_min << endl;

        // Then normalizing by sqrt(beta/beta_min)
        Real beta = pin->GetOrAddReal("torus", "beta_min", 100.);
        Real factor = sqrt(beta/beta_min);
        pmb = pman.pmesh->pblock;
        while (pmb != nullptr) {
            NormalizeBField(pmb, factor);
            pmb = pmb->next;
        }
    }
    FLAG("Normalized B Field");

    HARMDriver driver(pin, pman.pmesh.get());

    FLAG("Executing Driver");
    auto driver_status = driver.Execute();

    // Parthenon cleanup includes Kokkos, MPI
    FLAG("Finalizing");
    pman.ParthenonFinalize();

    return 0;
}
