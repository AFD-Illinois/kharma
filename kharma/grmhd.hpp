/**
 * This physics package implements General-Relativistic Magnetohydrodynamics
 * 
 * Unlike MHD, GRMHD has two independent sets of variables: the conserved variables, and a set of "primitive" variables more
 * amenable to reconstruction.  To evolve the fluid, the conserved variables must be:
 * 1. Transformed to the primitives (ConsToPrim)
 * 2. Have the right- and left-going components reconstructed (ideally via WENO5) (reconstruct)
 * 3. Merge these components into fluxes at zone faces (lr_to_flux)
 * 4. Transform *back* to conserved variable fluxes at faces (prim_to_flux)
 * 5. Update conserved variables via finite-differencing (flux kernel in advance_fluid) ()
 * 
 * HARM3D puts step 1 at the bottom, and syncs primitive variables, but either set can be treated as "fundamental,"
 * depending on what is easier to carry around or debug, and whether Parthenon has a better time reconstructing or
 * finite-differencing a derived mesh.
 */
#pragma once

#include <memory>

#include "interface/StateDescriptor.hpp"
#include "task_list/tasks.hpp"
#include "parameter_input.hpp"

using namespace parthenon;

namespace GRMHD {
    // For declaring meshes, as well as the full intermediates we need (right & left fluxes etc)
    std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

    // Necessary tasks for interface
    TaskStatus FillDerived(Container<Real>& rc);  // TODO not sure how this fits with HARM's many, many derived vars
    TaskStatus EstimateTimestep(Container<Real>& rc);

    // Full task to advance fluid by a certain timestep
    TaskStatus AdvanceFluid(MeshBlock *pmb);

    // Sub-tasks
    TaskStatus ConstoPrim(MeshBlock *pmb);
    TaskStatus Reconstruct(MeshBlock *pmb);
    TaskStatus CalculateFluxes(MeshBlock *pmb);
}