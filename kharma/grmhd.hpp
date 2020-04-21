/**
 * This physics package implements General-Relativistic Magnetohydrodynamics
 * 
 * Unlike MHD, GRMHD has two independent sets of variables: the conserved variables, and a set of "primitive" variables more
 * amenable to reconstruction.  To evolve the fluid, the conserved variables must be:
 * 1. Transformed to the primitives (ConsToPrim)
 * 2. Have the right- and left-going components reconstructed (ideally via WENO5) (Reconstruct)
 * 3. Merge these components into fluxes at zone faces (LRToFlux)
 * 4. Transform *back* to conserved variable fluxes at faces (PrimToCons)
 * 5. Update conserved variables via finite-differencing (flux kernel in advance_fluid) ()
 * 
 * HARM3D puts step 1 at the bottom, and syncs primitive variables, but either set can be treated as "fundamental,"
 * depending on what is easier to carry around or debug, and whether Parthenon has a better time reconstructing or
 * finite-differencing a derived mesh.
 */
#pragma once

#include <memory>

#include "interface/state_descriptor.hpp"
#include "task_list/tasks.hpp"
#include "parameter_input.hpp"

using namespace parthenon;

namespace GRMHD {
    // For declaring meshes, as well as the full intermediates we need (right & left fluxes etc)
    std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

    // Tasks to implement the interface:
    // FillDerived should end up with all derived variables in the StateDescriptor in consistent state for e.g. output
    // For HARM this means running U_to_P to recover primitives in all zones
    void FillDerived(Container<Real>& rc);
    // Reconstruct the primitives and calculate the LLF fluxes in each direction
    TaskStatus CalculateFluxes(Container<Real>& rc);
    // Apply the fluxes to calculate dU/dt, the RHS of the PDE. Time integration is applied separately.
    TaskStatus ApplyFluxes(Container<Real>& rc, Container<Real>& base);
    // Estimate the next timestep. For pure GRMHD, this is the minimum signal crossing time of a zone on the block
    Real EstimateTimestep(Container<Real>& rc);
}