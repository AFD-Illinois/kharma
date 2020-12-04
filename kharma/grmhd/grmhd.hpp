/**
 * This physics package implements General-Relativistic Magnetohydrodynamics
 *
 * Unlike MHD, GRMHD has two independent sets of variables: the conserved variables, and a set of
 * "primitive" variables more amenable to reconstruction.  To evolve the fluid, the conserved
 * variables must be:
 * 1. Transformed to the primitives
 * 2. Reconstruct the right- and left-going components at zone faces
 * 3. Transform back to conserved quantities and calculate the fluxes at faces
 * 4. Update conserved variables using the divergence of conserved fluxes
 * 
 * (for higher-order schemes, this is more or less just repeated and added)
 *
 * iharm3d puts step 1 at the bottom, and syncs/fixes primitive variables between each step.
 * KHARMA runs through the steps as listed, applying floors after step 1 as iharm3d does, but
 * syncing the conserved variables 
 */
#pragma once

#include <memory>

#include "parthenon/parthenon.hpp"

using namespace parthenon;

namespace GRMHD {
    // For declaring meshes, as well as the full intermediates we need (right & left fluxes etc)
    std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

    // Tasks to implement the interface:
    // The "FillDerived" equivalent should return with all derived variables in the StateDescriptor in consistent state for e.g. output
    // For HARM this means running U_to_P, and applying any inversion fixes and floor criteria
    void UtoP(std::shared_ptr<MeshBlockData<Real>>& rc);
    // Constrained-transport step to preserve divB==0
    TaskStatus FluxCT(std::shared_ptr<MeshBlockData<Real>>& rc);
    // Add the HARM source term to the RHS dudt
    TaskStatus AddSourceTerm(std::shared_ptr<MeshBlockData<Real>>& rc, std::shared_ptr<MeshBlockData<Real>>& dudt);
    // Or apply the flux divergence and add the source in a quick way
    TaskStatus ApplyFluxes(std::shared_ptr<MeshBlockData<Real>>& rc, std::shared_ptr<MeshBlockData<Real>>& dudt);
    // Estimate the next timestep. For pure GRMHD, this is the minimum signal crossing time of a zone on the block
    Real EstimateTimestep(std::shared_ptr<MeshBlockData<Real>>& rc);
}