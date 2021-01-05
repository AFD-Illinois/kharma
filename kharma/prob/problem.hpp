// 

#pragma once

#include "decs.hpp"
#include <parthenon/parthenon.hpp>

/**
 * Generate the initial condition on (the physical zones of) a meshblock
 * This is the callback from Parthenon -- we apply normalization and transformation
 * afterward
 * 
 * An example of each supported problem with parameters is provided in the pars/ folder
 */
namespace KHARMA {
    void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
}

/**
 * Post-initialization functions:
 * 1. Normalize magnetic field to respect beta_min parameter
 * 2. Boundary sync
 */
void PostInitialize(ParameterInput *pin, Mesh *pmesh);

/**
 * Force a Parthenon boundary synchronization/ghost zone fill
 * 
 * This is a custom re-roll of what Parthenon does automatically each step,
 * thus it does not usually need to be called 
 */
void SyncAllBounds(Mesh *pmesh);
