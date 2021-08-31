// 

#pragma once

#include "decs.hpp"
#include <parthenon/parthenon.hpp>

/**
 * Generate the initial condition on (the physical zones of) a meshblock
 * This is the callback from Parthenon -- we apply normalization and transformation
 * afterward
 * 
 * An example of running each supported problem with parameters is provided in the pars/ folder
 */
namespace KHARMA {

/**
 * Generate the initial conditions inside a meshblock according to the input parameters.
 * This mostly involves including the rest of the code from the prob/ folder, and calling
 * the appropriate function based on the parameter "problem_id"
 * This function also performs some initial consistency operations, such as applying floors,
 * calculating the conserved values, and synchronizing boundaries.
 */
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);

}
