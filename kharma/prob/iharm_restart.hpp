// Load the grid variables up with primitives from an old iharm3d run
#pragma once

#include "decs.hpp"

#include "mesh/mesh.hpp"

/**
 * Read the header of an iharm3d HDF5 restart file, and set appropriate parameters
 * Call this before mesh creation!
 */
void ReadIharmRestartHeader(std::string fname, std::unique_ptr<ParameterInput>& pin);

/**
 * Read data from an iharm3d restart file. Does not support >1 meshblock in Parthenon
 * 
 * Returns stop time tf of the original simulation, for e.g. replicating regression tests
 */
double ReadIharmRestart(MeshBlock *pmb, GRCoordinates G, GridVars P, std::string fname);
