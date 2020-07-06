// Load the grid variables up with primitives from an old iharm3d run
#pragma once

#include "decs.hpp"

#include "mesh/mesh.hpp"

double ReadIharmRestart(MeshBlock *pmb, GRCoordinates G, GridVars P, std::string fname);