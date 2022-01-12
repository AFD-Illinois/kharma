// Fishbone-Moncrief torus initialization functions
#pragma once

#include "decs.hpp"
#include <parthenon/parthenon.hpp>



/**
 * Initialize a wide variety of different fishbone-moncrief torii.
 *
 * @param rin is the torus innermost radius, in r_g
 * @param rmax is the radius of maximum density of the F-M torus in r_g
 */
TaskStatus InitializeBZMonopole(MeshBlockData<Real> *rc, ParameterInput *pin);

