// Monopole problem initialization with floors
#pragma once

#include "decs.hpp"
#include "types.hpp"

/**
 * Initialize a Blandford-Znajek monopole setup
 */
TaskStatus InitializeBZMonopole(MeshBlockData<Real> *rc, ParameterInput *pin);

