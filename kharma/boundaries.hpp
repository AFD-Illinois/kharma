//
#pragma once

#include "decs.hpp"

#include "bondi.hpp"
#include "phys.hpp"

/**
 * Any user-defined boundaries, i.e. not traditional periodic/outflow/reflecting but some function
 * Currently only used for Bondi problem, but may include an inflow check in future.
 */
TaskStatus ApplyCustomBoundaries(std::shared_ptr<Container<Real>>& rc);

/**
 * Fix fluxes on physical boundaries. Ensure no inflow flux, correct B fields on reflecting conditions.
 * TODO I bet strongly that Parthenon does this, if given to understand B is a vector
 */
void FixFlux(std::shared_ptr<Container<Real>>& rc);