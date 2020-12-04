//
#pragma once

#include "decs.hpp"

#include "bondi.hpp"
#include "phys.hpp"

/**
 * Any user-defined boundaries, i.e. not traditional periodic/outflow/reflecting but some function
 * Note it needs *P* unlike Parthenon boundaries that need *U*
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
TaskStatus ApplyCustomBoundaries(std::shared_ptr<MeshBlockData<Real>>& rc);

/**
 * Fix fluxes on physical boundaries. Ensure no inflow flux, correct B fields on reflecting conditions.
 */
TaskStatus FixFlux(std::shared_ptr<MeshBlockData<Real>>& rc);