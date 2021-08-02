//
#pragma once

#include "decs.hpp"

#include "bondi.hpp"
#include "mhd_functions.hpp"

/**
 * Any KHARMA-defined boundaries.
 * These are equivalent to Parthenon's implementations of the same, except that they
 * operate on the fluid primitive variables p,u,u1,u2,u3.  All other variables
 * are unchanged.
 * 
 * LOCKSTEP: these functions respect P and returns consistent P<->U
 */
void OutflowInnerX1_KHARMA(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OutflowOuterX1_KHARMA(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectInnerX2_KHARMA(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectOuterX2_KHARMA(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);

/**
 * Fix fluxes on physical boundaries. Ensure no inflow flux, correct B fields on reflecting conditions.
 */
TaskStatus FixFlux(MeshBlockData<Real> *rc);
