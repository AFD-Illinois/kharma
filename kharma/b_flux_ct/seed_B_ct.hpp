// Seed a torus of some type with a magnetic field according to its density
#pragma once

#include "decs.hpp"
#include "types.hpp"

namespace B_FluxCT
{

/**
 * Seed an axisymmetric initialization with magnetic field proportional to fluid density,
 * or density and radius, to create a SANE or MAD flow
 * Note this function expects a normalized P for which rho_max==1
 *
 * @param rin is the interior radius of the torus
 * @param min_rho_q is the minimum density at which there will be magnetic vector potential
 * @param b_field_type is one of "sane" "ryan" "r3s3" or "gaussian", described below (TODO test or remove opts)
 */
TaskStatus SeedBField(MeshBlockData<Real> *rc, ParameterInput *pin);

/**
 * Add flux to BH horizon
 * Applicable to any Kerr-space GRMHD sim, run after import/initialization
 * Preserves divB==0 with a Flux-CT step at end
 */
//void SeedBHFlux(MeshBlockData<Real> *rc, Real BHflux);

} // namespace B_FluxCT
