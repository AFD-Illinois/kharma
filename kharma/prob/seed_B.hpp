// Seed a torus of some type with a magnetic field according to its density
#pragma once

#include "decs.hpp"
#include <parthenon/parthenon.hpp>

/**
 * Seed an axisymmetric initialization with magnetic field proportional to fluid density,
 * or density and radius, to create a SANE or MAD flow
 * Note this function expects a normalized P for which rho_max==1
 *
 * @param rin is the interior radius of the torus
 * @param min_rho_q is the minimum density at which there will be magnetic vector potential
 * @param b_field_type is one of "sane" "ryan" "r3s3" or "gaussian", described below (TODO test or remove opts)
 */
TaskStatus SeedBField(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin);

/**
 * Get the minimum value of plasma beta on the (physical, non-ghost) domain
 */
Real GetLocalBetaMin(std::shared_ptr<MeshBlockData<Real>>& rc);

/**
 * Normalize the magnetic field by dividing by 'factor'
 * 
 * LOCKSTEP: this function expects and should preserve P==U
 */
TaskStatus NormalizeBField(std::shared_ptr<MeshBlockData<Real>>& rc, Real factor);

/**
 * Add flux to BH horizon
 * Applicable to any Kerr-space GRMHD sim, run after import/initialization
 * Preserves divB==0 with a Flux-CT step at end
 */
//void SeedBHFlux(std::shared_ptr<MeshBlockData<Real>>& rc, Real BHflux);