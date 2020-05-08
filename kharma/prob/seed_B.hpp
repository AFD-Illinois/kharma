// Seed a torus of some type with a magnetic field according to its density
#pragma once

#include "decs.hpp"

#include "phys.hpp"

// Internal representation of the field initialization preference for quick switch
// Mostly for fun; the loop for vector potential is 2D
enum BSeedType{sane, ryan, r3s3, gaussian};

/**
 * Seed an axisymmetric initialization with magnetic field proportional to fluid density,
 * or density and radius, to create a SANE or MAD flow
 * Note this function expects a normalized P for which rho_max==1
 *
 * @param rin is the interior radius of the torus
 * @param min_rho_q is the minimum density at which there will be magnetic vector potential
 * @param b_field_type is one of "sane" "ryan" "r3s3" or "gaussian", described below (TODO test or remove opts)
 */
void SeedBField(MeshBlock *pmb, Grid G, GridVars P,
                Real rin, Real min_rho_q, std::string b_field_type);
/**
 * Get the minimum beta on the domain
 */
Real GetLocalBetaMin(MeshBlock *pmb);

/**
 * Normalize the magnetic field
 * 
 * LOCKSTEP: this function expects and should preserve P<->U
 */
void NormalizeBField(MeshBlock *pmb, Real factor);