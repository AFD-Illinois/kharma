// 

#pragma once

#include "decs.hpp"
#include <parthenon/parthenon.hpp>

/**
 * Initialize the entire mesh.  This involves:
 * 
 * 1. Initializing physical zones with InitializeProblem
 * 2. Syncing to fill ghost zones
 * 3. Seeding any extra B field required by the problem
 * 4. Syncing that
 * 
 * At the end of *each* substep, the primitives P and
 * conserved variables U correspond to each others' values.
 * This is important and generally denoted LOCKSTEP in the code
 */
void InitializeMesh(ParameterInput *pin, Mesh *pmesh);

/**
 * Generate the initial condition on (the physical zones of) a meshblock
 * Takes the problem name from parameter "parthenon/job" and parameters from several sections
 * as documented in example parameter files
 *
 * Problems:
 * torus: Fishbone-Moncrief torus, main initialization for accretion runs.
 *        Must be seeded with any magnetic field required.
 * bondi: Bondi flow accretion, should be static when used alone.
 *        May optionally be seeded with B to make it more interesting
 * 
 * Tests:
 * mhdmodes: Take a wave mode of the MHD equations and follow it once across the domain
 * (soon) orszag_tang
 * (soon) explosion: Spherical (& cylindrical?) Komissarov explosion problems
 * 
 * Restarts:
 * iharm_restart: Read a checkpoint file from iharm3d and continue that simulation.
 *                Note this can't yet replace Parthenon parameters, so they must be
 *                specified in input correctly.
 */
TaskStatus InitializeProblem(std::shared_ptr<Container<Real>>& rc, ParameterInput *pin);

/**
 * Force a Parthenon boundary synchronization/ghost zone fill
 * 
 * This is a custom re-roll of what Parthenon does per-step, used because I
 * need to control syncs during initialization.  Do NOT use elsewhere
 */
void SyncAllBounds(Mesh *pmesh);