// Seed a torus of some type with a magnetic field according to its density
#pragma once

#include "decs.hpp"
#include <parthenon/parthenon.hpp>

// Internal representation of the field initialization preference for quick switch
// Avoids string comparsion in kernels
enum BSeedType{constant, monopole, sane, ryan, r3s3, gaussian};

/**
 * Get the minimum value of plasma beta on the (physical, non-ghost) domain
 * 
 * The "legacy" option first calculates bsq_max and p_max (wherever they may be),
 * then divides them.  This is to emulate iharm2d/3d for comparisons, it is *not*
 * consistent under multiple meshes/MPI
 */
Real GetLocalBetaMin(MeshBlockData<Real> *rc);

/**
 * Version of the above for legacy support -- iharm2d/3d initialize "beta_min" not
 * locally but with global max p and max bsq.
 */
Real GetLocalBsqMax(MeshBlockData<Real> *rc);
Real GetLocalPMax(MeshBlockData<Real> *rc);

/**
 * Normalize the magnetic field by dividing by 'factor'
 * 
 * LOCKSTEP: this function expects and should preserve P==U
 */
TaskStatus NormalizeBField(MeshBlockData<Real> *rc, Real factor);
