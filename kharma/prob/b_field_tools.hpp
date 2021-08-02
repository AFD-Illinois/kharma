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
 * Likely not actually what you want
 */
Real GetLocalBetaMin(MeshBlockData<Real> *rc);

/**
 * Compatible versions of the above, taking max bsq and max p separately over
 * the whole grid.
 */
Real GetLocalBsqMax(MeshBlockData<Real> *rc);
Real GetLocalPMax(MeshBlockData<Real> *rc);

/**
 * Normalize the magnetic field by dividing by 'factor'
 * 
 * LOCKSTEP: this function expects and preserves P==U
 */
TaskStatus NormalizeBField(MeshBlockData<Real> *rc, Real factor);
