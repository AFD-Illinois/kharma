/*
 * Bondi flow functions
 */

#include "decs.hpp"

#include "grid.hpp"
#include "eos.hpp"

/**
 * Initialization of a Bondi problem with specified sonic point, BH mdot, and horizon radius
 * TODO this can/should be just mdot (and the grid ofc), if this problem is to be used as anything more than a test
 */
void InitializeBondi(MeshBlock *pmb, const Grid& G, GridVars P,
                     const EOS* eos, const Real mdot, const Real rs);

/**
 * Apply the Bondi flow condition on right X1 boundary
 */
void ApplyBondiBoundary(Container<Real>& rc);
