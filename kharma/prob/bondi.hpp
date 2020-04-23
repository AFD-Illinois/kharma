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
void bondi(MeshBlock *pmb, const Grid& G, GridVars P,
                    const EOS* eos, const Real mdot, const Real rs);

/**
 * Custom boundary function to set ghost zones for stable flow
 * See hook into HARMDriver's Parthenon task list
 */
TaskStatus ApplyBondiBoundary(Container<Real>& rc);
