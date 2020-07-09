// Fixups.  Apply limits and fix bad fluid values to maintain integrable state
// ApplyFloors, FixUtoP

#include "floors.hpp"

#include "debug.hpp"
#include "fixup.hpp"
#include "phys.hpp"
#include "U_to_P.hpp"

/**
 * Apply density and internal energy floors and ceilings
 * 
 * Note that fixup_ceiling and fixup_floor are called from some other places for most applications
 * This applies only to the physical zones.  It's used in initialization just before the first boundary sync.
 * 
 * LOCKSTEP: this function 
 */
TaskStatus ApplyFloors(std::shared_ptr<Container<Real>>& rc)
{
    FLAG("Apply floors");
    MeshBlock *pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GRCoordinates G = pmb->coords;

    GridInt fflag("fflag", n3, n2, n1);

    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    // Note floors are applied only to physical zones
    pmb->par_for("apply_floors", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            fflag(k, j, i) = 0;
            fflag(k, j, i) |= fixup_ceiling(G, P, U, eos, k, j, i);
            fflag(k, j, i) |= (fixup_floor(G, P, U, eos, k, j, i) / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;
        }
    );

    DelEOS(eos);

#if DEBUG
    // Print some diagnostic info about which floors were hit
    CountFFlags(pmb, fflag.GetHostMirrorAndCopy());
#endif

    FLAG("Applied");
    return TaskStatus::complete;
}