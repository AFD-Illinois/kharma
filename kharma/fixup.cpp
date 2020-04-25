// Fixups.  Apply limits and fix bad fluid values to maintain integrable state
// ApplyFloors, FixUtoP

#include "decs.hpp"

#include "debug.hpp"
#include "fixup.hpp"
#include "phys.hpp"
#include "U_to_P.hpp"

/**
 * Apply density and internal energy floors and ceilings
 */
TaskStatus ApplyFloors(Container<Real>& rc)
{
    FLAG("Apply floors");
    MeshBlock *pmb = rc.pmy_block;
    GridVars P = rc.Get("c.c.bulk.prims").data;
    GridVars U = rc.Get("c.c.bulk.cons").data;

    GridInt fflag("fflag", pmb->ncells3, pmb->ncells2, pmb->ncells1);

    Grid G(pmb);
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = new GammaLaw(gamma);

    // Note floors are applied only to physical zones
    // Therefore initialization, which requires initializing ghost zones, should *not* rely on a floors call for its operation
    pmb->par_for("apply_floors", pmb->ks, pmb->ke, pmb->js, pmb->je, pmb->is, pmb->ie,
        KOKKOS_LAMBDA_3D {
            fflag(k, j, i) = 0;
            fflag(k, j, i) |= fixup_ceiling(G, P, k, j, i);
            fflag(k, j, i) |= fixup_floor(G, P, U, eos, k, j, i);
        }
    );

#if DEBUG
    // Print some diagnostic info about which floors were hit
    count_print_fflags(pmb, fflag);
#endif

    FLAG("Applied");
    return TaskStatus::complete;
}

/**
 * Exclude certain zones from being used for fixups.
 *
 * TODO add back all but the physical corners
 */
void ClearCorners(MeshBlock *pmb, GridInt pflag) {
    int is = pmb->is, ie = pmb->ie;
    int js = pmb->js, je = pmb->je;
    int ks = pmb->ks, ke = pmb->ke;
    pmb->par_for("clear_corners", 0, NGHOST-1, 0, NGHOST-1, 0, NGHOST-1,
        KOKKOS_LAMBDA_3D {
            //if(global_start[2] == 0 && global_start[1] == 0 && global_start[0] == 0)
            pflag(k, j, i) = -1;
            //if(global_start[2] == 0 && global_start[1] == 0 && global_stop[0] == N1TOT)
            pflag(k, j, ie+i) = -1;
            //if(global_start[2] == 0 && global_stop[1] == N2TOT && global_start[0] == 0)
            pflag(k, je+j, i) = -1;
            //if(global_stop[2] == N3TOT && global_start[1] == 0 && global_start[0] == 0)
            pflag(ke+k, j, i) = -1;
            //if(global_start[2] == 0 && global_stop[1] == N2TOT && global_stop[0] == N1TOT)
            pflag(k, je+j, ie+i) = -1;
            //if(global_stop[2] == N3TOT && global_start[1] == 0 && global_stop[0] == N1TOT)
            pflag(ke+k, j, ie+i) = -1;
            //if(global_stop[2] == N3TOT && global_stop[1] == N2TOT && global_start[0] == 0)
            pflag(ke+k, je+j, i) = -1;
            //if(global_stop[2] == N3TOT && global_stop[1] == N2TOT && global_stop[0] == N1TOT)
            pflag(ke+k, je+j, ie+i) = -1;
        }
    );
}