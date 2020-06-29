
#include "fixup.hpp"

/**
 * Exclude certain zones from being used for fixups.
 *
 * TODO add back all but the physical corners
 */
void ClearCorners(MeshBlock *pmb, GridInt pflag) {
    // TODO make corner domains for the loop here
    // TODO this doesn't work with <3 dimensions
    FLAG("Clearing corners");
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    
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

    FLAG("Cleared corner flags");
}