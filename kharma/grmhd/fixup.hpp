
// Routines for fixing integration failures, but not like, all integration failures

#include "decs.hpp"

#include "mesh/mesh.hpp"

#include "floors.hpp"

/**
 * Clear corner zones from being used for fixups
 */
void ClearCorners(MeshBlock *pmb, GridInt pflag);

/**
 * Smooth over inversion failures by averaging values from each neighboring zone
 * a.k.a. Diffusion?  What diffusion?  There is no diffusion here.
 *
 * TODO parallelize this
 * LOCKSTEP: this function expects and should preserve P<->U
 * FLUID: this function 
 */
KOKKOS_INLINE_FUNCTION int fix_U_to_P(const GRCoordinates& G, GridVars P, GridVars U, EOS *eos, GridInt pflag, const int& k, const int& j, const int& i)
{
    int fflag = 0;

    // Negative flags are physical corners, which shouldn't be fixed
    if (pflag(k, j, i) > InversionStatus::success) {
        double wsum = 0.;
        double sum[NFLUID] = {0};
        // For all neighboring cells...
        for (int n = -1; n <= 1; n++) {
            for (int m = -1; m <= 1; m++) {
                for (int l = -1; l <= 1; l++) {
                    // Weight by distance and whether the cell is itself flagged
                    // interpolated "fixed" cells stay flagged
                    double w = 1./(abs(l) + abs(m) + abs(n) + 1) *
                        (pflag(k+n, j+m, i+l) == InversionStatus::success);
                    wsum += w;
                    FLOOP sum[p] += w * P(p, k+n, j+m, i+l);
                }
            }
        }

        if(wsum < 1.e-10) {
#if DEBUG
            //printf("fixup_utoprim: No usable neighbors at %d %d %d\n", i, j, k);
#endif
            // TODO set to something ~okay here and LOG IT, or exit screaming
            // This should happen /very rarely/
            //exit(-1);
        } else {
            FLOOP P(p, k, j, i) = sum[p]/wsum;

            // Make sure fixed values still abide by floors
            fflag |= fixup_ceiling(G, P, U, eos, k, j, i);
            fflag |= fixup_floor(G, P, U, eos, k, j, i);
            // Make sure the original conserved variables match
            FourVectors Dtmp;
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
        }
    }

    return fflag;
}