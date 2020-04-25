// Fixups.  Apply limits and fix bad fluid values to maintain integrable state
// ApplyFloors, FixUtoP

#include "decs.hpp"

#include "phys.hpp"
#include "U_to_P.hpp"

// TODO move these to runtime options with defaults
// GAMMA FLOOR
// Maximum gamma factor allowed for fluid velocity
// Defined in decs.hpp since it's also needed by U_to_P
//#define GAMMAMAX 200.

// GEOMETRY FLOORS
// Limiting values for density and internal energy
// These are scaled with radius for spherical sims,
// and multiplied by an additional 0.01 for cartesian sims
#define RHOMIN 1.e-6
#define UUMIN  1.e-8
// Hard limit values lest they scale too far
#define RHOMINLIMIT 1.e-20
#define UUMINLIMIT  1.e-20
// Radius in M, around which to steepen floor prescription from r^-2 to r^-3
#define FLOOR_R_CHAR 10.

// RATIO CEILINGS
// Maximum ratio of internal energy to density (i.e. Temperature)
#define UORHOMAX   100.
// Same for magnetic field (i.e. magnetization sigma)
#define BSQORHOMAX 100.
#define BSQOUMAX   (BSQORHOMAX * UORHOMAX)

KOKKOS_INLINE_FUNCTION int fixup_ceiling(const Grid& G, GridVars P, const int& k, const int& j, const int& i);
KOKKOS_INLINE_FUNCTION int fixup_floor(const Grid& G, GridVars P, GridVars U, EOS *eos, const int& k, const int& j, const int& i);

/**
 * Apply density and internal energy floors and ceilings
 */
TaskStatus ApplyFloors(Container<Real>& rc)
{
    FLAG("Apply floors");
    MeshBlock *pmb = rc.pmy_block;
    GridVars P = rc.Get("c.c.bulk.prims").data;
    GridVars U = rc.Get("c.c.bulk.cons").data;

    GridInt fflag("fflag", pmb->ncells1, pmb->ncells2, pmb->ncells3);

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

KOKKOS_INLINE_FUNCTION int fixup_ceiling(const Grid& G, GridVars P, const int& k, const int& j, const int& i)
{
    int fflag = 0;
    // First apply ceilings:
    // 1. Limit gamma with respect to normal observer
    Real gamma = mhd_gamma_calc(G, P, k, j, i, Loci::center);

    if (gamma > GAMMAMAX) {
        fflag |= HIT_FLOOR_GAMMA;

        Real f = sqrt((GAMMAMAX*GAMMAMAX - 1.)/(gamma*gamma - 1.));
        P(prims::u1, k, j, i) *= f;
        P(prims::u2, k, j, i) *= f;
        P(prims::u3, k, j, i) *= f;
    }
    return fflag;
}

KOKKOS_INLINE_FUNCTION int fixup_floor(const Grid& G, GridVars P, GridVars U, EOS *eos, const int& k, const int& j, const int& i)
{
    int fflag = 0;
    // Then apply floors:
    // 1. Geometric hard floors, not based on fluid relationships
    Real rhoflr_geom, uflr_geom;
    if(G.spherical) {
        GReal Xembed[NDIM];
        G.coord_embed(k, j, i, Loci::center, Xembed);
        GReal r = Xembed[1];

        // New, steeper floor in rho
        // Previously raw r^-2, r^-1.5
        Real rhoscal = pow(r, -2.) * 1 / (1 + r/FLOOR_R_CHAR);
        rhoflr_geom = RHOMIN*rhoscal;
        uflr_geom = UUMIN*pow(rhoscal, eos->gam);

        // Impose overall minimum
        // TODO These would only be hit at by r^-3 floors for r_out = 100,000M.  Worth keeping?
        rhoflr_geom = max(rhoflr_geom, RHOMINLIMIT);
        uflr_geom = max(uflr_geom, UUMINLIMIT);
    } else {
        rhoflr_geom = RHOMIN*1.e-2;
        uflr_geom = UUMIN*1.e-2;
    }

    // Record Geometric floor hits
    if (rhoflr_geom > P(prims::rho, k, j, i)) fflag |= HIT_FLOOR_GEOM_RHO;
    if (uflr_geom > P(prims::u, k, j, i)) fflag |= HIT_FLOOR_GEOM_U;


    // 2. Magnetization ceilings: impose maximum magnetization sigma = bsq/rho, and inverse beta prop. to bsq/U
    FourVectors Dtmp;
    get_state(G, P, k, j, i, Loci::center, Dtmp); // Recall this gets re-used below
    double bsq = bsq_calc(Dtmp);
    double rhoflr_b = bsq/BSQORHOMAX;
    double uflr_b = bsq/BSQOUMAX;

    // Record Magnetic floor hits
    if (rhoflr_b > P(prims::rho, k, j, i)) fflag |= HIT_FLOOR_B_RHO;
    if (uflr_b > P(prims::u, k, j, i)) fflag |= HIT_FLOOR_B_U;

    // Evaluate highest U floor
    double uflr_max = max(uflr_geom, uflr_b);

    // 3. Temperature ceiling: impose maximum temperature u/rho
    // Take floors on U into account
    double rhoflr_temp = max(P(prims::u, k, j, i) / UORHOMAX, uflr_max / UORHOMAX);

    // Record hitting temperature ceiling
    if (rhoflr_temp > P(prims::rho, k, j, i)) fflag |= HIT_FLOOR_TEMP; // Misnomer for consistency

    // Evaluate highest RHO floor
    double rhoflr_max = max(max(rhoflr_geom, rhoflr_b), rhoflr_temp);

    if (rhoflr_max > P(prims::rho, k, j, i) || uflr_max > P(prims::u, k, j, i)) { // Apply floors

        // Initialize a dummy fluid parcel
        Real Pnew[NPRIM] = {0}, Unew[NPRIM] = {0};
        FourVectors Dnew = {0};

        // Add mass and internal energy, but not velocity
        Pnew[prims::rho] = max(0., rhoflr_max - P(prims::rho, k, j, i));
        Pnew[prims::u] = max(0., uflr_max - P(prims::u, k, j, i));

        // Get conserved variables for the new parcel
        get_state(G, Pnew, k, j, i, Loci::center, Dnew);
        prim_to_flux(G, Pnew, Dnew, eos, k, j, i, Loci::center, 0, Unew);

        // And for the current state, by re-using Dtmp from above
        prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);

        // Add new conserved variables to current values
        PLOOP {
            U(p, k, j, i) += Unew[p];
            P(p, k, j, i) += Pnew[p];
        }

        // Recover primitive variables
        // TODO record these and print, because it would suck to call this
        // function from inside fix_U_to_P and *still* get an inversion error
        // pflag(k, j, i) =
        U_to_P(G, U, eos, k, j, i, Loci::center, P);
    }
    return fflag;
}

/**
 * Smooth over inversion failures by averaging points from neighbors
 * a.k.a. Diffusion?  What diffusion?  There is no diffusion here.
 *
 * TODO can we parallelize this function without losing determinism?
 */
KOKKOS_INLINE_FUNCTION int fix_U_to_P(const Grid& G, GridVars P, GridVars U, EOS *eos, GridInt pflag, const int& k, const int& j, const int& i)
{
    // TODO Make sure we are not using/in the ill defined physical corner regions
    // Requires asking Parthenon where we are on the global mesh
    // May be unnecessary...

    int fflag = 0;

    if (pflag(k, j, i) != InversionStatus::success) {
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
        FLOOP P(p, k, j, i) = sum[p]/wsum;

        // Make sure fixed values still abide by floors
        // fflag |= fixup_ceiling(G, P, k, j, i);
        // fflag |= fixup_floor(G, P, U, eos, k, j, i);
    }

    return fflag;
}
