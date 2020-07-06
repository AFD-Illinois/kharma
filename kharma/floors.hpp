// Fixups.  Apply limits and fix bad fluid values to maintain integrable state
// ApplyFloors, FixUtoP
#pragma once

#include "decs.hpp"
#include "eos.hpp"
#include "phys.hpp"
#include "U_to_P.hpp"

#include "interface/container.hpp"

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

/**
 * Apply density and internal energy floors and ceilings
 */
TaskStatus ApplyFloors(std::shared_ptr<Container<Real>>& rc);

/**
 * Apply a fluid velocity ceiling
 * 
 * LOCKSTEP: this function expects and should preserve P<->U
 */
KOKKOS_INLINE_FUNCTION int fixup_ceiling(const GRCoordinates& G, GridVars P, GridVars U, EOS *eos, const int& k, const int& j, const int& i)
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

        FourVectors Dtmp;
        get_state(G, P, k, j, i, Loci::center, Dtmp);
        prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
    }
    return fflag;
}

/**
 * Apply floors of several types in determining how to add mass and internal energy to preserve stability.
 * All floors which might apply are recorded separately, then mass/energy are added in normal observer frame
 * 
 * LOCKSTEP: this function expects and should preserve P<->U
 */
KOKKOS_INLINE_FUNCTION int fixup_floor(const GRCoordinates& G, GridVars P, GridVars U, EOS *eos, const int& k, const int& j, const int& i)
{
    int fflag = 0;
    // Then apply floors:
    // 1. Geometric hard floors, not based on fluid relationships
    Real rhoflr_geom, uflr_geom;
    if(G.coords.spherical()) {
        GReal Xembed[GR_DIM];
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

    // Add the material in the normal observer frame, by:
    if (rhoflr_max > P(prims::rho, k, j, i) || uflr_max > P(prims::u, k, j, i)) { // Apply floors

        // Initializing a dummy fluid parcel
        Real Pnew[NPRIM] = {0}, Unew[NPRIM] = {0};
        FourVectors Dnew = {0};

        // Add mass and internal energy to the primitives, but not velocity
        Pnew[prims::rho] = max(0., rhoflr_max - P(prims::rho, k, j, i));
        Pnew[prims::u] = max(0., uflr_max - P(prims::u, k, j, i));

        // Get conserved variables for the new parcel
        get_state(G, Pnew, k, j, i, Loci::center, Dnew);
        prim_to_flux(G, Pnew, Dnew, eos, k, j, i, Loci::center, 0, Unew);

        // And for the current state, by re-using Dtmp from above
        prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);

        // Add new conserved mass/energy to the current "conserved" state
        PLOOP {
            U(p, k, j, i) += Unew[p];
            // This is just the guess at primitive values, needed for U_to_P to converge.
            P(p, k, j, i) += Pnew[p];
        }

        // Recover primitive variables from conserved versions
        // TODO record these and print, because it would suck to call this
        // function from inside fix_U_to_P and *still* get an inversion error
        InversionStatus pflag = U_to_P(G, U, eos, k, j, i, Loci::center, P);
#if DEBUG
        // Yell?  Combine the flags?
#endif
    }
    return fflag;
}
