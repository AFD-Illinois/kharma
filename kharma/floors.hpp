// Fixups.  Apply limits and fix bad fluid values to maintain integrable state
// ApplyFloors, FixUtoP
#pragma once

#include "decs.hpp"
#include "eos.hpp"
#include "phys.hpp"
#include "U_to_P.hpp"

#include <parthenon/parthenon.hpp>

/**
 * Apply density and internal energy floors and ceilings
 */
TaskStatus ApplyFloors(std::shared_ptr<Container<Real>>& rc);

/**
 * Apply a fluid velocity ceiling
 * 
 * @return fflag, a bitflag indicating whether each particular floor was hit, allowing representation of arbitrary combinations
 * See decs.h for bit names.
 * 
 * LOCKSTEP: this function should preserve P<->U.  It does not require it on entry
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
 * @return fflag + pflag: fflag is a flagset starting at the sixth bit from the right.  pflag is a number <32.
 * This returns the sum, with the caller responsible for separating what's desired.
 * 
 * LOCKSTEP: this function should preserve P<->U. It does not require it on entry
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
    } else {
        rhoflr_geom = RHOMIN*1.e-2;
        uflr_geom = UUMIN*1.e-2;
    }
    Real rho = P(prims::rho, k, j, i);
    Real u = P(prims::u, k, j, i);

    // 2. Magnetization ceilings: impose maximum magnetization sigma = bsq/rho, and inverse beta prop. to bsq/U
    FourVectors Dtmp;
    get_state(G, P, k, j, i, Loci::center, Dtmp); // Recall this gets re-used below
    double bsq = bsq_calc(Dtmp);
    double rhoflr_b = bsq/BSQORHOMAX;
    double uflr_b = bsq/BSQOUMAX;

    // Evaluate max U floor, needed for temp ceiling below
    double uflr_max = max(uflr_geom, uflr_b);

    // 3. Temperature ceiling: impose maximum temperature u/rho
    // Take floors on U into account
    double rhoflr_temp = max(u / UORHOMAX, uflr_max / UORHOMAX);

    // Evaluate max rho floor
    double rhoflr_max = max(max(rhoflr_geom, rhoflr_b), rhoflr_temp);

#if DEBUG
    // Record all the floors that were hit, even if there were multiple
    // Record Geometric floor hits
    fflag |= (rhoflr_geom > rho) * HIT_FLOOR_GEOM_RHO;
    fflag |= (uflr_geom > u) * HIT_FLOOR_GEOM_U;
    // Record Magnetic floor hits
    fflag |= (rhoflr_b > rho) * HIT_FLOOR_B_RHO;
    fflag |= (uflr_b > u) * HIT_FLOOR_B_U;
    // Record hitting temperature ceiling
    fflag |= (rhoflr_temp > rho) * HIT_FLOOR_TEMP; // Misnomer for consistency
#endif

    InversionStatus pflag = InversionStatus::success;
    if (rhoflr_max > rho || uflr_max > u) { // Apply floors

        // Add the material in the normal observer frame, by:
        // Initializing a dummy fluid parcel
        Real Pnew[NPRIM] = {0}, Unew[NPRIM] = {0};
        FourVectors Dnew;

        // Add mass and internal energy to the primitives, but not velocity
        Pnew[prims::rho] = max(0., rhoflr_max - rho);
        Pnew[prims::u] = max(0., uflr_max - u);

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
        pflag = U_to_P(G, U, eos, k, j, i, Loci::center, P);
    }
    return fflag + pflag;
}
