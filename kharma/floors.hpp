/* 
 *  File: floors.cpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *  
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include "decs.hpp"


#include "b_flux_ct.hpp"
#include "mhd_functions.hpp"
#include "phys_functions.hpp"
#include "U_to_P.hpp"

#include <parthenon/parthenon.hpp>

// Return which floors are hit post-reconstruction
// Currently not used by caller, so disabled
#define RECORD_POST_RECON 0

/**
 * Struct to hold floor values without cumbersome dictionary/string logistics.
 * Hopefully faster.
 */
class FloorPrescription {
    public:
        // Purely geometric limits
        double rho_min_geom, u_min_geom, r_char;
        // Dynamic limits on magnetization/temperature
        double bsq_over_rho_max, bsq_over_u_max, u_over_rho_max;
        // Limit entropy -- currently only for legacy but potentially useful
        double ktot_max;
        // Limit fluid Lorentz factor
        double gamma_max;
        // Floor options
        bool temp_adjust_u, fluid_frame;

        FloorPrescription(parthenon::Params& params)
        {
            rho_min_geom = params.Get<Real>("rho_min_geom");
            u_min_geom = params.Get<Real>("u_min_geom");
            r_char = params.Get<Real>("floor_r_char");

            bsq_over_rho_max = params.Get<Real>("bsq_over_rho_max");
            bsq_over_u_max = params.Get<Real>("bsq_over_u_max");
            u_over_rho_max = params.Get<Real>("u_over_rho_max");
            ktot_max = params.Get<Real>("ktot_max");
            gamma_max = params.Get<Real>("gamma_max");

            temp_adjust_u = params.Get<bool>("temp_adjust_u");
            fluid_frame = params.Get<bool>("fluid_frame");
        }
};

/**
 * Apply density and internal energy floors and ceilings
 */
TaskStatus ApplyFloors(MeshBlockData<Real> *rc);

/**
 * Apply all ceilings, together, currently at most one on velocity and two on internal energy
 * 
 * @return fflag, a bitflag indicating whether each particular floor was hit, allowing representation of arbitrary combinations
 * See decs.h for bit names.
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
KOKKOS_INLINE_FUNCTION int apply_ceilings(const GRCoordinates& G, GridVars P, GridVector B_P, GridVars U, const Real& gam, const int& k, const int& j, const int& i,
                                          const FloorPrescription& floors, const Loci loc=Loci::center)
{
    int fflag = 0;
    // First apply ceilings:
    // 1. Limit gamma with respect to normal observer
    Real gamma = lorentz_calc(G, P, k, j, i, loc);

    if (gamma > floors.gamma_max) {
        fflag |= HIT_FLOOR_GAMMA;

        Real f = sqrt((pow(floors.gamma_max, 2) - 1.)/(pow(gamma, 2) - 1.));
        P(prims::u1, k, j, i) *= f;
        P(prims::u2, k, j, i) *= f;
        P(prims::u3, k, j, i) *= f;
    }

    // 2. Limit the entropy by controlling u, to avoid anomalous cooling from funnel wall
    // Pretty much only for matching legacy runs
    // Note this technically applies the condition *one step sooner* than legacy, since it operates on
    // the entropy as calculated from current conditions, rather than the value kept from the previous
    // step for calculating dissipation.
    Real ktot = (gam - 1.) * P(prims::u, k, j, i) / pow(P(prims::rho, k, j, i), gam);
    if (ktot > floors.ktot_max) {
        fflag |= HIT_FLOOR_KTOT;

        P(prims::u, k, j, i) = floors.ktot_max / ktot * P(prims::u, k, j, i);
    }

    // 3. Limit the temperature by controlling u.  Can optionally add density instead, implemented in apply_floors
    if (floors.temp_adjust_u && P(prims::u, k, j, i) / P(prims::rho, k, j, i) > floors.u_over_rho_max) {
        fflag |= HIT_FLOOR_TEMP;

        P(prims::u, k, j, i) = floors.u_over_rho_max * P(prims::rho, k, j, i);
    }

    if (fflag) {
        // Keep lockstep!
        GRMHD::p_to_u(G, P, B_P, gam, k, j, i, U, loc);
    }

    return fflag;
}

/**
 * Apply floors of several types in determining how to add mass and internal energy to preserve stability.
 * All floors which might apply are recorded separately, then mass/energy are added *in normal observer frame*
 * 
 * @return fflag + pflag: fflag is a flagset starting at the sixth bit from the right.  pflag is a number <32.
 * This returns the sum, with the caller responsible for separating what's desired.
 * 
 * LOCKSTEP: this function respects P and ignores U in order to return consistent P<->U
 */
KOKKOS_INLINE_FUNCTION int apply_floors(const GRCoordinates& G, GridVars P, GridVector B_P, GridVars U, GridVector B_U,
                                        const Real& gam, const int& k, const int& j, const int& i,
                                        const FloorPrescription& floors, const Loci loc=Loci::center)
{
    int fflag = 0;
    // Then apply floors:
    // 1. Geometric hard floors, not based on fluid relationships
    Real rhoflr_geom, uflr_geom;
    if(G.coords.spherical()) {
        GReal Xembed[GR_DIM];
        G.coord_embed(k, j, i, loc, Xembed);
        GReal r = Xembed[1];

        // New, steeper floor in rho
        // Previously raw r^-2, r^-1.5
        Real rhoscal = pow(r, -2.) * 1 / (1 + r / floors.r_char);
        rhoflr_geom = floors.rho_min_geom * rhoscal;
        uflr_geom = floors.u_min_geom * pow(rhoscal, gam);
    } else {
        rhoflr_geom = floors.rho_min_geom;
        uflr_geom = floors.u_min_geom;
    }
    Real rho = P(prims::rho, k, j, i);
    Real u = P(prims::u, k, j, i);

    // 2. Magnetization ceilings: impose maximum magnetization sigma = bsq/rho, and inverse beta prop. to bsq/U
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, B_P, k, j, i, loc, Dtmp);
    double bsq = dot(Dtmp.bcon, Dtmp.bcov);
    double rhoflr_b = bsq / floors.bsq_over_rho_max;
    double uflr_b = bsq / floors.bsq_over_u_max;

    // Evaluate max U floor, needed for temp ceiling below
    double uflr_max = max(uflr_geom, uflr_b);

    double rhoflr_max;
    if (!floors.temp_adjust_u) {
        // 3. Temperature ceiling: impose maximum temperature u/rho
        // Take floors on U into account
        double rhoflr_temp = max(u, uflr_max) / floors.u_over_rho_max;
        // Record hitting temperature ceiling
        fflag |= (rhoflr_temp > rho) * HIT_FLOOR_TEMP; // Misnomer for consistency

        // Evaluate max rho floor
        rhoflr_max = max(max(rhoflr_geom, rhoflr_b), rhoflr_temp);
    } else {
        // Evaluate max rho floor
        rhoflr_max = max(rhoflr_geom, rhoflr_b);
    }

    // Record all the floors that were hit, using bitflags
    // Record Geometric floor hits
    fflag |= (rhoflr_geom > rho) * HIT_FLOOR_GEOM_RHO;
    fflag |= (uflr_geom > u) * HIT_FLOOR_GEOM_U;
    // Record Magnetic floor hits
    fflag |= (rhoflr_b > rho) * HIT_FLOOR_B_RHO;
    fflag |= (uflr_b > u) * HIT_FLOOR_B_U;

    InversionStatus pflag = InversionStatus::success;
    if (floors.fluid_frame) {
        P(prims::rho, k, j, i) += max(0., rhoflr_max - rho);
        P(prims::u, k, j, i) += max(0., uflr_max - u);
        GRMHD::p_to_u(G, P, B_P, gam, k, j, i, U, loc);
    } else {
        if (rhoflr_max > rho || uflr_max > u) { // Apply floors

            // Add the material in the normal observer frame, by:
            // Initializing a dummy fluid parcel
            Real Pnew[NPRIM] = {0}, Unew[NPRIM] = {0}, B_Pnew[NVEC] = {0};

            // Add mass and internal energy to the primitives, but not velocity
            Pnew[prims::rho] = max(0., rhoflr_max - rho);
            Pnew[prims::u] = max(0., uflr_max - u);

            // Get conserved variables for the new and old parcels
            //GRMHD::p_to_u(G, P, B_P, gam, k, j, i, U, loc);
            GRMHD::p_to_u(G, Pnew, B_Pnew, gam, k, j, i, Unew, loc);

            // Add new conserved mass/energy to the current "conserved" state
            // Prims are needed as a guess
            PLOOP {
                U(p, k, j, i) += Unew[p];
                P(p, k, j, i) += Pnew[p];
            }

            // Recover primitive variables from conserved versions
            pflag = GRMHD::u_to_p(G, U, B_U, gam, k, j, i, loc, P);
            // If that fails, we've effectively already applied the floors in fluid-frame to the prims,
            // so we just formalize that
            if (pflag) GRMHD::p_to_u(G, P, B_P, gam, k, j, i, U, loc);
        }
    }
    return fflag + pflag;
}

/**
 * Apply just the geometric floors to a set of local primitives.
 * Specifically called after reconstruction when using non-TVD schemes, e.g. WENO5.
 * Reimplemented to be fast and fit the general prim_to_flux calling convention.
 * 
 * @return fflag: since no inversion is performed, this just returns a flag representing which geometric floors were hit
 * 
 * LOCKSTEP: Operates on and respects primitives *only*
 */
KOKKOS_INLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, ScratchPad2D<Real>& P, const struct varmap& m,
                                            const Real& gam, const int& k, const int& j, const int& i,
                                            const FloorPrescription& floors, const Loci loc=Loci::center)
{
    // Apply only the geometric floors.  Th
    Real rhoflr_geom, uflr_geom;
    if(G.coords.spherical()) {
        GReal Xembed[GR_DIM];
        G.coord_embed(k, j, i, loc, Xembed);
        GReal r = Xembed[1];

        // New, steeper floor in rho
        // Previously raw r^-2, r^-1.5
        Real rhoscal = pow(r, -2.) * 1 / (1 + r / floors.r_char);
        rhoflr_geom = floors.rho_min_geom * rhoscal;
        uflr_geom = floors.u_min_geom * pow(rhoscal, gam);
    } else {
        rhoflr_geom = floors.rho_min_geom;
        uflr_geom = floors.u_min_geom;
    }

    P(m.p + prims::rho, i) += max(0., rhoflr_geom - P(m.p + prims::rho, i));
    P(m.p + prims::u, i) += max(0., uflr_geom - P(m.p + prims::u, i));

#if RECORD_POST_RECON
    // Record all the floors that were hit, using bitflags
    // Record Geometric floor hits
    int fflag = 0;
    fflag |= (rhoflr_geom > P(m.p + prims::rho, i)) * HIT_FLOOR_GEOM_RHO;
    fflag |= (uflr_geom > P(m.p + prims::u, i)) * HIT_FLOOR_GEOM_U;
    return fflag;
#else
    return 0;
#endif
}
