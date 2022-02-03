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
#include "U_to_P.hpp"

#include <parthenon/parthenon.hpp>

// Return which floors are hit post-reconstruction
// Currently not recorded by the caller, so disabled
#define RECORD_POST_RECON 0

// Floor codes are non-exclusive, so it makes little sense to use an enum
// Instead, we use bitflags, starting high enough that we can stick the enum in the bottom 5 bits
// See floors.hpp for explanations of the flags
#define HIT_FLOOR_GEOM_RHO 32
#define HIT_FLOOR_GEOM_U 64
#define HIT_FLOOR_B_RHO 128
#define HIT_FLOOR_B_U 256
#define HIT_FLOOR_TEMP 512
#define HIT_FLOOR_GAMMA 1024
#define HIT_FLOOR_KTOT 2048
// Separate flags for when the floors are applied after reconstruction.
// Not yet used, as this will likely have some speed penalty paid even if
// the flags aren't written
#define HIT_FLOOR_GEOM_RHO_FLUX 4096
#define HIT_FLOOR_GEOM_U_FLUX 8192

namespace Floors
{

/**
 * Initialization.  Set parameters.
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

/**
 * Apply density and internal energy floors and ceilings
 * 
 * This function is called just after UtoP finishes, and
 * applies to the same subset of zones (anything "on" the grid,
 * i.e. not past a polar or outflow boundary)
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
TaskStatus ApplyFloors(MeshBlockData<Real> *rc);

/**
 * Parthenon wrapper for ApplyFloors.  Decides whether to apply floors, then does so
 */
TaskStatus PostFillDerivedBlock(MeshBlockData<Real> *rc);

/**
 * Struct to hold floor values without cumbersome dictionary/string logistics.
 * Hopefully faster than dragging the full Params object device side,
 * similar reasoning to VarMap above.
 */
class Prescription {
    public:
        // Purely geometric limits
        double rho_min_geom, u_min_geom, r_char, frame_switch;
        // Dynamic limits on magnetization/temperature
        double bsq_over_rho_max, bsq_over_u_max, u_over_rho_max;
        // Limit entropy
        double ktot_max;
        // Limit fluid Lorentz factor
        double gamma_max;
        // Floor options
        bool fluid_frame, mixed_frame;
        bool use_r_char, temp_adjust_u, adjust_k;

        Prescription(const parthenon::Params& params)
        {
            rho_min_geom = params.Get<Real>("rho_min_geom");
            u_min_geom = params.Get<Real>("u_min_geom");
            r_char = params.Get<GReal>("r_char");
            frame_switch = params.Get<GReal>("frame_switch");

            bsq_over_rho_max = params.Get<Real>("bsq_over_rho_max");
            bsq_over_u_max = params.Get<Real>("bsq_over_u_max");
            u_over_rho_max = params.Get<Real>("u_over_rho_max");
            ktot_max = params.Get<Real>("ktot_max");
            gamma_max = params.Get<Real>("gamma_max");

            fluid_frame = params.Get<bool>("fluid_frame");
            mixed_frame = params.Get<bool>("mixed_frame");
            use_r_char = params.Get<bool>("use_r_char");
            temp_adjust_u = params.Get<bool>("temp_adjust_u");
            adjust_k = params.Get<bool>("adjust_k");
        }
};

/**
 * Apply all ceilings together, currently at most one on velocity and two on internal energy
 * 
 * @return fflag, a bitflag indicating whether each particular floor was hit, allowing representation of arbitrary combinations
 * See decs.h for bit names.
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
KOKKOS_INLINE_FUNCTION int apply_ceilings(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                          const Real& gam, const int& k, const int& j, const int& i, const Floors::Prescription& floors,
                                          const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
    int fflag = 0;
    // First apply ceilings:
    // 1. Limit gamma with respect to normal observer
    Real gamma = GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc);

    if (gamma > floors.gamma_max) {
        fflag |= HIT_FLOOR_GAMMA;

        Real f = sqrt((pow(floors.gamma_max, 2) - 1.)/(pow(gamma, 2) - 1.));
        VLOOP P(m_p.U1+v, k, j, i) *= f;
    }

    // 2. Limit the entropy by controlling u, to avoid anomalous cooling from funnel wall
    // Note this technically applies the condition *one step sooner* than legacy, since it operates on
    // the entropy as calculated from current conditions, rather than the value kept from the previous
    // step for calculating dissipation.
    Real ktot = (gam - 1.) * P(m_p.UU, k, j, i) / pow(P(m_p.RHO, k, j, i), gam);
    if (ktot > floors.ktot_max) {
        fflag |= HIT_FLOOR_KTOT;

        P(m_p.UU, k, j, i) = floors.ktot_max / ktot * P(m_p.UU, k, j, i);
    }
    // Also apply the ceiling to the advected entropy KTOT, if we're keeping track of that
    // (either for electrons, or robust primitive inversions in future)
    // TODO make a separate flag for hitting this vs the "fake" version above
    if (m_p.KTOT >= 0 && (P(m_p.KTOT, k, j, i) > floors.ktot_max)) {
        fflag |= HIT_FLOOR_KTOT;
        P(m_p.KTOT, k, j, i) = floors.ktot_max;
    }

    // 3. Limit the temperature by controlling u.  Can optionally add density instead, implemented in apply_floors
    if (floors.temp_adjust_u && P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i) > floors.u_over_rho_max) {
        fflag |= HIT_FLOOR_TEMP;

        P(m_p.UU, k, j, i) = floors.u_over_rho_max * P(m_p.RHO, k, j, i);
    }

    if (fflag) {
        // Keep lockstep!
        GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u, loc);
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
KOKKOS_INLINE_FUNCTION int apply_floors(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                        const Real& gam, const int& k, const int& j, const int& i, const Floors::Prescription& floors,
                                        const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
    int fflag = 0;
    InversionStatus pflag = InversionStatus::success;
    // Then apply floors:
    // 1. Geometric hard floors, not based on fluid relationships
    Real rhoflr_geom, uflr_geom;
    bool use_ff;
    if(G.coords.spherical()) {
        GReal Xembed[GR_DIM];
        G.coord_embed(k, j, i, loc, Xembed);
        GReal r = Xembed[1];
        // TODO measure whether this/if 1 is really faster
        // GReal r = exp(G.x1v(i));

        // Use the fluid frame if specified, or in outer domain
        use_ff = floors.fluid_frame || (floors.mixed_frame && r > floors.frame_switch);

        if (floors.use_r_char) {
            // Steeper floor from iharm3d
            Real rhoscal = pow(r, -2.) * 1 / (1 + r / floors.r_char);
            rhoflr_geom = floors.rho_min_geom * rhoscal;
            uflr_geom = floors.u_min_geom * pow(rhoscal, gam);
        } else {
            // Original floors from iharm2d
            rhoflr_geom = floors.rho_min_geom * pow(r, -1.5);
            uflr_geom = floors.u_min_geom * pow(r, -2.5); //rhoscal/r as in iharm2d
        }
    } else {
        rhoflr_geom = floors.rho_min_geom;
        uflr_geom = floors.u_min_geom;
        use_ff = floors.fluid_frame;
    }
    Real rho = P(m_p.RHO, k, j, i);
    Real u = P(m_p.UU, k, j, i);

    // 2. Magnetization ceilings: impose maximum magnetization sigma = bsq/rho, and inverse beta prop. to bsq/U
    FourVectors Dtmp;
    // TODO is there a more efficient way to calculate just bsq?
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, loc, Dtmp);
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

    // If we need to do anything...
    if (rhoflr_max > rho || uflr_max > u) {

        // Record all the floors that were hit, using bitflags
        // Record Geometric floor hits
        fflag |= (rhoflr_geom > rho) * HIT_FLOOR_GEOM_RHO;
        fflag |= (uflr_geom > u) * HIT_FLOOR_GEOM_U;
        // Record Magnetic floor hits
        fflag |= (rhoflr_b > rho) * HIT_FLOOR_B_RHO;
        fflag |= (uflr_b > u) * HIT_FLOOR_B_U;

        if (use_ff) {
            P(m_p.RHO, k, j, i) += max(0., rhoflr_max - rho);
            P(m_p.UU, k, j, i) += max(0., uflr_max - u);
            GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u, loc);
        } else {
            // Add the material in the normal observer frame, by:
            // Adding the floors to the primitive variables
            const Real rho_add = max(0., rhoflr_max - rho);
            const Real u_add = max(0., uflr_max - u);
            const Real uvec[NVEC] = {0}, B[NVEC] = {0};

            // Calculating the corresponding conserved variables
            Real rho_ut, T[GR_DIM];
            GRMHD::p_to_u_loc(G, rho_add, u_add, uvec, B, gam, k, j, i, rho_ut, T, loc);

            // Add new conserved mass/energy to the current "conserved" state,
            // and to the local primitives as a guess
            P(m_p.RHO, k, j, i) += rho_add;
            P(m_p.UU, k, j, i) += u_add;
            // Add any velocity here
            U(m_u.RHO, k, j, i) += rho_ut;
            U(m_u.UU, k, j, i) += T[0]; // Note this shouldn't be a single loop: m_u.U1 != m_u.UU + 1 necessarily
            U(m_u.U1, k, j, i) += T[1];
            U(m_u.U2, k, j, i) += T[2];
            U(m_u.U3, k, j, i) += T[3];
            
            // Recover primitive variables from conserved versions
            pflag = GRMHD::u_to_p(G, U, m_u, gam, k, j, i, loc, P, m_p);
            // If that fails, we've effectively already applied the floors in fluid-frame to the prims,
            // so we just formalize that
            if (pflag) {
                GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u, loc);
            }
        }
    }

    // Ressler adjusts KTOT & KEL to conserve u whenever adjusting rho
    // but does *not* recommend adjusting them when u hits floors/ceilings
    // This is in contrast to ebhlight, which heats electrons before applying *any* floors,
    // and resets KTOT during floor application without touching KEL
    // TODO move to another loop/function, over electrons.  Have to preserve rho/rho_old ratio tho
    if (floors.adjust_k && (fflag & HIT_FLOOR_GEOM_RHO || fflag & HIT_FLOOR_B_RHO)) {
        const Real reduce   = pow(rho / P(m_p.RHO, k, j, i), gam);
        const Real reduce_e = pow(rho / P(m_p.RHO, k, j, i), 4./3); // TODO pipe in real gam_e
        if (m_p.KTOT >= 0) P(m_p.KTOT, k, j, i) *= reduce;
        if (m_p.K_CONSTANT >= 0) P(m_p.K_CONSTANT, k, j, i) *= reduce_e;
        if (m_p.K_HOWES >= 0)    P(m_p.K_HOWES, k, j, i)    *= reduce_e;
        if (m_p.K_KAWAZURA >= 0) P(m_p.K_KAWAZURA, k, j, i) *= reduce_e;
        if (m_p.K_WERNER >= 0)   P(m_p.K_WERNER, k, j, i)   *= reduce_e;
        if (m_p.K_ROWAN >= 0)    P(m_p.K_ROWAN, k, j, i)    *= reduce_e;
        if (m_p.K_SHARMA >= 0)   P(m_p.K_SHARMA, k, j, i)   *= reduce_e;
    }

    // Return both flags
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
KOKKOS_INLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, ScratchPad2D<Real>& P, const VarMap& m,
                                            const Real& gam, const int& k, const int& j, const int& i,
                                            const Floors::Prescription& floors, const Loci loc=Loci::center)
{
    // Apply only the geometric floors
    Real rhoflr_geom, uflr_geom;
    if(G.coords.spherical()) {
        GReal Xembed[GR_DIM];
        G.coord_embed(k, j, i, loc, Xembed);
        GReal r = Xembed[1];

        if (floors.use_r_char) {
            // Steeper floor from iharm3d
            Real rhoscal = pow(r, -2.) * 1 / (1 + r / floors.r_char);
            rhoflr_geom = floors.rho_min_geom * rhoscal;
            uflr_geom = floors.u_min_geom * pow(rhoscal, gam);
        } else {
            // Original floors from iharm2d
            rhoflr_geom = floors.rho_min_geom * pow(r, -1.5);
            uflr_geom = floors.u_min_geom * pow(r, -2.5); //rhoscal/r as in iharm2d
        }
    } else {
        rhoflr_geom = floors.rho_min_geom;
        uflr_geom = floors.u_min_geom;
    }

    int fflag = 0;
#if RECORD_POST_RECON
    // Record all the floors that were hit, using bitflags
    // Record Geometric floor hits
    fflag |= (rhoflr_geom > P(m.RHO, i)) * HIT_FLOOR_GEOM_RHO_FLUX;
    fflag |= (uflr_geom > P(m.UU, i)) * HIT_FLOOR_GEOM_U_FLUX;
#endif

    P(m.RHO, i) += max(0., rhoflr_geom - P(m.RHO, i));
    P(m.UU, i) += max(0., uflr_geom - P(m.UU, i));

    return fflag;
}

} // namespace Floors
