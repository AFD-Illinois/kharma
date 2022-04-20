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
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
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
 * This function definitely applies floors (regardless of "disable_floors")
 * to the interior domain (not ghost zones).
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
TaskStatus ApplyFloors(MeshBlockData<Real> *rc);

/**
 * Parthenon call wrapper for ApplyFloors, called just after FillDerived == UtoP
 * Decides whether to apply floors based on options, then does so
 */
TaskStatus PostFillDerivedBlock(MeshBlockData<Real> *rc);

enum FloorFrame{normal_observer, fluid, mixed_nof_ff};

/**
 * Struct to hold floor values without cumbersome dictionary/string logistics.
 * Faster than consulting the full Params object on the device side,
 * cleaner than adding a million new floats
 * This idea is also used in VarMaps and EMHD_parameters structs.
 * Constructed once in Floors::Initialize
 */
struct Prescription {
    // Purely geometric limits
    Real rho_min_geom, u_min_geom;
    // Dynamic limits on magnetization/temperature
    Real bsq_over_rho_max, bsq_over_u_max, u_over_rho_max;
    // Limit entropy
    Real ktot_max;
    // Limit fluid Lorentz factor
    Real gamma_max;
    // Floor options
    // Frame to apply floors. Radius at which to switch if using mixed frame
    FloorFrame frame; GReal mixed_frame_switch;

    bool use_r_char; GReal r_char;
    // Whether to adjust internal energy when limiting temperature (vs density)
    bool temp_adjust_u;
    // Whether to adjust entropy when applying density floor
    bool adjust_k;
    // Whether to apply spherical (radially decreasing) geometric floors
    bool spherical;
};

/**
 * Apply all ceilings together, currently at most one on velocity and two on internal energy
 * 
 * @return fflag, a bitflag indicating whether each particular floor was hit, allowing representation of arbitrary combinations
 * See decs.h for bit names.
 */
KOKKOS_FORCEINLINE_FUNCTION int apply_ceilings(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                          const Real& gam, const int& k, const int& j, const int& i, const Floors::Prescription& floors,
                                          const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
    int fflag = 0;
    // First apply ceilings:
    // 1. Limit gamma with respect to normal observer
    const Real gamma = GRMHD::lorentz_calc(G, P, m_p, k, j, i, loc);

    if (gamma > floors.gamma_max) {
        fflag |= HIT_FLOOR_GAMMA;

        VLOOP P(m_p.U1+v, k, j, i) *= sqrt((pow(floors.gamma_max, 2) - 1.)/(pow(gamma, 2) - 1.));
    }

    // 2. Limit the entropy by controlling u, to avoid anomalous cooling from funnel wall
    // Note this technically applies the condition *one step sooner* than legacy, since it operates on
    // the entropy as calculated from current conditions, rather than the value kept from the previous
    // step for calculating dissipation.
    // TODO can we avoid this when the floor is disabled?
    const Real ktot = (gam - 1.) * P(m_p.UU, k, j, i) / pow(P(m_p.RHO, k, j, i), gam);
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

    return fflag;
}

/**
 * Apply floors of several types in determining how to add mass and internal energy to preserve stability.
 * All floors which might apply are recorded separately, then mass/energy are added *in normal observer frame*
 * 
 * @return fflag + pflag: fflag is a bitflag starting at the sixth bit from the right.  pflag is a number <32.
 * This returns the sum, with the caller responsible for separating what's desired.
 */
KOKKOS_FORCEINLINE_FUNCTION int apply_floors(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                        const Real& gam, const int& k, const int& j, const int& i, const Floors::Prescription& floors,
                                        const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
    int fflag = 0;
    // Then apply floors:
    // 1. Geometric hard floors, not based on fluid relationships
    Real rhoflr_geom, uflr_geom;
    bool use_ff;
    if(floors.spherical) {
        // GReal Xembed[GR_DIM];
        // G.coord_embed(k, j, i, loc, Xembed);
        // GReal r = Xembed[1];
        // This is faster for now, working on it
        const GReal r = exp(G.x1v(i));

        // Whether to use fluid frame for *this zone*
        use_ff = floors.frame == FloorFrame::fluid ||
                 (floors.frame == FloorFrame::mixed_nof_ff && r > floors.mixed_frame_switch);

        if (floors.use_r_char) {
            // Steeper floor from iharm3d
            GReal rhoscal = pow(r, -2.) * 1 / (1 + r / floors.r_char);
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
        use_ff = (floors.frame == FloorFrame::fluid);
    }
    // These must not be copies, as we'll use them to adjust ktot later
    const Real rho = P(m_p.RHO, k, j, i);
    const Real u = P(m_p.UU, k, j, i);

    // 2. Magnetization ceilings: impose maximum magnetization sigma = bsq/rho, and inverse beta prop. to bsq/U
    FourVectors Dtmp;
    // TODO is there a more efficient way to calculate just bsq?
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, loc, Dtmp);
    const double bsq = dot(Dtmp.bcon, Dtmp.bcov);
    const Real rhoflr_b = bsq / floors.bsq_over_rho_max;
    const Real uflr_b = bsq / floors.bsq_over_u_max;

    // Evaluate max U floor, needed for temp ceiling below
    const Real uflr_max = max(uflr_geom, uflr_b);

    Real rhoflr_max = max(rhoflr_geom, rhoflr_b);
    if (!floors.temp_adjust_u) {
        // 3. Temperature ceiling: impose maximum temperature u/rho
        // Take floors on U into account
        const Real rhoflr_temp = max(u, uflr_max) / floors.u_over_rho_max;
        // Record hitting temperature ceiling
        fflag |= (rhoflr_temp > rho) * HIT_FLOOR_TEMP; // Misnomer for consistency

        // Evaluate max rho floor
        rhoflr_max = max(rhoflr_max, rhoflr_temp);
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

        // Apply floors to primitive values
        const Real rho_add = max(0., rhoflr_max - rho);
        const Real u_add = max(0., uflr_max - u);
        P(m_p.RHO, k, j, i) += rho_add;
        P(m_p.UU, k, j, i) += u_add;
        // In fluid frame, that's all we needed. In other frames...
        if (!use_ff) {
            // Add the material in the normal observer frame, by
            // calculating the conserved variables corresponding to the new material
            // in NOF...
            const Real uvec[NVEC] = {0}, B[NVEC] = {0};
            Real rho_ut, T[GR_DIM];
            GRMHD::p_to_u_mhd(G, rho_add, u_add, uvec, B, gam, k, j, i, rho_ut, T, loc);

            // ...and adding these to the conserved state
            U(m_u.RHO, k, j, i) += rho_ut;
            U(m_u.UU, k, j, i) += T[0]; // Note this shouldn't be a single loop: m_u.U1 != m_u.UU + 1 necessarily
            U(m_u.U1, k, j, i) += T[1];
            U(m_u.U2, k, j, i) += T[2];
            U(m_u.U3, k, j, i) += T[3];
            
            // Recover primitive variables from conserved versions
            fflag += GRMHD::u_to_p(G, U, m_u, gam, k, j, i, loc, P, m_p);
            // If this fails, we've already effectively applied the floors in fluid frame
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

    // Return floor bitflag (inversion status has been added if nonzero)
    return fflag;
}

/**
 * Apply just the geometric floors to a set of local primitives.  Must be applied in fluid frame!
 * Specifically called after reconstruction when using non-TVD schemes, e.g. WENO5.
 * Reimplemented so as to be 
 * 
 * @return fflag: since no inversion is performed, this just returns a flag representing which geometric floors were hit
 * 
 * NOT LOCKSTEP: Operates on and respects primitives *only*
 */
template<typename Local>
KOKKOS_FORCEINLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, Local& P, const VarMap& m,
                                            const Real& gam, const int& j, const int& i,
                                            const Floors::Prescription& floors, const Loci loc=Loci::center)
{
    // Apply only the geometric floors, in fluid frame
    Real rhoflr_geom, uflr_geom;
    if(floors.spherical) {
        // GReal Xembed[GR_DIM];
        // G.coord_embed(0, j, i, loc, Xembed);
        // GReal r = Xembed[1];
        // This is faster for now, working on it
        GReal r = exp(G.x1v(i));

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
    fflag |= (rhoflr_geom > P(m.RHO)) * HIT_FLOOR_GEOM_RHO_FLUX;
    fflag |= (uflr_geom > P(m.UU)) * HIT_FLOOR_GEOM_U_FLUX;
#endif

    P(m.RHO) += max(0., rhoflr_geom - P(m.RHO));
    P(m.UU) += max(0., uflr_geom - P(m.UU));

    return fflag;
}

template<typename Global>
KOKKOS_FORCEINLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, Global& P, const VarMap& m,
                                            const Real& gam, const int& k, const int& j, const int& i,
                                            const Floors::Prescription& floors, const Loci loc=Loci::center)
{
    // Apply only the geometric floors, in fluid frame
    Real rhoflr_geom, uflr_geom;
    if(floors.spherical) {
        // GReal Xembed[GR_DIM];
        // G.coord_embed(k, j, i, loc, Xembed);
        // GReal r = Xembed[1];
        // This is faster for now, working on it
        GReal r = exp(G.x1v(i));

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
    fflag |= (rhoflr_geom > P(m.RHO, k, j, i)) * HIT_FLOOR_GEOM_RHO_FLUX;
    fflag |= (uflr_geom > P(m.UU, k, j, i)) * HIT_FLOOR_GEOM_U_FLUX;
#endif

    P(m.RHO, k, j, i) += max(0., rhoflr_geom - P(m.RHO, k, j, i));
    P(m.UU, k, j, i) += max(0., uflr_geom - P(m.UU, k, j, i));

    return fflag;
}

} // namespace Floors
