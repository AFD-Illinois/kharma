/* 
 *  File: floors_functions.hpp
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

#include "floors.hpp"

/**
 * Device-side functions for applying GRMHD floors
 */
namespace Floors {

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
        fflag |= FFlag::GAMMA;

        Real f = m::sqrt((m::pow(floors.gamma_max, 2) - 1.)/(m::pow(gamma, 2) - 1.));
        VLOOP P(m_p.U1+v, k, j, i) *= f;
    }

    // 2. Limit the entropy by controlling u, to avoid anomalous cooling from funnel wall
    // Note this technically applies the condition *one step sooner* than legacy, since it operates on
    // the entropy as calculated from current conditions, rather than the value kept from the previous
    // step for calculating dissipation.
    Real ktot = (gam - 1.) * P(m_p.UU, k, j, i) / m::pow(P(m_p.RHO, k, j, i), gam);
    if (ktot > floors.ktot_max) {
        fflag |= FFlag::KTOT;

        P(m_p.UU, k, j, i) = floors.ktot_max / ktot * P(m_p.UU, k, j, i);
    }
    // Also apply the ceiling to the advected entropy KTOT, if we're keeping track of that
    // (either for electrons, or robust primitive inversions in future)
    // TODO TODO MOVE TO ELECTRONS PACKAGE (or Flux::p_to_u below!!)
    if (m_p.KTOT >= 0 && (P(m_p.KTOT, k, j, i) > floors.ktot_max)) {
        fflag |= FFlag::KTOT;
        P(m_p.KTOT, k, j, i) = floors.ktot_max;
    }

    // 3. Limit the temperature by controlling u.  Can optionally add density instead, implemented in apply_floors
    if (floors.temp_adjust_u && P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i) > floors.u_over_rho_max) {
        fflag |= FFlag::TEMP;

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
                                        const Real& gam, const EMHD::EMHD_parameters& emhd_params,
                                        const int& k, const int& j, const int& i, const Floors::Prescription& floors,
                                        const VariablePack<Real>& U, const VarMap& m_u, const Loci loc=Loci::center)
{
    int fflag = 0;
    // Then apply floors:
    // 1. Geometric hard floors, not based on fluid relationships
    Real rhoflr_geom, uflr_geom;
    bool use_ff, use_df;
    if(G.coords.spherical()) {
        GReal Xembed[GR_DIM];
        G.coord_embed(k, j, i, loc, Xembed);
        GReal r = Xembed[1];
        // TODO measure whether this/if 1 is really faster
        // GReal r = m::exp(G.x1v(i));

        // Use the fluid frame if specified, or in outer domain
        use_ff = floors.fluid_frame || (floors.mixed_frame && r > floors.frame_switch);
        // Use the drift frame if specified
        use_df = floors.drift_frame;

        if (floors.use_r_char) {
            // Steeper floor from iharm3d
            const Real rhoscal = 1/(r * r * (1 + r / floors.r_char));
            rhoflr_geom  = floors.rho_min_geom * rhoscal;
            uflr_geom    = floors.u_min_geom * m::pow(rhoscal, gam);
        } else {
            // Original floors from iharm2d
            rhoflr_geom = floors.rho_min_geom * m::pow(r, -1.5);
            uflr_geom   = floors.u_min_geom * m::pow(r, -2.5); //rhoscal/r as in iharm2d
        }
    } else {
        rhoflr_geom = floors.rho_min_geom;
        uflr_geom   = floors.u_min_geom;
        use_ff      = floors.fluid_frame;
        use_df      = floors.drift_frame;
    }
    Real rho = P(m_p.RHO, k, j, i);
    Real u   = P(m_p.UU, k, j, i);

    // 2. Magnetization ceilings: impose maximum magnetization sigma = bsq/rho, and inverse beta prop. to bsq/U
    FourVectors Dtmp;
    // TODO is there a more efficient way to calculate just bsq?
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, loc, Dtmp);
    double bsq      = dot(Dtmp.bcon, Dtmp.bcov);
    double rhoflr_b = bsq / floors.bsq_over_rho_max;
    double uflr_b   = bsq / floors.bsq_over_u_max;

    // Evaluate max U floor, needed for temp ceiling below
    double uflr_max = m::max(uflr_geom, uflr_b);

    double rhoflr_max;
    if (!floors.temp_adjust_u) {
        // 3. Temperature ceiling: impose maximum temperature u/rho
        // Take floors on U into account
        double rhoflr_temp = m::max(u, uflr_max) / floors.u_over_rho_max;
        // Record hitting temperature ceiling
        fflag |= (rhoflr_temp > rho) * FFlag::TEMP; // Misnomer for consistency

        // Evaluate max rho floor
        rhoflr_max = m::max(m::max(rhoflr_geom, rhoflr_b), rhoflr_temp);
    } else {
        // Evaluate max rho floor
        rhoflr_max = m::max(rhoflr_geom, rhoflr_b);
    }

    // If we need to do anything...
    if (rhoflr_max > rho || uflr_max > u) {

        // Record all the floors that were hit, using bitflags
        // Record Geometric floor hits
        fflag |= (rhoflr_geom > rho) * FFlag::GEOM_RHO;
        fflag |= (uflr_geom > u) * FFlag::GEOM_U;
        // Record Magnetic floor hits
        fflag |= (rhoflr_b > rho) * FFlag::B_RHO;
        fflag |= (uflr_b > u) * FFlag::B_U;

        if (use_ff) {
            P(m_p.RHO, k, j, i) += m::max(0., rhoflr_max - rho);
            P(m_p.UU, k, j, i)  += m::max(0., uflr_max - u);
            // TODO should be all Flux
            GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u, loc);

        } else if (use_df) {
            // Drift frame floors. Refer to Appendix B3 in https://doi.org/10.1093/mnras/stx364 (hereafter R17)
            const Real gdet     = G.gdet(Loci::center, j, i);
            const Real lapse    = 1./m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
            double beta[GR_DIM] = {0};

            beta[1] = lapse * lapse * G.gcon(Loci::center, j, i, 0, 1);
            beta[2] = lapse * lapse * G.gcon(Loci::center, j, i, 0, 2);
            beta[3] = lapse * lapse * G.gcon(Loci::center, j, i, 0, 3);

            // Fluid quantities (four velocities have been computed above)
            const Real rho   = P(m_p.RHO, k, j, i);
            const Real uu    = P(m_p.UU, k, j, i);
            const Real pg    = (gam - 1.) * uu;
            const Real w_old = m::max(rho + uu + pg, SMALL);

            // Normal observer magnetic field
            Real Bcon[GR_DIM] = {0};
            Real Bcov[GR_DIM] = {0};
            Bcon[0] = 0;
            Bcon[1] = P(m_p.B1, k, j, i);
            Bcon[2] = P(m_p.B2, k, j, i);
            Bcon[3] = P(m_p.B3, k, j, i);
            DLOOP2 Bcov[mu] += G.gcov(Loci::center, j, i, mu, nu) * Bcon[nu];
            const Real Bsq   = m::max(dot(Bcon, Bcov), SMALL);

            // Normal observer fluid momentum
            Real Qcov[GR_DIM] = {0};
            Qcov[0] = w_old * Dtmp.ucon[0] * Dtmp.ucov[0] + pg;
            Qcov[1] = w_old * Dtmp.ucon[0] * Dtmp.ucov[1];
            Qcov[2] = w_old * Dtmp.ucon[0] * Dtmp.ucov[2];
            Qcov[3] = w_old * Dtmp.ucon[0] * Dtmp.ucov[3];

            // Momentum along magnetic field lines (must be held constant)
            double QdotB = dot(Bcon, Qcov);

            // Initial parallel velocity (refer R17 Eqn B10)
            Real vpar = QdotB / (sqrt(Bsq) * w_old * pow(Dtmp.ucon[0], 2.));

            Real ucon_dr[GR_DIM] = {0};
            // t-component of drift velocity (refer R17 Eqn B13)
            ucon_dr[0] = 1. / sqrt(pow(Dtmp.ucon[0], -2.) + pow(vpar, 2.));
            // spatial components of drift velocity (refer R17 Eqn B11)
            for (int mu = 1; mu < GR_DIM; mu++) {
                ucon_dr[mu] = Dtmp.ucon[mu] * (ucon_dr[0] / Dtmp.ucon[0]) - (vpar * Bcon[mu] * ucon_dr[0] / sqrt(Bsq));
            }

            // Update rho, uu and compute new enthalpy
            P(m_p.RHO, k, j, i) = m::max(rho, rhoflr_max);
            P(m_p.UU, k, j, i)  = m::max(uu, uflr_max);
            const Real pg_new   = (gam - 1.) * P(m_p.UU, k, j, i);
            const Real w_new    = P(m_p.RHO, k, j, i) + P(m_p.UU, k, j, i) + pg_new;

            // New parallel velocity (refer R17 Eqn B14)
            const Real x = (2. * QdotB) / (sqrt(Bsq) * w_new * ucon_dr[0]);
            vpar = x / (1 + sqrt(1 + x*x)) * (1. / ucon_dr[0]);

            // New fluid four velocity (refer R17 Eqns B13 and B11)
            Dtmp.ucon[0] = 1. / sqrt(pow(ucon_dr[0], -2.) - pow(vpar, 2.));
            for (int mu = 1; mu < GR_DIM; mu++) {
                Dtmp.ucon[mu] = ucon_dr[mu] * (Dtmp.ucon[0] / ucon_dr[0]) + (vpar * Bcon[mu] * Dtmp.ucon[0] / sqrt(Bsq));
            }
            G.lower(Dtmp.ucon, Dtmp.ucov, k, j, i, Loci::center);

            // New Lorentz factor
            const Real gamma = Dtmp.ucon[0] * lapse;

            // New velocity primitives
            P(m_p.U1, k, j, i) = Dtmp.ucon[1] + (beta[1] * gamma/lapse);
            P(m_p.U2, k, j, i) = Dtmp.ucon[2] + (beta[2] * gamma/lapse);
            P(m_p.U3, k, j, i) = Dtmp.ucon[3] + (beta[3] * gamma/lapse);

            // Update the conserved variables
            Flux::p_to_u(G, P, m_p, emhd_params, gam, k, j, i, U, m_u, loc);

        } else {
            // Add the material in the normal observer frame, by:
            // Adding the floors to the primitive variables
            const Real rho_add    = m::max(0., rhoflr_max - rho);
            const Real u_add      = m::max(0., uflr_max - u);
            const Real uvec[NVEC] = {0}, B[NVEC] = {0};

            // Calculating the corresponding conserved variables
            Real rho_ut, T[GR_DIM];
            GRMHD::p_to_u_mhd(G, rho_add, u_add, uvec, B, gam, k, j, i, rho_ut, T, loc);

            // Add new conserved mass/energy to the current "conserved" state,
            // and to the local primitives as a guess
            P(m_p.RHO, k, j, i) += rho_add;
            P(m_p.UU, k, j, i)  += u_add;
            // Add any velocity here
            U(m_u.RHO, k, j, i) += rho_ut;
            U(m_u.UU, k, j, i)  += T[0]; // Note this shouldn't be a single loop: m_u.U1 != m_u.UU + 1 necessarily
            U(m_u.U1, k, j, i)  += T[1];
            U(m_u.U2, k, j, i)  += T[2];
            U(m_u.U3, k, j, i)  += T[3];
            
            // Recover primitive variables from conserved versions
            // TODO selector here when we get more
            Inverter::Status pflag = Inverter::u_to_p<Inverter::Type::onedw>(G, U, m_u, gam, k, j, i, P, m_p, loc);
            // If that fails, we've effectively already applied the floors in fluid-frame to the prims,
            // so we just formalize that
            if (Inverter::failed(pflag)) {
                Flux::p_to_u(G, P, m_p, emhd_params, gam, k, j, i, U, m_u, loc);
                fflag += static_cast<int>(pflag);
            }
        }
    }

    // TODO separate electron floors!
    // Ressler adjusts KTOT & KEL to conserve u whenever adjusting rho
    // but does *not* recommend adjusting them when u hits floors/ceilings
    // This is in contrast to ebhlight, which heats electrons before applying *any* floors,
    // and resets KTOT during floor application without touching KEL
    if (floors.adjust_k && (fflag & FFlag::GEOM_RHO || fflag & FFlag::B_RHO)) {
        const Real reduce   = m::pow(rho / P(m_p.RHO, k, j, i), gam);
        const Real reduce_e = m::pow(rho / P(m_p.RHO, k, j, i), 4./3); // TODO pipe in real gam_e
        if (m_p.KTOT >= 0) P(m_p.KTOT, k, j, i) *= reduce;
        if (m_p.K_CONSTANT >= 0) P(m_p.K_CONSTANT, k, j, i) *= reduce_e;
        if (m_p.K_HOWES >= 0)    P(m_p.K_HOWES, k, j, i)    *= reduce_e;
        if (m_p.K_KAWAZURA >= 0) P(m_p.K_KAWAZURA, k, j, i) *= reduce_e;
        if (m_p.K_WERNER >= 0)   P(m_p.K_WERNER, k, j, i)   *= reduce_e;
        if (m_p.K_ROWAN >= 0)    P(m_p.K_ROWAN, k, j, i)    *= reduce_e;
        if (m_p.K_SHARMA >= 0)   P(m_p.K_SHARMA, k, j, i)   *= reduce_e;
    }

    // Return fflag (with pflag added if NOF floors were used!)
    return fflag;
}

/**
 * Apply just the geometric floors to a set of local primitives.
 * Specifically called after reconstruction when using non-TVD schemes, e.g. WENO5.
 * Reimplemented to be fast and fit the general prim_to_flux calling convention.
 * 
 * @return fflag: since no inversion is performed, this just returns a flag representing which geometric floors were hit
 * 
 * NOT LOCKSTEP: Operates on and respects primitives *only*
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, Local& P, const VarMap& m,
                                            const Real& gam, const int& j, const int& i,
                                            const Floors::Prescription& floors, const Loci loc=Loci::center)
{
    // Apply only the geometric floors
    Real rhoflr_geom, uflr_geom;
    if(G.coords.spherical()) {
        GReal Xembed[GR_DIM];
        G.coord_embed(0, j, i, loc, Xembed);
        GReal r = Xembed[1];

        if (floors.use_r_char) {
            // Steeper floor from iharm3d
            Real rhoscal = m::pow(r, -2.) * 1 / (1 + r / floors.r_char);
            rhoflr_geom  = floors.rho_min_geom * rhoscal;
            uflr_geom    = floors.u_min_geom * m::pow(rhoscal, gam);
        } else {
            // Original floors from iharm2d
            rhoflr_geom = floors.rho_min_geom * m::pow(r, -1.5);
            uflr_geom   = floors.u_min_geom * m::pow(r, -2.5); //rhoscal/r as in iharm2d
        }
    } else {
        rhoflr_geom = floors.rho_min_geom;
        uflr_geom   = floors.u_min_geom;
    }

    int fflag = 0;
#if RECORD_POST_RECON
    // Record all the floors that were hit, using bitflags
    // Record Geometric floor hits
    fflag |= (rhoflr_geom > P(m.RHO)) * FFlag::GEOM_RHO_FLUX;
    fflag |= (uflr_geom > P(m.UU)) * FFlag::GEOM_U_FLUX;
#endif

    P(m.RHO) += m::max(0., rhoflr_geom - P(m.RHO));
    P(m.UU)  += m::max(0., uflr_geom - P(m.UU));

    return fflag;
}

template<typename Global>
KOKKOS_INLINE_FUNCTION int apply_geo_floors(const GRCoordinates& G, Global& P, const VarMap& m,
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
            Real rhoscal = m::pow(r, -2.) * 1 / (1 + r / floors.r_char);
            rhoflr_geom  = floors.rho_min_geom * rhoscal;
            uflr_geom    = floors.u_min_geom * m::pow(rhoscal, gam);
        } else {
            // Original floors from iharm2d
            rhoflr_geom = floors.rho_min_geom * m::pow(r, -1.5);
            uflr_geom   = floors.u_min_geom * m::pow(r, -2.5); //rhoscal/r as in iharm2d
        }
    } else {
        rhoflr_geom = floors.rho_min_geom;
        uflr_geom   = floors.u_min_geom;
    }

    int fflag = 0;
#if RECORD_POST_RECON
    // Record all the floors that were hit, using bitflags
    // Record Geometric floor hits
    fflag |= (rhoflr_geom > P(m.RHO, k, j, i)) * FFlag::GEOM_RHO_FLUX;
    fflag |= (uflr_geom > P(m.UU, k, j, i)) * FFlag::GEOM_U_FLUX;
#endif

    P(m.RHO, k, j, i) += m::max(0., rhoflr_geom - P(m.RHO, k, j, i));
    P(m.UU, k, j, i)  += m::max(0., uflr_geom - P(m.UU, k, j, i));

    return fflag;
}

} // Floors