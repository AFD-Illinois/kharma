/* 
 *  File: flux_functions.hpp
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

#include "emhd.hpp"
#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"
#include "kharma_utils.hpp"
#include "types.hpp"
/**
 * Device-side functions prim_to_flux and vchar, which will depend on
 * the set of enabled packages.
 */

namespace Flux
{

template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G, const Local& P, const VarMap& m_p, const FourVectors D,
                                        const EMHD::EMHD_parameters& emhd_params, const Real& gam, const int& dir,
                                        Real T[GR_DIM])
{
    if (m_p.Q >= 0) {
        // Apply higher-order terms conversion if necessary
        Real q, dP;
        const Real Theta = (gam - 1) * P(m_p.UU) / P(m_p.RHO);
        const Real cs2   = gam * (gam - 1) * P(m_p.UU) / (P(m_p.RHO) + gam * P(m_p.UU));
        EMHD::convert_prims_to_q_dP(P(m_p.Q), P(m_p.DP), P(m_p.RHO), Theta, cs2, emhd_params, q, dP);

        // Then calculate the tensor
        EMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), emhd_params, q, dP, D, dir, T);
    } else if (m_p.B1 >= 0) {
        // GRMHD stress-energy tensor w/ first index up, second index down
        GRMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, T);
    } else {
        // GRHD stress-energy tensor w/ first index up, second index down
        GRHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, T);
    }
}

template<typename Global>
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G, const Global& P, const VarMap& m_p, const FourVectors D,
                                        const EMHD::EMHD_parameters& emhd_params, const Real& gam, 
                                        const int& k, const int& j, const int& i, const int& dir,
                                        Real T[GR_DIM])
{
    if (m_p.Q >= 0) {

        // Apply higher-order terms conversion if necessary
        Real q, dP;
        const Real Theta = (gam - 1) * P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i);
        const Real cs2   = gam * (gam - 1) * P(m_p.UU, k, j, i) / (P(m_p.RHO, k, j, i) + gam * P(m_p.UU, k, j, i));
        EMHD::convert_prims_to_q_dP(P(m_p.Q, k, j, i), P(m_p.DP, k, j, i), P(m_p.RHO, k, j, i), Theta, cs2, emhd_params, q, dP);

        // Then calculate the tensor
        EMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), emhd_params, q, dP, D, dir, T);
    } else if (m_p.B1 >= 0) {
        // GRMHD stress-energy tensor w/ first index up, second index down
        GRMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
    } else {
        // GRHD stress-energy tensor w/ first index up, second index down
        GRHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
    }
}

template<typename Local>
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G, const Local& P, const VarMap& m_p, const FourVectors D,
                                         const Real& gam, const int& dir,
                                         Real T[GR_DIM])
{
    if (m_p.B1 >= 0) {
        // GRMHD stress-energy tensor w/ first index up, second index down
        GRMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, T);
    } else {
        // GRHD stress-energy tensor w/ first index up, second index down
        GRHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, T);
    }
}

template<typename Global>
KOKKOS_INLINE_FUNCTION void calc_tensor(const GRCoordinates& G, const Global& P, const VarMap& m_p, const FourVectors D,
                                         const Real& gam, const int& k, const int& j, const int& i, const int& dir,
                                         Real T[GR_DIM])
{
    if (m_p.B1 >= 0) {
        // GRMHD stress-energy tensor w/ first index up, second index down
        GRMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
    } else {
        // GRHD stress-energy tensor w/ first index up, second index down
        GRHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
    }
}

/**
 * Turn the primitive variables at a location into:
 * a. conserved variables (dir==0), or
 * b. fluxes in a direction (dir!=0)
 * Keep in mind loc should usually correspond to dir for perpendicuar fluxes
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const Local& P, const VarMap& m_p, const FourVectors D,
                                         const EMHD::EMHD_parameters& emhd_params, const Real& gam, const int& j, const int& i, const int& dir,
                                         const Local& flux, const VarMap& m_u, const Loci loc=Loci::center)
{
    Real gdet = G.gdet(loc, j, i);
    // Particle number flux
    flux(m_u.RHO) = P(m_p.RHO) * D.ucon[dir] * gdet;

    // Stress-energy tensor
    Real T[GR_DIM];
    calc_tensor(G, P, m_p, D, emhd_params, gam, dir, T);
    flux(m_u.UU) = T[0] * gdet + flux(m_u.RHO);
    flux(m_u.U1) = T[1] * gdet;
    flux(m_u.U2) = T[2] * gdet;
    flux(m_u.U3) = T[3] * gdet;

    // Magnetic field
    if (m_p.B1 >= 0) {
        // Magnetic field
        if (dir == 0) {
            VLOOP flux(m_u.B1 + v) = P(m_p.B1 + v) * gdet;
        } else {
            // Constraint damping w/Dedner may add also P(m_p.psi) * gdet,
            // but for us this is in the source term
            VLOOP flux(m_u.B1 + v) = (D.bcon[v+1] * D.ucon[dir] - D.bcon[dir] * D.ucon[v+1]) * gdet;
        }
        // Extra scalar psi for constraint damping, see B_CD
        if (m_p.PSI >= 0) {
            if (dir == 0) {
                flux(m_u.PSI) = P(m_p.PSI) * gdet;
            } else {
                // Psi field update as in Mosta et al (IllinoisGRMHD), alternate explanation Jesse et al (2020)
                //Real alpha = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
                //Real beta_dir = G.gcon(Loci::center, j, i, 0, dir) * alpha * alpha;
                flux(m_u.PSI) = (D.bcon[dir] - G.gcon(Loci::center, j, i, 0, dir) * P(m_p.PSI)) * gdet;
            }
        }
    }

    // EMHD Variables: advect like rho
    if (m_p.Q >= 0) {
        flux(m_u.Q) = P(m_p.Q) * D.ucon[dir] * gdet;
        flux(m_u.DP) = P(m_p.DP) * D.ucon[dir] * gdet;
    }

    // Electrons: normalized by density
    if (m_p.KTOT >= 0) {
        flux(m_u.KTOT) = flux(m_u.RHO) * P(m_p.KTOT);
        if (m_p.K_CONSTANT >= 0)
            flux(m_u.K_CONSTANT) = flux(m_u.RHO) * P(m_p.K_CONSTANT);
        if (m_p.K_HOWES >= 0)
            flux(m_u.K_HOWES) = flux(m_u.RHO) * P(m_p.K_HOWES);
        if (m_p.K_KAWAZURA >= 0)
            flux(m_u.K_KAWAZURA) = flux(m_u.RHO) * P(m_p.K_KAWAZURA);
        if (m_p.K_WERNER >= 0)
            flux(m_u.K_WERNER) = flux(m_u.RHO) * P(m_p.K_WERNER);
        if (m_p.K_ROWAN >= 0)
            flux(m_u.K_ROWAN) = flux(m_u.RHO) * P(m_p.K_ROWAN);
        if (m_p.K_SHARMA >= 0)
            flux(m_u.K_SHARMA) = flux(m_u.RHO) * P(m_p.K_SHARMA);
    }

}

template<typename Global>
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const Global& P, const VarMap& m_p, const FourVectors D,
                                         const EMHD::EMHD_parameters& emhd_params, const Real& gam, 
                                         const int& k, const int& j, const int& i, const int dir,
                                         const Global& flux, const VarMap& m_u, const Loci loc=Loci::center)
{
    const Real gdet = G.gdet(loc, j, i);
    // Particle number flux
    flux(m_u.RHO, k, j, i) = P(m_p.RHO, k, j, i) * D.ucon[dir] * gdet;

    Real T[GR_DIM];
    if (m_p.Q >= 0) {

        // Apply higher-order terms conversion if necessary
        Real q, dP;
        const Real Theta = (gam - 1) * P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i);
        const Real cs2   = gam * (gam - 1) * P(m_p.UU, k, j, i) / (P(m_p.RHO, k, j, i) + gam * P(m_p.UU, k, j, i));
        EMHD::convert_prims_to_q_dP(P(m_p.Q, k, j, i), P(m_p.DP, k, j, i), P(m_p.RHO, k, j, i), Theta, cs2, emhd_params, q, dP);

        // Then calculate the tensor
        EMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), emhd_params, q, dP, D, dir, T);
    } else if (m_p.B1 >= 0) {
        // GRMHD stress-energy tensor w/ first index up, second index down
        GRMHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
    } else {
        // GRHD stress-energy tensor w/ first index up, second index down
        GRHD::calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), D, dir, T);
    }
    // if (i == 11 && j == 11) printf("mhd: %6.5e %6.5e %6.5e %6.5e %6.5e\n", flux(m_u.RHO), T[0], T[1], T[2], T[3]);
    flux(m_u.UU, k, j, i) = T[0] * gdet + flux(m_u.RHO, k, j, i);
    flux(m_u.U1, k, j, i) = T[1] * gdet;
    flux(m_u.U2, k, j, i) = T[2] * gdet;
    flux(m_u.U3, k, j, i) = T[3] * gdet;

    // Magnetic field
    if (m_p.B1 >= 0) {
        // Magnetic field
        if (dir == 0) {
            VLOOP flux(m_u.B1 + v, k, j, i) = P(m_p.B1 + v, k, j, i) * gdet;
        } else {
            // Constraint damping w/Dedner may add also P(m_p.psi) * gdet,
            // but for us this is in the source term
            VLOOP flux(m_u.B1 + v, k, j, i) = (D.bcon[v+1] * D.ucon[dir] - D.bcon[dir] * D.ucon[v+1]) * gdet;
        }
        // Extra scalar psi for constraint damping, see B_CD
        if (m_p.PSI >= 0) {
            if (dir == 0) {
                flux(m_u.PSI, k, j, i) = P(m_p.PSI, k, j, i) * gdet;
            } else {
                // Psi field update as in Mosta et al (IllinoisGRMHD), alternate explanation Jesse et al (2020)
                //Real alpha = 1. / sqrt(-G.gcon(Loci::center, j, i, 0, 0));
                //Real beta_dir = G.gcon(Loci::center, j, i, 0, dir) * alpha * alpha;
                flux(m_u.PSI, k, j, i) = (D.bcon[dir] - G.gcon(Loci::center, j, i, 0, dir) * P(m_p.PSI, k, j, i)) * gdet;
            }
        }
    }

    // EMHD Variables: advect like rho
    if (m_p.Q >= 0) {
        flux(m_u.Q, k, j, i)  = P(m_p.Q, k, j, i) * D.ucon[dir] * gdet;
        flux(m_u.DP, k, j, i) = P(m_p.DP, k, j, i) * D.ucon[dir] * gdet;
    }

    // Electrons: normalized by density
    if (m_p.KTOT >= 0) {
        flux(m_u.KTOT, k, j, i)  = flux(m_u.RHO, k, j, i) * P(m_p.KTOT, k, j, i);
        if (m_p.K_CONSTANT >= 0)
            flux(m_u.K_CONSTANT, k, j, i) = flux(m_u.RHO, k, j, i) * P(m_p.K_CONSTANT, k, j, i);
        if (m_p.K_HOWES >= 0)
            flux(m_u.K_HOWES, k, j, i)    = flux(m_u.RHO, k, j, i) * P(m_p.K_HOWES, k, j, i);
        if (m_p.K_KAWAZURA >= 0)
            flux(m_u.K_KAWAZURA, k, j, i) = flux(m_u.RHO, k, j, i) * P(m_p.K_KAWAZURA, k, j, i);
        if (m_p.K_WERNER >= 0)
            flux(m_u.K_WERNER, k, j, i)   = flux(m_u.RHO, k, j, i) * P(m_p.K_WERNER, k, j, i);
        if (m_p.K_ROWAN >= 0)
            flux(m_u.K_ROWAN, k, j, i)    = flux(m_u.RHO, k, j, i) * P(m_p.K_ROWAN, k, j, i);
        if (m_p.K_SHARMA >= 0)
            flux(m_u.K_SHARMA, k, j, i)   = flux(m_u.RHO, k, j, i) * P(m_p.K_SHARMA, k, j, i);
    }

}

/**
 * Get the conserved (E)GRMHD variables corresponding to primitives in a zone. Equivalent to prim_to_flux with dir==0
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Local& P, const VarMap& m_p,
                                   const EMHD::EMHD_parameters& emhd_params, const Real& gam, const int& j, const int& i,
                                   const Local& U, const VarMap& m_u, const Loci& loc=Loci::center)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, j, i, loc, Dtmp); // TODO switch GRHD/GRMHD?
    prim_to_flux(G, P, m_p, Dtmp, emhd_params, gam, j, i, 0, U, m_u, loc);
    // printf("%d %d %6.5e %6.5e\n", i, j, P(m_p.Q), P(m_p.DP));
}

template<typename Global>
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Global& P, const VarMap& m_p,
                                   const EMHD::EMHD_parameters& emhd_params, const Real& gam, 
                                   const int& k, const int& j, const int& i,
                                   const Global& U, const VarMap& m_u, const Loci& loc=Loci::center)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    prim_to_flux(G, P, m_p, Dtmp, emhd_params, gam, k, j, i, 0, U, m_u, loc);
}

/**
 * Calculate components of magnetosonic velocity from primitive variables
 * This is only called in GetFlux, so we only provide a ScratchPad form
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION void vchar(const GRCoordinates& G, const Local& P, const VarMap& m, const FourVectors& D,
                                  const Real& gam, const EMHD::EMHD_parameters& emhd_params, 
                                  const int& k, const int& j, const int& i, const Loci& loc, const int& dir,
                                  Real& cmax, Real& cmin)
{
    // Find sound speed
    const Real ef  = P(m.RHO) + gam * P(m.UU);
    const Real cs2 = gam * (gam - 1) * P(m.UU) / ef;
    Real cms2;
    if (m.Q > 0) {
         // Get the EGRMHD parameters
        Real tau, chi_e, nu_e;
        EMHD::set_parameters(G, P, m, emhd_params, gam, k, j, i, tau, chi_e, nu_e);        
        
        // Find fast magnetosonic speed
        const Real bsq = m::max(dot(D.bcon, D.bcov), SMALL);
        const Real ee  = bsq + ef;
        const Real va2 = bsq / ee;

        const Real cvis2  = (4./3.) / (P(m.RHO) + (gam * P(m.UU)) ) * P(m.RHO) * emhd_params.viscosity_alpha * cs2;
        const Real ccond2 = (gam - 1.) * emhd_params.conduction_alpha * cs2;

        const Real cscond   = 0.5*(cs2 + ccond2 + sqrt(cs2*cs2 + ccond2*ccond2) ) ;
        const Real cs2_emhd = cscond + cvis2;

        cms2 = cs2_emhd + va2 - cs2_emhd*va2;
    } else if (m.B1 >= 0) {
        // Find fast magnetosonic speed
        const Real bsq = m::max(dot(D.bcon, D.bcov), SMALL);
        const Real ee  = bsq + ef;
        const Real va2 = bsq / ee;

        cms2 = cs2 + va2 - cs2 * va2;
    } else {
        cms2 = cs2;
    }
    clip(cms2, SMALL, 1.);

    // Require that speed of wave measured by observer q.ucon is cms2
    Real A, B, C;
    {
        Real Bcov[GR_DIM] = {1., 0., 0., 0.};
        Real Acov[GR_DIM] = {0}; Acov[dir] = 1.;

        Real Acon[GR_DIM], Bcon[GR_DIM];
        G.raise(Acov, Acon, k, j, i, loc);
        G.raise(Bcov, Bcon, k, j, i, loc);

        const Real Asq  = dot(Acon, Acov);
        const Real Bsq  = dot(Bcon, Bcov);
        const Real Au   = dot(Acov, D.ucon);
        const Real Bu   = dot(Bcov, D.ucon);
        const Real AB   = dot(Acon, Bcov);
        const Real Au2  = Au * Au;
        const Real Bu2  = Bu * Bu;
        const Real AuBu = Au * Bu;

        A = Bu2 - (Bsq + Bu2) * cms2;
        B = 2. * (AuBu - (AB + AuBu) * cms2);
        C = Au2 - (Asq + Au2) * cms2;
    }

    Real discr = m::sqrt(m::max(B * B - 4. * A * C, 0.));

    Real vp = -(-B + discr) / (2. * A);
    Real vm = -(-B - discr) / (2. * A);

    cmax = m::max(vp, vm);
    cmin = m::min(vp, vm);
}

} // namespace Flux
