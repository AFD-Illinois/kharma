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

/**
 * Turn the primitive variables at a location into:
 * a. conserved variables (dir==0), or
 * b. fluxes in a direction (dir!=0)
 * Keep in mind loc should usually correspond to dir for perpendicuar fluxes
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const Local& P, const VarMap& m_p, const FourVectors D,
                                         const Real& gam, const int& j, const int& i, const int dir,
                                         const Local& flux, const VarMap& m_u, const Loci loc=Loci::center)
{
    Real gdet = G.gdet(loc, j, i);
    // Particle number flux
    flux(m_u.RHO) = P(m_p.RHO) * D.ucon[dir] * gdet;

    if (m_p.B1 >= 0) {
        // MHD stress-energy tensor w/ first index up, second index down
        Real mhd[GR_DIM];
        GRMHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, mhd);
        flux(m_u.UU) = mhd[0] * gdet + flux(m_u.RHO);
        flux(m_u.U1) = mhd[1] * gdet;
        flux(m_u.U2) = mhd[2] * gdet;
        flux(m_u.U3) = mhd[3] * gdet;

        // Magnetic field
        if (dir == 0) {
            VLOOP flux(m_u.B1 + v) = P(m_p.B1 + v) * gdet;
        } else {
            // Constraint damping w/Dedner may add also P(m_p.psi) * gdet,
            // but for us this is in the source term
            VLOOP flux(m_u.B1 + v) = (D.bcon[v+1] * D.ucon[dir] - D.bcon[dir] * D.ucon[v+1]) * gdet;
        }
    } else {
        // HD stress-energy tensor w/ first index up, second index down
        Real hd[GR_DIM];
        GRHD::calc_tensor(P(m_p.RHO), P(m_p.UU), (gam - 1) * P(m_p.UU), D, dir, hd);
        flux(m_u.UU) = hd[0] * gdet + flux(m_u.RHO);
        flux(m_u.U1) = hd[1] * gdet;
        flux(m_u.U2) = hd[2] * gdet;
        flux(m_u.U3) = hd[3] * gdet;
    }
    if (m_p.PSI >= 0) {
        // Extra scalar psi for constraint damping, see B_CD
        if (dir == 0) {
            flux(m_u.PSI) = P(m_p.PSI) * gdet;
        } else {
            // Psi field update as in Mosta et al (IllinoisGRMHD), alternate explanation Jesse et al (2020)
            //Real alpha = 1. / sqrt(-G.gcon(Loci::center, j, i, 0, 0));
            //Real beta_dir = G.gcon(Loci::center, j, i, 0, dir) * alpha * alpha;
            flux(m_u.PSI) = (D.bcon[dir] - G.gcon(Loci::center, j, i, 0, dir) * P(m_p.PSI)) * gdet;
        }
    }

    if (m_p.KTOT >= 0) {
        // Take the factor from the primitives, in case we need to reorder this to happen before GRMHD::prim_to_flux later
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

/**
 * Get the conserved GRHD variables corresponding to primitives in a zone. Equivalent to prim_to_flux with dir==0
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Local& P, const VarMap& m_p,
                                   const Real& gam, const int& j, const int& i,
                                   const Local& U, const VarMap& m_u, const Loci& loc=Loci::center)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, j, i, loc, Dtmp); // TODO switch GRHD/GRMHD
    prim_to_flux(G, P, m_p, Dtmp, gam, j, i, 0, U, m_u, loc);
}
// KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
//                                    const Real& gam, const int& k, const int& j, const int& i,
//                                    const VariablePack<Real>& U, const VarMap& m_u, const Loci& loc=Loci::center)
// {

// }

/**
 * Calculate components of magnetosonic velocity from primitive variables
 * This is only called in GetFlux, so we only provide a ScratchPad form
 */
template<typename Local>
KOKKOS_INLINE_FUNCTION void vchar(const GRCoordinates& G, const Local& P, const VarMap& m, const FourVectors& D,
                                  const Real& gam, const int& k, const int& j, const int& i, const Loci& loc, const int& dir,
                                  Real& cmax, Real& cmin)
{
    // Find sound speed
    const Real ef = P(m.RHO) + gam * P(m.UU);
    const Real cs2 = gam * (gam - 1) * P(m.UU) / ef;
    Real cms2;
    if (m.B1 >= 0) {
        // Find fast magnetosonic speed
        const Real bsq = dot(D.bcon, D.bcov);
        const Real ee = bsq + ef;
        const Real va2 = bsq / ee;
        cms2 = cs2 + va2 - cs2 * va2;
    } else {
        cms2 = cs2;
    }
    clip(cms2, 1.e-20, 1.);

    // Require that speed of wave measured by observer q.ucon is cms2
    Real A, B, C;
    {
        Real Bcov[GR_DIM] = {1., 0., 0., 0.};
        Real Acov[GR_DIM] = {0}; Acov[dir] = 1.;

        Real Acon[GR_DIM], Bcon[GR_DIM];
        G.raise(Acov, Acon, k, j, i, loc);
        G.raise(Bcov, Bcon, k, j, i, loc);

        const Real Asq = dot(Acon, Acov);
        const Real Bsq = dot(Bcon, Bcov);
        const Real Au = dot(Acov, D.ucon);
        const Real Bu = dot(Bcov, D.ucon);
        const Real AB = dot(Acon, Bcov);
        const Real Au2 = Au * Au;
        const Real Bu2 = Bu * Bu;
        const Real AuBu = Au * Bu;

        A = Bu2 - (Bsq + Bu2) * cms2;
        B = 2. * (AuBu - (AB + AuBu) * cms2);
        C = Au2 - (Asq + Au2) * cms2;
    }

    Real discr = sqrt(max(B * B - 4. * A * C, 0.));

    Real vp = -(-B + discr) / (2. * A);
    Real vm = -(-B - discr) / (2. * A);

    cmax = max(vp, vm);
    cmin = min(vp, vm);
}

} // namespace Flux
