/* 
 *  File: mhd_functions.hpp
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

#include "eos.hpp"
#include "gr_coordinates.hpp"
#include "phys_functions.hpp"
#include "utils.hpp"

/**
 * Device-side MHD functions
 * They are specifically the subset which require the fluid primitives P & B field both
 *
 * These functions mostly have several overloads, related to local vs global variables
 * One version usually takes a local cache e.g. P[NPRIM] of state indexed P[p]
 * The other version(s) take e.g. P, the pointer to the full array indexed by P(p,i,j,k)
 *
 * This allows easy fusing/splitting of loops & use in different contexts
 */

namespace GRMHD
{

/**
 * Get a row of the MHD stress-energy tensor with first index up, second index down.
 * A factor of sqrt(4 pi) is absorbed into the definition of b.
 * See Gammie & McKinney '04
 */
KOKKOS_INLINE_FUNCTION void calc_tensor(const Real& rho, const Real& u, const Real& pgas,
                                            const FourVectors& D, const int dir,
                                            Real mhd[GR_DIM])
{
    Real bsq = dot(D.bcon, D.bcov);
    Real eta = pgas + rho + u + bsq;
    Real ptot = pgas + 0.5 * bsq;

    DLOOP1 {
        mhd[mu] = eta * D.ucon[dir] * D.ucov[mu] +
                  ptot * (dir == mu) -
                  D.bcon[dir] * D.bcov[mu];
    }
}

/**
 * Velocity 4-vector
 */
KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates &G, const GridVars P,
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      Real ucon[GR_DIM])
{
    Real gamma = lorentz_calc(G, P, k, j, i, loc);
    // TODO try caching alpha?
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));
    ucon[0] = gamma / alpha;

    VLOOP ucon[v+1] = P(prims::u1 + v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);
}

/**
 *  Versions of ucon_calc/bcon_calc specifically for filling D (TODO merge?)
 */
KOKKOS_INLINE_FUNCTION void add_ucon(const GRCoordinates &G, const Real uv[NVEC],
                                      const int& j, const int& i, const Loci loc,
                                      FourVectors& D)
{
    Real gamma = lorentz_calc(G, uv, j, i, loc);
    // TODO try caching alpha?
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));
    D.ucon[0] = gamma / alpha;

    VLOOP D.ucon[v+1] = uv[v] - gamma * alpha * G.gcon(loc, j, i, 0, v+1);
}
KOKKOS_INLINE_FUNCTION void add_bcon(const Real B_P[NVEC], FourVectors& D)
{
    D.bcon[0] = 0;
    VLOOP D.bcon[0] += B_P[v] * D.ucov[v+1];
    VLOOP {
        D.bcon[v+1] = (B_P[v] + D.bcon[0] * D.ucon[v+1]) / D.ucon[0];
    }
}

/**
 * Calculate ucon, ucov, bcon, bcov from primitive variables
 */
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const Real P[NVEC], const Real B_P[NVEC],
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      FourVectors& D)
{
    // All-local version: straight from the velocities
    add_ucon(G, &(P[prims::u1]), j, i, loc, D);
    G.lower(D.ucon, D.ucov, k, j, i, loc);
    add_bcon(B_P, D);
    G.lower(D.bcon, D.bcov, k, j, i, loc);
}
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const GridVars P, const GridVector B_P,
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      FourVectors& D)
{
    // Build the local vectors on the fly
    Real Pl[NPRIM] = {P(prims::rho, k, j, i), P(prims::u, k, j, i), P(prims::u1, k, j, i),
                      P(prims::u2, k, j, i), P(prims::u3, k, j, i)};
    Real B_Pl[NVEC] = {B_P(0, k, j, i), B_P(1, k, j, i), B_P(2, k, j, i)};
    calc_4vecs(G, Pl, B_Pl, k, j, i, loc, D);
}

/**
 * Turn the primitive variables at a location into the local conserved variables, or fluxes at a face
 * 
 * Note this is fluid only -- each package defines a prim_to_flux, which are called in GetFlux
 */
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const Real& rho, const Real& u, const Real& pgas,
                                         const FourVectors D, const int& k, const int& j, const int& i, const Loci loc,
                                         const int dir, Real flux[NPRIM])
{
    Real mhd[GR_DIM];
    Real gdet = G.gdet(loc, j, i);

    // Particle number flux
    flux[prims::rho] = rho * D.ucon[dir] * gdet;

    // MHD stress-energy tensor w/ first index up, second index down
    calc_tensor(rho, u, pgas, D, dir, mhd);
    flux[prims::u] = mhd[0] * gdet + flux[prims::rho];
    flux[prims::u1] = mhd[1] * gdet;
    flux[prims::u2] = mhd[2] * gdet;
    flux[prims::u3] = mhd[3] * gdet;
}

/**
 * Get the conserved (fluid only!) variables corresponding to primitives in a zone
 * 
 * The convenient version here is actually probably pretty slow. TODO re-implement everything above?
 */
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates &G, const GridVars P, const GridVector B_P, const EOS* eos,
                                         const int& k, const int& j, const int& i,
                                         GridVars U, const Loci loc=Loci::center)
{
    Real Pl[NPRIM] = {P(prims::rho, k, j, i), P(prims::u, k, j, i), P(prims::u1, k, j, i),
                      P(prims::u2, k, j, i), P(prims::u3, k, j, i)};
    Real B_Pl[NVEC] = {B_P(0, k, j, i), B_P(1, k, j, i), B_P(2, k, j, i)};
    FourVectors Dtmp;
    calc_4vecs(G, Pl, B_Pl, k, j, i, loc, Dtmp);
    Real rho = P(prims::rho, k, j, i);
    Real u = P(prims::u, k, j, i);
    Real pgas = eos->p(rho, u);
    Real Ul[NPRIM] = {0};
    prim_to_flux(G, rho, u, pgas, Dtmp, k, j, i, loc, 0, Ul);

    PLOOP U(p, k, j, i) = Ul[p];
}
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates &G, const GridVars P, const GridVector B_P, const EOS* eos,
                                         const int& k, const int& j, const int& i,
                                         Real U[NPRIM], const Loci loc=Loci::center)
{
    Real Pl[NPRIM] = {P(prims::rho, k, j, i), P(prims::u, k, j, i), P(prims::u1, k, j, i),
                      P(prims::u2, k, j, i), P(prims::u3, k, j, i)};
    Real B_Pl[NVEC] = {B_P(0, k, j, i), B_P(1, k, j, i), B_P(2, k, j, i)};
    FourVectors Dtmp;
    calc_4vecs(G, Pl, B_Pl, k, j, i, loc, Dtmp);
    Real rho = P(prims::rho, k, j, i);
    Real u = P(prims::u, k, j, i);
    Real pgas = eos->p(rho, u);
    prim_to_flux(G, rho, u, pgas, Dtmp, k, j, i, loc, 0, U);
}
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates &G, const Real P[NPRIM], const Real B_P[NVEC], const EOS* eos,
                                         const int& k, const int& j, const int& i,
                                         Real U[NPRIM], const Loci loc=Loci::center)
{
    FourVectors Dtmp;
    calc_4vecs(G, P, B_P, k, j, i, loc, Dtmp);
    Real rho = P[prims::rho];
    Real u = P[prims::u];
    Real pgas = eos->p(rho, u);
    prim_to_flux(G, rho, u, pgas, Dtmp, k, j, i, loc, 0, U);
}
// KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates &G, const Real P[NPRIM], const EOS* eos,
//                                          const int& k, const int& j, const int& i,
//                                          Real U[NPRIM], const Loci loc=Loci::center)
// {
//     FourVectors Dtmp;
//     Real B_P[NVEC] = {0};
//     calc_4vecs(G, P, B_P, k, j, i, loc, Dtmp);
//     Real rho = P[prims::rho];
//     Real u = P[prims::u];
//     Real pgas = eos->p(rho, u);
//     prim_to_flux(G, rho, u, pgas, Dtmp, k, j, i, loc, 0, U);
// }

/**
 *  Calculate components of magnetosonic velocity from primitive variables
 */
KOKKOS_INLINE_FUNCTION void vchar(const GRCoordinates &G, const Real& rho, const Real& u, const Real& pgas,
                                  const FourVectors& D, const EOS* eos,
                                  const int& k, const int& j, const int& i, const Loci loc, const int dir,
                                  Real& cmax, Real& cmin)
{
    Real discr, vp, vm, bsq, ee, ef, va2, cs2, cms2;
    Real Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;
    Real Acov[GR_DIM] = {0}, Bcov[GR_DIM] = {0};
    Real Acon[GR_DIM] = {0}, Bcon[GR_DIM] = {0};

    Acov[dir] = 1.;
    Bcov[0] = 1.;

    DLOOP2 // TODO use lower() & compare speed
    {
        Acon[mu] += G.gcon(loc, j, i, mu, nu) * Acov[nu];
        Bcon[mu] += G.gcon(loc, j, i, mu, nu) * Bcov[nu];
    }

    // Find fast magnetosonic speed
    bsq = dot(D.bcon, D.bcov);
    ef = rho + eos->gam * u;
    ee = bsq + ef;
    va2 = bsq / ee;
    cs2 = eos->gam * pgas / ef;

    cms2 = cs2 + va2 - cs2 * va2;

    clip(cms2, 1.e-20, 1.);

    // Require that speed of wave measured by observer q.ucon is cms2
    Asq = dot(Acon, Acov);
    Bsq = dot(Bcon, Bcov);
    Au = dot(Acov, D.ucon);
    Bu = dot(Bcov, D.ucon);
    AB = dot(Acon, Bcov);
    Au2 = Au * Au;
    Bu2 = Bu * Bu;
    AuBu = Au * Bu;

    A = Bu2 - (Bsq + Bu2) * cms2;
    B = 2. * (AuBu - (AB + AuBu) * cms2);
    C = Au2 - (Asq + Au2) * cms2;

    discr = sqrt(max(B * B - 4. * A * C, 0.));

    vp = -(-B + discr) / (2. * A);
    vm = -(-B - discr) / (2. * A);

    cmax = max(vp, vm);
    cmin = min(vp, vm);
}

}
