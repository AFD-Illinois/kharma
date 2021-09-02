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

#include "gr_coordinates.hpp"
#include "types.hpp"
#include "utils.hpp"

/**
 * Device-side MHD functions
 * They are specifically the subset which require the fluid primitives P & B field both
 *
 * These functions mostly have several overloads, related to local vs global variables
 * Arguments can come in the form of global array or VariablePack references 
 *
 * This allows easy fusing/splitting of loops & use in different contexts
 */

namespace GRMHD
{

/**
 * Find gamma-factor of the fluid w.r.t. normal observer
 */
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const GridVector uvec,
                                         const int& k, const int& j, const int& i,
                                         const Loci loc)
{

    Real qsq = G.gcov(loc, j, i, 1, 1) * uvec(0, k, j, i) * uvec(0, k, j, i) +
               G.gcov(loc, j, i, 2, 2) * uvec(1, k, j, i) * uvec(1, k, j, i) +
               G.gcov(loc, j, i, 3, 3) * uvec(2, k, j, i) * uvec(2, k, j, i) +
            2. * (G.gcov(loc, j, i, 1, 2) * uvec(0, k, j, i) * uvec(1, k, j, i) +
                  G.gcov(loc, j, i, 1, 3) * uvec(0, k, j, i) * uvec(2, k, j, i) +
                  G.gcov(loc, j, i, 2, 3) * uvec(1, k, j, i) * uvec(2, k, j, i));

    return sqrt(1. + qsq);
}
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const Real uv[NVEC],
                                         const int& k, const int& j, const int& i,
                                         const Loci loc)
{
    Real qsq = G.gcov(loc, j, i, 1, 1) * uv[0] * uv[0] +
               G.gcov(loc, j, i, 2, 2) * uv[1] * uv[1] +
               G.gcov(loc, j, i, 3, 3) * uv[2] * uv[2] +
            2. * (G.gcov(loc, j, i, 1, 2) * uv[0] * uv[1] +
                  G.gcov(loc, j, i, 1, 3) * uv[0] * uv[2] +
                  G.gcov(loc, j, i, 2, 3) * uv[1] * uv[2]);

    return sqrt(1. + qsq);
}
// Version for full primitives array
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m,
                                         const int& k, const int& j, const int& i, const Loci& loc)
{

    Real qsq = G.gcov(loc, j, i, 1, 1) * P(m.U1, k, j, i) * P(m.U1, k, j, i) +
               G.gcov(loc, j, i, 2, 2) * P(m.U2, k, j, i) * P(m.U2, k, j, i) +
               G.gcov(loc, j, i, 3, 3) * P(m.U3, k, j, i) * P(m.U3, k, j, i) +
            2. * (G.gcov(loc, j, i, 1, 2) * P(m.U1, k, j, i) * P(m.U2, k, j, i) +
                  G.gcov(loc, j, i, 1, 3) * P(m.U1, k, j, i) * P(m.U3, k, j, i) +
                  G.gcov(loc, j, i, 2, 3) * P(m.U2, k, j, i) * P(m.U3, k, j, i));

    return sqrt(1. + qsq);
}
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates& G, const ScratchPad2D<Real>& P, const VarMap& m,
                                         const int& k, const int& j, const int& i, const Loci& loc)
{
    Real qsq = G.gcov(loc, j, i, 1, 1) * P(m.U1, i) * P(m.U1, i) +
               G.gcov(loc, j, i, 2, 2) * P(m.U2, i) * P(m.U2, i) +
               G.gcov(loc, j, i, 3, 3) * P(m.U3, i) * P(m.U3, i) +
            2. * (G.gcov(loc, j, i, 1, 2) * P(m.U1, i) * P(m.U2, i) +
                  G.gcov(loc, j, i, 1, 3) * P(m.U1, i) * P(m.U3, i) +
                  G.gcov(loc, j, i, 2, 3) * P(m.U2, i) * P(m.U3, i));

    return sqrt(1. + qsq);
}

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
 * Just the velocity 4-vector
 */
KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates &G, const GridVector uvec,
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      Real ucon[GR_DIM])
{
    Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));

    ucon[0] = gamma / alpha;
    VLOOP ucon[v+1] = uvec(v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);
}
KOKKOS_INLINE_FUNCTION void calc_ucon(const GRCoordinates &G, const Real uvec[NVEC],
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      Real ucon[GR_DIM])
{
    Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));

    ucon[0] = gamma / alpha;
    VLOOP ucon[v+1] = uvec[v] - gamma * alpha * G.gcon(loc, j, i, 0, v+1);
}

/**
 * Calculate the 4-velocities ucon, ucov, and 4-fields bcon, bcov from primitive versions
 */
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const Real uvec[NVEC], const Real B_P[NVEC],
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      FourVectors& D)
{
    Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));

    D.ucon[0] = gamma / alpha;
    VLOOP D.ucon[v+1] = uvec[v] - gamma * alpha * G.gcon(loc, j, i, 0, v+1);

    G.lower(D.ucon, D.ucov, k, j, i, loc);

    D.bcon[0] = 0;
    VLOOP D.bcon[0] += B_P[v] * D.ucov[v+1];
    VLOOP D.bcon[v+1] = (B_P[v] + D.bcon[0] * D.ucon[v+1]) / D.ucon[0];

    G.lower(D.bcon, D.bcov, k, j, i, loc);
}
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const GridVector uvec, const GridVector B_P,
                                      const int& k, const int& j, const int& i, const Loci loc,
                                      FourVectors& D)
{
    Real gamma = lorentz_calc(G, uvec, k, j, i, loc);
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));

    D.ucon[0] = gamma / alpha;
    VLOOP D.ucon[v+1] = uvec(v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);

    G.lower(D.ucon, D.ucov, k, j, i, loc);

    D.bcon[0] = 0;
    VLOOP D.bcon[0] += B_P(v, k, j, i) * D.ucov[v+1];
    VLOOP D.bcon[v+1] = (B_P(v, k, j, i) + D.bcon[0] * D.ucon[v+1]) / D.ucon[0];

    G.lower(D.bcon, D.bcov, k, j, i, loc);
}

// Primitive/VarMap version of calc_4vecs for kernels that use "packed" primitives
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m,
                                      const int& k, const int& j, const int& i, const Loci loc, FourVectors& D)
{
    Real gamma = lorentz_calc(G, P, m, k, j, i, loc);
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));

    D.ucon[0] = gamma / alpha;
    VLOOP D.ucon[v+1] = P(m.U1 + v, k, j, i) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);

    G.lower(D.ucon, D.ucov, k, j, i, loc);

    D.bcon[0] = 0;
    VLOOP D.bcon[0] += P(m.B1 + v, k, j, i) * D.ucov[v+1];
    VLOOP D.bcon[v+1] = (P(m.B1 + v, k, j, i) + D.bcon[0] * D.ucon[v+1]) / D.ucon[0];

    G.lower(D.bcon, D.bcov, k, j, i, loc);
}
KOKKOS_INLINE_FUNCTION void calc_4vecs(const GRCoordinates& G, const ScratchPad2D<Real>& P, const VarMap& m,
                                      const int& k, const int& j, const int& i, const Loci loc, FourVectors& D)
{
    Real gamma = lorentz_calc(G, P, m, k, j, i, loc);
    Real alpha = 1. / sqrt(-G.gcon(loc, j, i, 0, 0));

    D.ucon[0] = gamma / alpha;
    VLOOP D.ucon[v+1] = P(m.U1 + v, i) - gamma * alpha * G.gcon(loc, j, i, 0, v+1);

    G.lower(D.ucon, D.ucov, k, j, i, loc);

    D.bcon[0] = 0;
    VLOOP D.bcon[0] += P(m.B1 + v, i) * D.ucov[v+1];
    VLOOP D.bcon[v+1] = (P(m.B1 + v, i) + D.bcon[0] * D.ucon[v+1]) / D.ucon[0];

    G.lower(D.bcon, D.bcov, k, j, i, loc);
}

/**
 * Turn the primitive variables at a location into the local conserved variables, or fluxes at a face
 * 
 * Note this is for the five fluid variables only -- each package defines a prim_to_flux, which are called in GetFlux
 */
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const ScratchPad2D<Real>& P, const VarMap& m_p, const FourVectors D,
                                         const Real& gam, const int& k, const int& j, const int& i, const int dir,
                                         ScratchPad2D<Real>& flux, const VarMap m_u, const Loci loc=Loci::center)
{
    Real gdet = G.gdet(loc, j, i);

    // Particle number flux
    flux(m_u.RHO, i) = P(m_p.RHO, i) * D.ucon[dir] * gdet;

    // MHD stress-energy tensor w/ first index up, second index down
    Real mhd[GR_DIM];
    calc_tensor(P(m_p.RHO, i), P(m_p.UU, i), (gam - 1) * P(m_p.UU, i), D, dir, mhd);
    flux(m_u.UU, i)  = mhd[0] * gdet + flux(m_u.RHO, i);
    flux(m_u.U1, i) = mhd[1] * gdet;
    flux(m_u.U2, i) = mhd[2] * gdet;
    flux(m_u.U3, i) = mhd[3] * gdet;
}

/**
 * Get the conserved (fluid only!) variables corresponding to primitives in a zone. Equivalent to prim_to_flux with dir==0
 */
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                   const Real& gam, const int& k, const int& j, const int& i,
                                   const VariablePack<Real>& U, const VarMap m_u, const Loci loc=Loci::center)
{
    Real gdet = G.gdet(loc, j, i);

    FourVectors Dtmp;
    calc_4vecs(G, P, m_p, k, j, i, loc, Dtmp);

    // Particle number flux
    U(m_u.RHO, k, j, i) = P(m_p.RHO, k, j, i) * Dtmp.ucon[0] * gdet;

    // MHD stress-energy tensor w/ first index up, second index down
    Real mhd[GR_DIM];
    calc_tensor(P(m_p.RHO, k, j, i), P(m_p.UU, k, j, i), (gam - 1) * P(m_p.UU, k, j, i), Dtmp, 0, mhd);

    U(m_u.UU, k, j, i)  = mhd[0] * gdet + U(m_u.RHO, k, j, i);
    VLOOP U(m_u.U1 + v, k, j, i) = mhd[1 + v] * gdet;
}

/**
 * Special p_to_u call for fluid frame floors, which require a speculative transformation to add to existing U
 * Also used in the wind source term calculation, of all places
 */
KOKKOS_INLINE_FUNCTION void p_to_u_floor(const GRCoordinates& G, const Real& rho, const Real& u, const Real uvec[NVEC],
                                   const Real& gam, const int& k, const int& j, const int& i,
                                   Real& rho_ut, Real T[GR_DIM], const Loci loc=Loci::center)
{
    Real gdet = G.gdet(loc, j, i);

    FourVectors Dtmp;
    Real B[NVEC] = {0}; // We will never be adding field
    calc_4vecs(G, uvec, B, k, j, i, loc, Dtmp);

    // Particle number flux
    rho_ut = rho * Dtmp.ucon[0] * gdet;

    // MHD stress-energy tensor w/ first index up, second index down
    Real mhd[GR_DIM];
    calc_tensor(rho, u, (gam - 1) * u, Dtmp, 0, mhd);

    T[0]  = mhd[0] * gdet + rho_ut;
    VLOOP T[1 + v] = mhd[1 + v] * gdet;
}


/**
 * Calculate components of magnetosonic velocity from primitive variables
 * This is only called in GetFlux, so we only provide a ScratchPad form
 */
KOKKOS_INLINE_FUNCTION void vchar(const GRCoordinates &G, const ScratchPad2D<Real>& P, const VarMap& m, const FourVectors& D,
                                  const Real& gam, const int& k, const int& j, const int& i, const Loci loc, const int& dir,
                                  Real& cmax, Real& cmin)
{
    // Find fast magnetosonic speed
    Real cms2;
    {
        Real bsq = dot(D.bcon, D.bcov);
        Real ef = P(m.RHO, i) + gam * P(m.UU, i);
        Real ee = bsq + ef;
        Real va2 = bsq / ee;
        Real cs2 = gam * (gam - 1) * P(m.UU, i) / ef;
        cms2 = cs2 + va2 - cs2 * va2;
        clip(cms2, 1.e-20, 1.);
    }

    // Require that speed of wave measured by observer q.ucon is cms2
    Real A, B, C;
    {
        Real Bcov[GR_DIM] = {1., 0., 0., 0.};
        Real Acov[GR_DIM] = {0}; Acov[dir] = 1.;

        Real Acon[GR_DIM], Bcon[GR_DIM];
        G.raise(Acov, Acon, k, j, i, loc);
        G.raise(Bcov, Bcon, k, j, i, loc);

        Real Asq = dot(Acon, Acov);
        Real Bsq = dot(Bcon, Bcov);
        Real Au = dot(Acov, D.ucon);
        Real Bu = dot(Bcov, D.ucon);
        Real AB = dot(Acon, Bcov);
        Real Au2 = Au * Au;
        Real Bu2 = Bu * Bu;
        Real AuBu = Au * Bu;

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

}
