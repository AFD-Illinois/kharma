/* 
 *  File: grmhd_functions.hpp
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

#include "grmhd_functions.hpp"
#include "reductions.hpp"

// GRMHD REDUCTIONS
// Each of these has an identical macro-defined argument list, designed
// to be used in the routines in reductions.cpp.
// You're free to use them elsewhere though, you do you

namespace GRMHD {

// Accretion rates: return a zone's contribution to the surface integral
// forming each rate measurement.
KOKKOS_INLINE_FUNCTION Real mdot(REDUCE_FUNCTION_ARGS_EH)
{
    Real ucon[GR_DIM];
    GRMHD::calc_ucon(G, P, m_p, j, i, Loci::center, ucon);
    // \dot{M} == \int rho * u^1 * gdet * dx2 * dx3
    return -P(m_p.RHO, k, j, i) * ucon[X1DIR] * G.gdet(Loci::center, j, i);
}
KOKKOS_INLINE_FUNCTION Real edot(REDUCE_FUNCTION_ARGS_EH)
{
    FourVectors Dtmp;
    Real T1[GR_DIM];
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    Flux::calc_tensor(P, m_p, Dtmp, gam, k, j, i, X1DIR, T1);
    // \dot{E} == \int - T^1_0 * gdet * dx2 * dx3
    return -T1[X0DIR] * G.gdet(Loci::center, j, i);
}
KOKKOS_INLINE_FUNCTION Real ldot(REDUCE_FUNCTION_ARGS_EH)
{
    FourVectors Dtmp;
    Real T1[GR_DIM];
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    Flux::calc_tensor(P, m_p, Dtmp, gam, k, j, i, X1DIR, T1);
    // \dot{L} == \int T^1_3 * gdet * dx2 * dx3
    return T1[X3DIR] * G.gdet(Loci::center, j, i);
}

// Then we can define the same with fluxes.
KOKKOS_INLINE_FUNCTION Real mdot_flux(REDUCE_FUNCTION_ARGS_EH)
{
    return -U.flux(X1DIR, m_u.RHO, k, j, i);
}
KOKKOS_INLINE_FUNCTION Real edot_flux(REDUCE_FUNCTION_ARGS_EH)
{
    return (U.flux(X1DIR, m_u.UU, k, j, i) - U.flux(X1DIR, m_u.RHO, k, j, i));
}
KOKKOS_INLINE_FUNCTION Real ldot_flux(REDUCE_FUNCTION_ARGS_EH)
{
    return U.flux(X1DIR, m_u.U3, k, j, i);
}

// Luminosity proxy from (for example) Porth et al 2019.
// Notice that this will be totaled for *all zones*,
// but one could define a variable which checks sigma, G.coord_embed(), etc
KOKKOS_INLINE_FUNCTION Real eht_lum(REDUCE_FUNCTION_ARGS_MESH)
{
    // Within radius...
    GReal X[GR_DIM];
    G.coord_embed(k, j, i, Loci::face1, X);
    if (X[1] > arg) { // If we are *outside* given radius
        FourVectors Dtmp;
        GRMHD::calc_4vecs(G, P(b), m_p, k, j, i, Loci::center, Dtmp);
        Real rho = P(m_p.RHO, b, k, j, i);
        Real Pg = (gam - 1.) * P(b, m_p.UU, k, j, i);
        Real Bmag = m::sqrt(dot(Dtmp.bcon, Dtmp.bcov));
        Real j_eht = rho*rho*rho/Pg/Pg * m::exp(-0.2 * m::pow(rho * rho / (Bmag * Pg * Pg), 1./3.));
        return j_eht;
    } else {
        return 0.;
    }
}

// Example of checking extra conditions before adding local results:
// sums total jet power only at exactly r=radius, for areas with sig > 1
// TODO version w/E&M power only.  Needs "calc_tensor_EM"
KOKKOS_INLINE_FUNCTION Real jet_lum(REDUCE_FUNCTION_ARGS_MESH)
{
    // At r = radius, i.e. if our faces span acreoss it...
    GReal X_f[GR_DIM]; GReal X_b[GR_DIM];
    G.coord_embed(k, j, i, Loci::face1, X_b);
    G.coord_embed(k, j, i+1, Loci::face1, X_f);
    if (X_f[1] > arg && X_b[1] < arg) { // If we are *at* given radius
        FourVectors Dtmp;
        Real T1[GR_DIM];
        GRMHD::calc_4vecs(G, P(b), m_p, k, j, i, Loci::center, Dtmp);
        Flux::calc_tensor(P(b), m_p, Dtmp, gam, k, j, i, X1DIR, T1);
        // If sigma > 1...
        if ((dot(Dtmp.bcon, Dtmp.bcov) / P(b, m_p.RHO, k, j, i)) > 1.) {
            // Energy flux, like at EH. 2D integral jacobian, so we have to take X1 off of auto-applied dV
            return -T1[X0DIR] / G.Dxc<1>(i);
        }
    }
    return 0.;
}

}