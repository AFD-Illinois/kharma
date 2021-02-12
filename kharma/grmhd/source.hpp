/* 
 *  File: source.hpp
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
#include "mhd_functions.hpp"

namespace GRMHD
{

/**
 * Inline function to get the source term in the GRMHD equations T^\mu_nu \Gamma^\nu_\lam\mu
 * So named because it clobbers input dUdt!
 */
KOKKOS_INLINE_FUNCTION void get_source(const GRCoordinates &G, const GridVars P, const FourVectors& D,
                      const EOS* eos, const int& k, const int& j, const int& i, Real dUdt[NPRIM])
{
    // Get T^mu_nu
    Real mhd[GR_DIM][GR_DIM];
    Real rho = P(prims::rho, k, j, i);
    Real u = P(prims::u, k, j, i);
    Real pgas = eos->p(rho, u);
    DLOOP1 GRMHD::calc_tensor(rho, u, pgas, D, mu, mhd[mu]);

    // Initialize our addition [+U, +U1, +U2, +U3]
    Real du_loc[GR_DIM] = {0};
    // Contract mhd stress tensor with connection, and multiply by metric dterminant
    DLOOP3 du_loc[lam] += mhd[mu][nu] * G.conn(j, i, nu, lam, mu);
    // Multiply by the metric determinant and add to the current value
    PLOOP dUdt[p] = 0.;
    DLOOP1 dUdt[prims::u + mu] += du_loc[mu] * G.gdet(Loci::center, j, i);
}

/**
 * Function to add a "wind" source term, in addition to the usual GRMHD coordinate source term
 * 
 * This is a purely geometry-based particle source concentrated near the poles,
 * which prevents too many fixups & floor hits on the coordinate pole.
 * It has proven effective at eliminating the use of floors in 2D, but less so in 3D/MAD simulations
 */
KOKKOS_INLINE_FUNCTION void add_wind(const GRCoordinates &G,
                      const EOS* eos, const int& k, const int& j, const int& i,
                      const Real& n, const int& power, const Real& Tp, Real dUdt[NPRIM]) {
    Real dP[NPRIM] = {0}, dUw[NPRIM] = {0};

    // Need coordinates to evaluate particle addtn rate
    // Note that makes the wind spherical-only, TODO ensure this
    GReal Xembed[GR_DIM];
    G.coord_embed(k, j, i, Loci::center, Xembed);
    GReal r = Xembed[1], th = Xembed[2];

    // Particle addition rate: concentrate at poles & center
    // TODO poles only w/e.g. cos2?
    Real drhopdt = n * pow(cos(th), power) / pow(1. + r * r, 2);
    dP[prims::rho] = drhopdt;

    dP[prims::u] = drhopdt * Tp * 3.;

    // Leave everything else: we're inserting only fluid in normal observer frame

    // Add plasma to the T^t_a component of the stress-energy tensor
    // Notice that U already contains a factor of sqrt{-g}
    Real dB_P[NVEC] = {0};
    GRMHD::p_to_u(G, dP, dB_P, eos, k, j, i, dUw);

    PLOOP dUdt[p] += dUw[p];
}

} // namespace GRMHD