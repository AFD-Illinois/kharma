/* 
 *  File: bondi.hpp
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
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "prob_common.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

/**
 * Initialization of a Bondi problem with specified sonic point and BH accretion rate mdot
 * TODO mdot and rs are redundant and should be merged into one parameter
 */
TaskStatus InitializeBondi(MeshBlockData<Real> *rc, ParameterInput *pin);

/**
 * Set all values on a given domain to the Bondi inflow analytic steady-state solution
 * 
 * Used for initialization and boundary conditions
 */
TaskStatus SetBondi(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);

/**
 * Supporting functions for Bondi flow calculations
 * 
 * Adapted from M. Chandra
 */
KOKKOS_FORCEINLINE_FUNCTION Real get_Tfunc(const Real T, const GReal r, const Real C1, const Real C2, const Real n)
{
    return pow(1. + (1. + n) * T, 2.) * (1. - 2. / r + pow(C1 / pow(r,2) / pow(T, n), 2.)) - C2;
}
KOKKOS_FORCEINLINE_FUNCTION Real get_T(const GReal r, const Real C1, const Real C2, const Real n)
{
    Real rtol = 1.e-12;
    Real ftol = 1.e-14;
    Real Tmin = 0.6 * (sqrt(C2) - 1.) / (n + 1);
    Real Tmax = pow(C1 * sqrt(2. / pow(r,3)), 1. / n);

    Real f0, f1, fh;
    Real T0, T1, Th;
    T0 = Tmin;
    f0 = get_Tfunc(T0, r, C1, C2, n);
    T1 = Tmax;
    f1 = get_Tfunc(T1, r, C1, C2, n);
    if (f0 * f1 > 0) return -1;

    Th = (f1 * T0 - f0 * T1) / (f1 - f0);
    fh = get_Tfunc(Th, r, C1, C2, n);
    Real epsT = rtol * (Tmin + Tmax);
    while (fabs(Th - T0) > epsT && fabs(Th - T1) > epsT && fabs(fh) > ftol)
    {
        if (fh * f0 < 0.) {
            T0 = Th;
            f0 = fh;
        } else {
            T1 = Th;
            f1 = fh;
        }

        Th = (f1 * T0 - f0 * T1) / (f1 - f0);
        fh = get_Tfunc(Th, r, C1, C2, n);
    }

    return Th;
}

/**
 * Get the Bondi solution at a particular zone
 * Note this assumes that there are ghost zones!
 * 
 * TODO could put this back into SetBondi
 */
KOKKOS_FORCEINLINE_FUNCTION void get_prim_bondi(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
                                           const Real& gam, const SphBLCoords& bl,  const SphKSCoords& ks, 
                                           const Real mdot, const Real rs, const int& k, const int& j, const int& i)
{
    // Solution constants
    // Ideally these could be cached but preformance isn't an issue here
    Real n = 1. / (gam - 1.);
    Real uc = sqrt(mdot / (2. * rs));
    Real Vc = -sqrt(pow(uc, 2) / (1. - 3. * pow(uc, 2)));
    Real Tc = -n * pow(Vc, 2) / ((n + 1.) * (n * pow(Vc, 2) - 1.));
    Real C1 = uc * pow(rs, 2) * pow(Tc, n);
    Real C2 = pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + pow(C1, 2) / (pow(rs, 4) * pow(Tc, 2 * n)));

    GReal Xnative[GR_DIM], Xembed[GR_DIM];
    G.coord(k, j, i, Loci::center, Xnative);
    G.coord_embed(k, j, i, Loci::center, Xembed);
    GReal r = Xembed[1];
    // Unless we're doing a Schwarzchild problem & comparing solutions,
    // be a little cautious about initializing the Ergosphere zones
    if (ks.a > 0.1 && r < 2) return;

    Real T = get_T(r, C1, C2, n);
    Real ur = -C1 / (pow(T, n) * pow(r, 2));
    Real rho = pow(T, n);
    Real u = rho * T * n;

    // Set u^t to make u^r a 4-vector
    Real ucon_bl[GR_DIM] = {0, ur, 0, 0};
    Real gcov_bl[GR_DIM][GR_DIM];
    bl.gcov_embed(Xembed, gcov_bl);
    set_ut(gcov_bl, ucon_bl);

    // Then transform that 4-vector to KS, then to native
    Real ucon_ks[GR_DIM], ucon_mks[GR_DIM];
    ks.vec_from_bl(Xembed, ucon_bl, ucon_ks);
    coords.con_vec_to_native(Xnative, ucon_ks, ucon_mks);

    // Convert native 4-vector to primitive u-twiddle, see Gammie '04
    Real gcon[GR_DIM][GR_DIM], u_prim[NVEC];
    G.gcon(Loci::center, j, i, gcon);
    fourvel_to_prim(gcon, ucon_mks, u_prim);

    P(m_p.RHO, k, j, i) = rho;
    P(m_p.UU, k, j, i) = u;
    P(m_p.U1, k, j, i) = u_prim[0];
    P(m_p.U2, k, j, i) = u_prim[1];
    P(m_p.U3, k, j, i) = u_prim[2];
}
