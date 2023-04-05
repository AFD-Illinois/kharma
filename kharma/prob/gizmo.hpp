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

#include "bondi.hpp"
#include "gr_coordinates.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "coordinate_utils.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

/**
 * Initialize a Bondi problem over the domain
 */
TaskStatus InitializeGIZMO(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin);

/**
 * Set all values on a given domain to the Bondi inflow analytic steady-state solution
 * 
 * Used for initialization and boundary conditions
 */
TaskStatus SetGIZMO(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse=false);

KOKKOS_INLINE_FUNCTION void XtoindexGIZMO(const GReal XG[GR_DIM],
                                    const GridScalar& rarr, const int length, int& i, GReal& del)
{
    Real dx2, dx2_min;
    dx2_min = m::pow(XG[1]-rarr(0),2); //100000.; //arbitrarily large number

    i = 0; // initialize

    for (int itemp = 0; itemp < length; itemp++) {
        if (rarr(itemp) < XG[1]) { // only look for smaller side
            dx2 = m::pow(XG[1] - rarr(itemp), 2);

            // simplest interpolation (Hyerin 07/26/22)
            if (dx2 < dx2_min){
                dx2_min = dx2;
                i = itemp;
            }
        }
    }
    
    // interpolation (11/14/2022) TODO: write a case where indices hit the boundaries of the data file
    del = (XG[1]-rarr(i))/(rarr(i+1)-rarr(i));

    if (m::abs(dx2_min/m::pow(XG[1],2))>1.e-8) printf("XtoindexGizmo: dx2 pretty large = %g at r= %g \n",dx2_min, XG[1]);
}
/**
 * Get the GIZMO output values at a particular zone
 * Note this assumes that there are ghost zones!
 */
KOKKOS_INLINE_FUNCTION void get_prim_gizmo_shell(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
                                           const Real& gam,
                                           const Real rin_init, const Real rs, Real vacuum_rho, Real vacuum_u_over_rho,
                                           const GridScalar& rarr, const GridScalar& rhoarr, const GridScalar& Tarr, const GridScalar& vrarr, const int length,
                                           const int& k, const int& j, const int& i)
{
    // Solution constants for velocity prescriptions
    // Ideally these could be cached but preformance isn't an issue here
    Real mdot = 1.; // mdot and rs defined arbitrarily
    Real n = 1. / (gam - 1.);
    Real uc = sqrt(mdot / (2. * rs));
    Real Vc = -sqrt(pow(uc, 2) / (1. - 3. * pow(uc, 2)));
    Real Tc = -n * pow(Vc, 2) / ((n + 1.) * (n * pow(Vc, 2) - 1.));
    Real C1 = uc * pow(rs, 2) * pow(Tc, n);
    Real C2 = pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + pow(C1, 2) / (pow(rs, 4) * pow(Tc, 2 * n)));

    //Real rs = 1./sqrt(T); //1000.;
    GReal Xnative[GR_DIM], Xembed[GR_DIM];
    G.coord(k, j, i, Loci::center, Xnative);
    G.coord_embed(k, j, i, Loci::center, Xembed);
    GReal r = Xembed[1];

    // Use Bondi infall velocity
    Real rho, u;
    Real T = get_T(r, C1, C2, n, rs);
    Real ucon_bl[GR_DIM] = {0};
    if (r < rin_init * 0.9){
        // Vacuum values for interior
        rho = vacuum_rho;
        u = vacuum_rho * vacuum_u_over_rho;
        ucon_bl[1] = -C1 / (pow(T, n) * pow(r, 2));
    } else {
        // linear interpolation
        int itemp; GReal del;
        XtoindexGIZMO(Xembed, rarr, length, itemp, del);
        if (del < 0 ) { // when r is smaller than GIZMO's range
            del = 0; // just copy over the smallest r values
        }
        rho = rhoarr(itemp) * (1.-del) + rhoarr(itemp+1) * del;
        u = rho * (Tarr(itemp) * (1.-del) + Tarr(itemp+1) * del)*n;
        ucon_bl[1] = 0.;
    }

    // Set u^t and transform to native coordinates
    GReal ucon_native[GR_DIM];
    G.coords.bl_fourvel_to_native(Xnative, ucon_bl, ucon_native);

    // Convert native 4-vector to primitive u-twiddle, see Gammie '04
    Real gcon[GR_DIM][GR_DIM], u_prim[NVEC];
    G.gcon(Loci::center, j, i, gcon);
    fourvel_to_prim(gcon, ucon_native, u_prim);

    P(m_p.RHO, k, j, i) = rho;
    P(m_p.UU, k, j, i) = u;
    P(m_p.U1, k, j, i) = u_prim[0];
    P(m_p.U2, k, j, i) = u_prim[1];
    P(m_p.U3, k, j, i) = u_prim[2];
}
