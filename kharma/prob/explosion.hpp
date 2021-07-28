/* 
 *  File: explosion.hpp
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

#include <complex>

#include "decs.hpp"


using namespace std::literals::complex_literals;
using namespace std;
using namespace parthenon;

/**
 * Initialization of the strong cylindrical explosion of Komissarov 1999 section 7.3
 * 
 * Note the problem setup assumes gamma=4/3
 * 
 * Originally run on 2D Cartesian domain -6.0, 6.0 with a 200x200 grid, to tlim=4.0
 */
void InitializeExplosion(MeshBlock *pmb, GRCoordinates G, GridVars P)
{
    Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    Real u_out = 3.e-5 / (gam-1);
    Real rho_out = 1.e-4;

    Real u_in = 1.0 / (gam-1);
    Real rho_in = 1.e-2;

    // One buffer zone of linear decline, r_in -> r_out
    // Exponential decline inside here i.e. linear in logspace
    GReal r_in = 0.8;
    GReal r_out = 1.0;
    // Circle center
    GReal xoff = 0.0;
    GReal yoff = 0.0;

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    pmb->par_for("explosion_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);
            GReal rx = X[1] - xoff;
            GReal ry = X[2] - yoff;
            Real r = sqrt(rx*rx + ry*ry);

            if (r < r_in) {
                P(prims::rho, k, j, i) = rho_in;
                P(prims::u, k, j, i) = u_in;
            } else if (r >= r_in && r <= r_out) {
                Real ramp = (r_out - r) / (r_out - r_in);

                // P(prims::rho, k, j, i) = rho_out + ramp * (rho_in - rho_out);
                // P(prims::u, k, j, i) = u_in + ramp * (u_in - u_out);

                Real lrho_out = log(rho_out);
                Real lrho_in = log(rho_in);
                Real lu_out = log(u_out);
                Real lu_in = log(u_in);
                P(prims::rho, k, j, i) = exp(lrho_out + ramp * (lrho_in - lrho_out));
                P(prims::u, k, j, i) = exp(lu_out + ramp * (lu_in - lu_out));
            } else {
                P(prims::rho, k, j, i) = rho_out;
                P(prims::u, k, j, i) = u_out;
            }

            P(prims::u1, k, j, i) = 0.;
            P(prims::u2, k, j, i) = 0.;
            P(prims::u3, k, j, i) = 0.;
        }
    );
}
