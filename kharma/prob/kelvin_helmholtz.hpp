/*
 *  File: kelvin_helmholtz.hpp
 *
 *  BSD 3-Clause License
 *
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include "decs.hpp"
#include "eos.hpp"

/*
 * Kelvin-Helmholtz instability problem
 * Follows initial conditions from Lecoanet et al. 2015,
 * MNRAS 455, 4274.
 */
void InitializeKelvinHelmholtz(MeshBlock *pmb, const GRCoordinates &G,
                               const GridVars &P, Real tscale=0.01) {

    /* follows notation of Lecoanet et al. eq. 8 et seq. */
    double rho0 = 1.;
    double Drho = 0.1;
    double P0 = 10.;
    double uflow = 1.;
    double a = 0.05;
    double sigma = 0.2;
    double A = 0.01;
    double z1 = 0.5;
    double z2 = 1.5;

    Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    IndexDomain domain = IndexDomain::entire;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("kh_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            GReal X[GR_DIM];
            G.coord_embed(i, j, k, Loci::center, X);

            // Lecoanet's x <-> x1; z <-> x2
            GReal x = X[1];
            GReal z = X[2];

            P(prims::rho, k, j, i) =
                rho0 + Drho * 0.5 * (tanh((z - z1) / a) - tanh((z - z2) / a));
            P(prims::u, k, j, i) = P0 / (gam - 1.);
            P(prims::u1, k, j, i) = uflow * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1.);
            P(prims::u2, k, j, i) = A * sin(2. * M_PI * x) *
                        (exp(-(z - z1) * (z - z1) / (sigma * sigma)) +
                        exp(-(z - z2) * (z - z2) / (sigma * sigma)));
            P(prims::u3, k, j, i) = 0;
        }
    );
    // Rescale primitive velocities by tscale, and internal energy by the square.
    pmb->par_for("kh_renorm", prims::u, NPRIM-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_VARS {
            P(p, k, j, i) *= tscale * (p == prims::u ? tscale : 1);
        }
    );
}
