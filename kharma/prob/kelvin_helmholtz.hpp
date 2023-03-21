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

#include <parthenon/parthenon.hpp>

/*
 * Kelvin-Helmholtz instability problem
 * Follows initial conditions from Lecoanet et al. 2015,
 * MNRAS 455, 4274.
 */
TaskStatus InitializeKelvinHelmholtz(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;

    // follows notation of Lecoanet et al. eq. 8 et seq.
    const Real tscale = pin->GetOrAddReal("kelvin_helmholtz", "tscale", 0.05);
    const Real rho0 = pin->GetOrAddReal("kelvin_helmholtz", "rho0", 1.);
    const Real Drho = pin->GetOrAddReal("kelvin_helmholtz", "Drho", 0.1);
    const Real P0 = pin->GetOrAddReal("kelvin_helmholtz", "P0", 10.);
    const Real uflow = pin->GetOrAddReal("kelvin_helmholtz", "uflow", 1.);
    const Real a = pin->GetOrAddReal("kelvin_helmholtz", "a", 0.05);
    const Real sigma = pin->GetOrAddReal("kelvin_helmholtz", "sigma", 0.2);
    const Real A = pin->GetOrAddReal("kelvin_helmholtz", "A", 0.01);
    const Real z1 = pin->GetOrAddReal("kelvin_helmholtz", "z1", 0.5);
    const Real z2 = pin->GetOrAddReal("kelvin_helmholtz", "z2", 1.5);

    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("kh_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            GReal X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);

            // Lecoanet's x <-> x1; z <-> x2
            GReal x = X[1];
            GReal z = X[2];

            rho(k, j, i) =
                rho0 + Drho * 0.5 * (tanh((z - z1) / a) - tanh((z - z2) / a));
            u(k, j, i) = P0 / (gam - 1.);
            uvec(0, k, j, i) = uflow * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1.);
            uvec(1, k, j, i) = A * sin(2. * M_PI * x) *
                        (m::exp(-(z - z1) * (z - z1) / (sigma * sigma)) +
                        m::exp(-(z - z2) * (z - z2) / (sigma * sigma)));
            uvec(2, k, j, i) = 0;
        }
    );
    // Rescale primitive velocities by tscale, and internal energy by the square.
    pmb->par_for("kh_renorm", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            u(k, j, i) *= tscale * tscale;
            VLOOP uvec(v, k, j, i) *= tscale;
        }
    );

    return TaskStatus::complete;
}
