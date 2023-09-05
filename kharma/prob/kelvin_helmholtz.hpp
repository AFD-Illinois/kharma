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
#include "domain.hpp"
#include "types.hpp"

#include "b_ct.hpp"

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
    const Real amp = pin->GetOrAddReal("kelvin_helmholtz", "amp", 0.01);
    const Real z1 = pin->GetOrAddReal("kelvin_helmholtz", "z1", 0.5);
    const Real z2 = pin->GetOrAddReal("kelvin_helmholtz", "z2", 1.5);
    const Real added_b = pin->GetOrAddReal("kelvin_helmholtz", "added_b", 0.0);

    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    IndexDomain domain = IndexDomain::entire;
    IndexRange3 b = KDomain::GetRange(rc, domain, 0, 0);
    pmb->par_for("kh_init", b.ks, b.ke, b.js, b.je, b.is, b.ie,
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
            uvec(1, k, j, i) = amp * sin(2. * M_PI * x) *
                        (m::exp(-(z - z1) * (z - z1) / (sigma * sigma)) +
                        m::exp(-(z - z2) * (z - z2) / (sigma * sigma)));
            uvec(2, k, j, i) = 0;
        }
    );

    // if (pmb->packages.AllPackages().count("B_CT")) {
    //     auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
    //     // Halo one zone right for faces
    //     // We don't need any more than that, since curls never take d1dx1
    //     IndexRange3 bA = KDomain::GetRange(rc, IndexDomain::entire, 0, 0);
    //     IndexSize3 s = KDomain::GetBlockSize(rc);
    //     GridVector A("A", NVEC, s.n3, s.n2, s.n1);
    //     pmb->par_for("ot_A", bA.ks, bA.ke, bA.js, bA.je, bA.is, bA.ie,
    //         KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
    //             Real Xembed[GR_DIM];
    //             G.coord(k, j, i, Loci::corner, Xembed);
    //             A(V3, k, j, i)  = added_b * (Xembed[1]/G.Dxc<1>(i) + Xembed[2]/G.Dxc<2>(j)) * tscale;
    //         }
    //     );
    //     // This fills a couple zones outside the exact interior with bad data
    //     IndexRange3 bB = KDomain::GetRange(rc, domain, 0, -1);
    //     pmb->par_for("ot_B", bB.ks, bB.ke, bB.js, bB.je, bB.is, bB.ie,
    //         KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
    //             B_CT::curl_2D(G, A, B_Uf, k, j, i);
    //         }
    //     );
    //     B_CT::BlockUtoP(rc.get(), IndexDomain::entire, false);
    //     double max_divb = B_CT::BlockMaxDivB(rc.get());
    //     std::cout << "Block max DivB: " << max_divb << std::endl;

    // } else if (pmb->packages.AllPackages().count("B_FluxCT") ||
    //            pmb->packages.AllPackages().count("B_CD")) {
    //     GridVector B_P = rc->Get("prims.B").data;
    //     pmb->par_for("ot_B", b.ks, b.ke, b.js, b.je, b.is, b.ie,
    //         KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
    //             Real X[GR_DIM];
    //             G.coord(k, j, i, Loci::center, X);
    //             B_P(V1, k, j, i) = added_b * tscale;
    //             B_P(V2, k, j, i) = added_b * tscale;
    //             B_P(V3, k, j, i) = 0.;
    //         }
    //     );
    //     B_FluxCT::BlockPtoU(rc.get(), IndexDomain::entire, false);
    // }

    // Rescale primitive velocities by tscale, and internal energy by the square.
    pmb->par_for("kh_renorm", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            u(k, j, i) *= tscale * tscale;
            VLOOP uvec(v, k, j, i) *= tscale;
            //VLOOP B_P(v, k, j, i) *= tscale; //already done
        }
    );

    return TaskStatus::complete;
}
