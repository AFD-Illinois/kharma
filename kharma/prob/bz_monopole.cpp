/* 
 *  File: bz_monopole.cpp
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

#include "bz_monopole.hpp"

#include "prob_common.hpp"
#include "types.hpp"

#include <random>
#include "Kokkos_Random.hpp"

TaskStatus InitializeBZMonopole(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    Flag(rc, "Initializing BZ monopole problem");

    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;

    Real bsq_o_rho_max = pin->GetOrAddReal("floors", "bsq_over_rho_max", 1.e2);
    Real rho_min_limit = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-6);
    Real u_min_limit = pin->GetOrAddReal("floors", "u_min_geom", 1.e-8);

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    const auto& G = pmb->coords;
    const GReal a = G.coords.get_a();

    if (pmb->gid == 0 && pmb->packages.Get("Globals")->Param<int>("verbose") > 0) {
        std::cout << "Initializing BZ monopole." << std::endl;
    }

    pmb->par_for("fm_torus_init", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            GReal Xembed[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, Xembed);
            GReal r = Xembed[1];

            GReal r_horizon = 1. + m::sqrt(1. - a*a);
            GReal r_char = 10. * r_horizon;

            Real trho = rho_min_limit + (r / r_char) / m::pow(r, 4.) / bsq_o_rho_max;
            Real tu = u_min_limit + (r / r_char) / m::pow(r, 4.) / bsq_o_rho_max;

            rho(k, j, i) = trho;
            u(k, j, i) = tu;
            uvec(0, k, j, i) = 0.;
            uvec(1, k, j, i) = 0.;
            uvec(2, k, j, i) = 0.;
        }
    );

    Flag(rc, "Initialized");
    return TaskStatus::complete;
}

