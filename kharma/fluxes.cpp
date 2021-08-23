/* 
 *  File: fluxes.cpp
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

#include "fluxes.hpp"

#include <parthenon/parthenon.hpp>

// Package functions
#include "mhd_functions.hpp"
#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "b_flux_ct.hpp"

#include "debug.hpp"
#include "floors.hpp"
#include "reconstruction.hpp"
#include "source.hpp"

using namespace parthenon;

// GetFlux is in the header: it is a template on reconstruction scheme and flux direction

TaskStatus Flux::ApplyFluxes(MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt, const Real& dt)
{
    FLAG("Applying fluxes");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const int ndim = pmb->pmy_mesh->ndim;

    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars B_P = rc->Get("c.c.bulk.B_prim").data;

    PackIndexMap cons_map;
    auto U = rc->PackVariablesAndFluxes({Metadata::Conserved}, cons_map);
    auto dUdt = dudt->PackVariables({Metadata::Conserved});
    int nvar = U.GetDim(4);
    const int cons_start = cons_map["c.c.bulk.cons"].first;

    auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // TODO move wind to separate package/function?
    // bool wind_term = pmb->packages.Get("GRMHD")->Param<bool>("wind_term");
    // Real wind_n = pmb->packages.Get("GRMHD")->Param<Real>("wind_n");
    // Real wind_Tp = pmb->packages.Get("GRMHD")->Param<Real>("wind_Tp");
    // int wind_pow = pmb->packages.Get("GRMHD")->Param<int>("wind_pow");
    // Real wind_ramp_start = pmb->packages.Get("GRMHD")->Param<Real>("wind_ramp_start");
    // Real wind_ramp_end = pmb->packages.Get("GRMHD")->Param<Real>("wind_ramp_end");
    // Real current_wind_n = wind_n;
    // if (wind_ramp_end > 0.0) {
    //     current_wind_n = min((tm.time - wind_ramp_start) / (wind_ramp_end - wind_ramp_start), 1.0) * wind_n;
    // } else {
    //     current_wind_n = wind_n;
    // }

    // size_t total_scratch_bytes = 0;
    // int scratch_level = 0;

    // pmb->par_for_outer("apply_fluxes", total_scratch_bytes, scratch_level,
    //     ks, ke, js, je,
    //     KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
    //         parthenon::par_for_inner(member, is, ie,
    //             [&](const int& i) {

    pmb->par_for("apply_fluxes", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
                    // Apply all existing fluxes
                    for (int p=0; p < nvar; ++p) {
                        dUdt(p, k, j, i) = (U.flux(X1DIR, p, k, j, i) - U.flux(X1DIR, p, k, j, i+1)) / G.dx1v(i);
                        if (ndim > 1) dUdt(p, k, j, i) += (U.flux(X2DIR, p, k, j, i) - U.flux(X2DIR, p, k, j+1, i)) / G.dx2v(j);
                        if (ndim > 2) dUdt(p, k, j, i) += (U.flux(X3DIR, p, k, j, i) - U.flux(X3DIR, p, k+1, j, i)) / G.dx3v(k);
                    }

                    // Then calculate and add the source term(s)
                    FourVectors Dtmp;
                    //Real dU[NPRIM] = {0};
                    GRMHD::calc_4vecs(G, P, B_P, k, j, i, Loci::center, Dtmp);
                    GRMHD::get_source(G, P, Dtmp, gam, k, j, i, cons_start, dUdt);

                    // if (wind_term) {
                    //     GRMHD::add_wind(G, gam, k, j, i, current_wind_n, wind_pow, wind_Tp, dU);
                    // }
            //     }
            // );
        }
    );
    FLAG("Applied");

    return TaskStatus::complete;
}
