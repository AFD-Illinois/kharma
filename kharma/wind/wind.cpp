/* 
 *  File: wind.cpp
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

#include "wind.hpp"

std::shared_ptr<StateDescriptor> Wind::Initialize(ParameterInput *pin)
{
    auto pkg = std::make_shared<StateDescriptor>("Wind");
    Params &params = pkg->AllParams();

    // Wind term in funnel
    bool wind_term = pin->GetOrAddBoolean("wind", "on", false);
    params.Add("wind_term", wind_term);
    Real wind_n = pin->GetOrAddReal("wind", "ne", 2.e-4);
    params.Add("wind_n", wind_n);
    Real wind_Tp = pin->GetOrAddReal("wind", "Tp", 10.);
    params.Add("wind_Tp", wind_Tp);
    int wind_pow = pin->GetOrAddInteger("wind", "pow", 4);
    params.Add("wind_pow", wind_pow);
    Real wind_ramp_start = pin->GetOrAddReal("wind", "ramp_start", 0.);
    params.Add("wind_ramp_start", wind_ramp_start);
    Real wind_ramp_end = pin->GetOrAddReal("wind", "ramp_end", 0.);
    params.Add("wind_ramp_end", wind_ramp_end);

    return pkg;
}

TaskStatus Wind::AddWind(MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt)
{
    FLAG("Adding wind");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const int ndim = pmb->pmy_mesh->ndim;

    PackIndexMap cons_map;
    auto dUdt = dudt->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true);

    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    Real wind_n = pmb->packages.Get("Wind")->Param<Real>("wind_n");
    Real wind_Tp = pmb->packages.Get("Wind")->Param<Real>("wind_Tp");
    int wind_pow = pmb->packages.Get("Wind")->Param<int>("wind_pow");
    Real wind_ramp_start = pmb->packages.Get("Wind")->Param<Real>("wind_ramp_start");
    Real wind_ramp_end = pmb->packages.Get("Wind")->Param<Real>("wind_ramp_end");
    Real current_wind_n = wind_n;
    // TODO pass simtime to this fn specifically!
    // if (wind_ramp_end > 0.0) {
    //     current_wind_n = min((tm.time - wind_ramp_start) / (wind_ramp_end - wind_ramp_start), 1.0) * wind_n;
    // } else {
    //     current_wind_n = wind_n;
    // }

    size_t total_scratch_bytes = 0;
    int scratch_level = 0;

    pmb->par_for_outer("apply_fluxes", total_scratch_bytes, scratch_level,
        ks, ke, js, je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
            parthenon::par_for_inner(member, is, ie,
                [&](const int& i) {
                    Wind::add_wind(G, gam, k, j, i, current_wind_n, wind_pow, wind_Tp, dUdt, m_u);
                }
            );
        }
    );

    FLAG("Added");
    return TaskStatus::complete;
}