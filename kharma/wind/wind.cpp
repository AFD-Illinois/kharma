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
    Real wind_n = pin->GetOrAddReal("wind", "ne", 2.e-4);
    params.Add("wind_n", wind_n);
    Real wind_Tp = pin->GetOrAddReal("wind", "Tp", 10.0);
    params.Add("wind_Tp", wind_Tp);
    int wind_pow = pin->GetOrAddInteger("wind", "pow", 4);
    params.Add("wind_pow", wind_pow);
    Real wind_ramp_start = pin->GetOrAddReal("wind", "ramp_start", 0.0);
    params.Add("wind_ramp_start", wind_ramp_start);
    Real wind_ramp_end = pin->GetOrAddReal("wind", "ramp_end", 0.0);
    params.Add("wind_ramp_end", wind_ramp_end);

    return pkg;
}

TaskStatus Wind::AddSource(MeshData<Real> *mdudt)
{
    FLAG("Adding wind");
    // Pointers
    auto pmesh = mdudt->GetMeshPointer();
    auto pmb0 = mdudt->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& gpars = pmb0->packages.Get("GRMHD")->AllParams();
    const auto& pars = pmb0->packages.Get("Wind")->AllParams();
    const auto& globals = pmb0->packages.Get("Globals")->AllParams();
    const Real gam = gpars.Get<Real>("gamma");
    const Real wind_n = pars.Get<Real>("wind_n");
    const Real wind_Tp = pars.Get<Real>("wind_Tp");
    const int wind_pow = pars.Get<int>("wind_pow");
    const Real wind_ramp_start = pars.Get<Real>("wind_ramp_start");
    const Real wind_ramp_end = pars.Get<Real>("wind_ramp_end");
    const Real time = globals.Get<Real>("time");

    // Pack variables
    PackIndexMap cons_map;
    auto dUdt = mdudt->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true);
    // Get sizes
    const IndexRange ib = mdudt->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = mdudt->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = mdudt->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, dUdt.GetDim(5) - 1};

    const auto& G = dUdt.coords;

    // Set the wind via linear ramp-up with time, if enabled
    const Real current_wind_n = (wind_ramp_end > 0.0) ? min((time - wind_ramp_start) / (wind_ramp_end - wind_ramp_start), 1.0) * wind_n : wind_n;

    pmb0->par_for("add_wind", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D {
            Wind::add_wind(G(b), gam, k, j, i, current_wind_n, wind_pow, wind_Tp, dUdt(b), m_u);
        }
    );

    FLAG("Added");
    return TaskStatus::complete;
}