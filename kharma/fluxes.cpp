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

#include "source.hpp"

using namespace parthenon;

// GetFlux is in the header, as it is templated on reconstruction scheme and flux direction
// That's also why we don't have any extra includes in here

TaskStatus Flux::PrimToFlux(MeshBlockData<Real> *rc, IndexDomain domain)
{
    Flag(rc, "Getting conserved fluxes");
    // Pointers
    auto pmb = rc->GetBlockPointer();
    // Options
    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");
    auto pkgs = pmb->packages.AllPackages();
    const bool flux_ct = pkgs.count("B_FluxCT");
    const bool b_cd = pkgs.count("B_CD");
    const bool use_electrons = pkgs.count("Electrons");
    MetadataFlag isPrimitive = pars.Get<MetadataFlag>("PrimitiveFlag");

    // Pack variables
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables({isPrimitive}, prims_map);
    const auto& U = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    IndexRange ib = rc->GetBoundsI(domain);
    IndexRange jb = rc->GetBoundsJ(domain);
    IndexRange kb = rc->GetBoundsK(domain);

    const auto& G = pmb->coords;

    pmb->par_for("P_to_U", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
            if (flux_ct) B_FluxCT::p_to_u(G, P, m_p, k, j, i, U, m_u);
            else if (b_cd) B_CD::p_to_u(G, P, m_p, k, j, i, U, m_u);
            if (use_electrons) Electrons::p_to_u(G, P, m_p, k, j, i, U, m_u);
        }
    );

    Flag(rc, "Got conserved fluxes");
    return TaskStatus::complete;
}
