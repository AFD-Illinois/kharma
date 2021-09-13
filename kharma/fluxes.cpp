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
    FLAG("Getting conserved fluxes");
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

    FLAG("Got conserved fluxes");
    return TaskStatus::complete;
}

TaskStatus Flux::ApplyFluxes(MeshData<Real> *md, MeshData<Real> *mdudt)
{
    FLAG("Applying fluxes");
    // Pointers
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");

    // Pack variables
    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(md, prims_map); // We only need MHD prims
    auto U = md->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map); // But we need all conserved vars
    auto dUdt = mdudt->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}); // TODO can we use cons_map to ensure same?
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    // Get sizes
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    IndexRange block = IndexRange{0, U.GetDim(5) - 1};
    const int ndim = pmesh->ndim;
    const int nvar = U.GetDim(4);
    const auto& G = U.coords;

    // TODO we might actually benefit from scratch here, e.g. of dUdt before applying it
    const size_t total_scratch_bytes = 0;
    const int scratch_level = 0;

    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "apply_fluxes", pmb0->exec_space,
        total_scratch_bytes, scratch_level, block.s, block.e, kb.s, kb.e, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
            // This at least has a *chance* of being SIMD-fied, without invoking double kernel launch overhead.
            // Parthenon has a version of this kernel but either (1) no launch overhead or (2) loss of generality
            // mean this one is very slightly faster in my experience. YMMV.
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, ib.s, ib.e,
                    [&](const int& i) {
                        // Apply all existing fluxes
                            dUdt(b, p, k, j, i) = (U(b).flux(X1DIR, p, k, j, i) - U(b).flux(X1DIR, p, k, j, i+1)) / G(b).dx1v(i);
                            if (ndim > 1) dUdt(b, p, k, j, i) += (U(b).flux(X2DIR, p, k, j, i) - U(b).flux(X2DIR, p, k, j+1, i)) / G(b).dx2v(j);
                            if (ndim > 2) dUdt(b, p, k, j, i) += (U(b).flux(X3DIR, p, k, j, i) - U(b).flux(X3DIR, p, k+1, j, i)) / G(b).dx3v(k);
                    }
                );
            }
            parthenon::par_for_inner(member, ib.s, ib.e,
                [&](const int& i) {
                    // Then calculate and add the GRMHD source term
                    FourVectors Dtmp;
                    GRMHD::calc_4vecs(G(b), P(b), m_p, k, j, i, Loci::center, Dtmp);
                    GRMHD::add_source(G(b), P(b), m_p, Dtmp, gam, k, j, i, dUdt(b), m_u);
                }
            );
        }
    );

    FLAG("Applied");
    return TaskStatus::complete;
}
