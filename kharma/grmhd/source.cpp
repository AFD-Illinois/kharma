/* 
 *  File: source.cpp
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

#include "source.hpp"

#include "pack.hpp"

TaskStatus GRMHD::AddSource(MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt)
{
    FLAG("Adding GRMHD source");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    auto ib = pmb->cellbounds.GetBoundsI(domain);
    auto jb = pmb->cellbounds.GetBoundsJ(domain);
    auto kb = pmb->cellbounds.GetBoundsK(domain);
    const int ndim = pmb->pmy_mesh->ndim;

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc, prims_map);
    auto dUdt = GRMHD::PackMHDCons(dudt, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    size_t total_scratch_bytes = 0;
    int scratch_level = 0;

    pmb->par_for_outer("grmhd_source", total_scratch_bytes, scratch_level,
        kb.s, kb.e, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
            parthenon::par_for_inner(member, ib.s, ib.e,
                [&](const int& i) {
                    // Then calculate and add the GRMHD source term
                    FourVectors Dtmp;
                    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
                    GRMHD::add_source(G, P, m_p, Dtmp, gam, k, j, i, dUdt, m_u);
                }
            );
        }
    );

    FLAG("Added");
    return TaskStatus::complete;
}