/* 
 *  File: dirichlet.cpp
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

#include "dirichlet.hpp"

#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

void KBoundaries::DirichletImpl(std::shared_ptr<MeshBlockData<Real>> &rc, BoundaryFace bface, bool coarse)
{
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // Get all ghosts, minus those in the B_Cleanup package if it is present
    using FC = Metadata::FlagCollection;
    FC main_ghosts = pmb->packages.AllPackages().count("B_Cleanup")
                            ? FC({Metadata::FillGhost}) - FC({Metadata::GetUserFlag("B_Cleanup")})
                            : FC({Metadata::FillGhost});
    PackIndexMap ghostmap;
    auto q = rc->PackVariables(main_ghosts, ghostmap, coarse);
    const int q_index = ghostmap["prims.q"].first;
    auto bound = rc->Get("bounds." + BoundaryName(bface)).data;

    if (q.GetDim(4) != bound.GetDim(4)) {
        std::cerr << "Boundary cache mismatch! " << bound.GetDim(4) << " vs " << q.GetDim(4) << std::endl;
        std::cerr << "Variables with ghost zones:" << std::endl;
        ghostmap.print();
    }

    const IndexRange vars = IndexRange{0, q.GetDim(4) - 1};
    const bool right = !BoundaryIsInner(bface);

    // Subtract off the starting index if we're on the right
    const auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const int dir = BoundaryDirection(bface);
    const int ie = (dir == 1) ? bounds.ie(IndexDomain::interior) + 1 : 0;
    const int je = (dir == 2) ? bounds.je(IndexDomain::interior) + 1 : 0;
    const int ke = (dir == 3) ? bounds.ke(IndexDomain::interior) + 1 : 0;

    const auto &G = pmb->coords;

    // printf("Freezing bounds:\n");
    const auto domain = BoundaryDomain(bface);
    pmb->par_for_bndry(
        "dirichlet_boundary", vars, domain, CC, coarse,
        KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
            if (right) {
                q(p, k, j, i) = bound(p, k - ke, j - je, i - ie);
            } else {
                q(p, k, j, i) = bound(p, k, j, i);
            }
            // if (p == q_index) printf("%g ", q(p, k, j, i));
        }
    );
    // Kokkos::fence();
    // printf("\n\n");
}

void KBoundaries::FreezeDirichlet(std::shared_ptr<MeshData<Real>> &md)
{
    // For each face...
    for (int i=0; i < BOUNDARY_NFACES; i++) {
        BoundaryFace bface = (BoundaryFace) i;
        auto bname = BoundaryName(bface);
        auto pmesh = md->GetMeshPointer();
        // ...if this boundary is dirichlet...
        if (pmesh->packages.Get("Boundaries")->Param<std::string>(bname) == "dirichlet") {
            //std::cout << "Freezing dirichlet " << bname << " on mesh." << std::endl;
            // ...on all blocks...
            for (int i=0; i < md->NumBlocks(); i++) {
                auto rc = md->GetBlockData(i).get();
                std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
                auto domain = BoundaryDomain(bface);
                // Set whatever is in that domain as the Dirichlet bound
                SetDomainDirichlet(rc, domain, false);
            }
        }
    }
}
void KBoundaries::FreezeDirichletBlock(MeshBlockData<Real> *rc)
{
    // For each face...
    for (int i=0; i < BOUNDARY_NFACES; i++) {
        BoundaryFace bface = (BoundaryFace) i;
        auto bname = BoundaryName(bface);
        auto pmb = rc->GetBlockPointer();
        // ...if this boundary is dirichlet...
        if (pmb->packages.Get("Boundaries")->Param<std::string>(bname) == "dirichlet") {
            //std::cout << "Freezing dirichlet " << bname << " on block." << std::endl;
            auto domain = BoundaryDomain(bface);
            // Set whatever is in that domain as the Dirichlet bound
            SetDomainDirichlet(rc, domain, false);
        }
    }
}

void KBoundaries::SetDomainDirichlet(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const BoundaryFace bface = BoundaryFaceOf(domain);

    using FC = Metadata::FlagCollection;
    FC main_ghosts = pmb->packages.AllPackages().count("B_Cleanup")
                            ? FC({Metadata::FillGhost}) - FC({Metadata::GetUserFlag("B_Cleanup")})
                            : FC({Metadata::FillGhost});
    auto q = rc->PackVariables(main_ghosts, coarse);
    auto bound = rc->Get("bounds." + BoundaryName(bface)).data;

    // TODO error?
    if (q.GetDim(4) != bound.GetDim(4)) {
        std::cerr << "Dirichlet boundary cache mismatch! " << bound.GetDim(4) << " vs " << q.GetDim(4) << std::endl;
    }

    const IndexRange vars = IndexRange{0, q.GetDim(4) - 1};
    const bool right = !BoundaryIsInner(domain);

    // Subtract off the starting index if we're on the right
    const auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const int dir = BoundaryDirection(bface);
    const int ie = (dir == 1) ? bounds.ie(IndexDomain::interior) + 1 : 0;
    const int je = (dir == 2) ? bounds.je(IndexDomain::interior) + 1 : 0;
    const int ke = (dir == 3) ? bounds.ke(IndexDomain::interior) + 1 : 0;

    const auto &G = pmb->coords;

    pmb->par_for_bndry(
        "dirichlet_boundary", vars, domain, CC, coarse,
        KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
            if (right) {
                bound(p, k - ke, j - je, i - ie) = q(p, k, j, i);
            } else {
                bound(p, k, j, i) = q(p, k, j, i);
            }
        }
    );
}
