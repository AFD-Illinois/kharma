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

void KBoundaries::DirichletImpl(MeshBlockData<Real> *rc, BoundaryFace bface, bool coarse, bool set)
{
    // Get all cell-centered ghosts, minus anything just used at startup
    using FC = Metadata::FlagCollection;
    FC ghost_vars = FC({Metadata::FillGhost, Metadata::Conserved})
                  + FC({Metadata::FillGhost, Metadata::GetUserFlag("Primitive")})
                  - FC({Metadata::GetUserFlag("StartupOnly")});
    auto q = rc->PackVariables(ghost_vars, coarse);
    DirichletSetFromField(rc, q, "Boundaries.", bface, coarse, set, false);

    FC ghost_vars_f = FC({Metadata::FillGhost, Metadata::Face})
                  - FC({Metadata::GetUserFlag("StartupOnly")});
    auto q_f = rc->PackVariables(ghost_vars_f, coarse);
    DirichletSetFromField(rc, q_f, "Boundaries.f.", bface, coarse, set, true);
}

void KBoundaries::DirichletSetFromField(MeshBlockData<Real> *rc, VariablePack<Real> q, std::string prefix,
                                        BoundaryFace bface, bool coarse, bool set, bool do_face)
{
    auto pmb = rc->GetBlockPointer();
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // We're sometimes called without any variables to sync (e.g. syncing flags, EMFs), just return
    if (q.GetDim(4) == 0) return;

    auto bound = rc->Get(prefix + BoundaryName(bface)).data;
    if (q.GetDim(4) != bound.GetDim(4)) {
        std::cerr << "Dirichlet boundary mismatch! Boundary cache: " << bound.GetDim(4) << " for pack: " << q.GetDim(4) << std::endl;
    }

    // Indices
    const IndexRange vars = IndexRange{0, q.GetDim(4) - 1};
    const bool right = !BoundaryIsInner(bface);
    const auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const int dir = BoundaryDirection(bface);
    const auto domain = BoundaryDomain(bface);

    if (do_face) {
        for (auto te : {F1, F2, F3}) {
            // Subtract off the starting index if we're on the right
            // Start numbering faces at 0 for both buffers
            const int ie = (dir == X1DIR) ? bounds.ie(IndexDomain::interior) + 1 + (te == F1) : 0;
            const int je = (dir == X2DIR) ? bounds.je(IndexDomain::interior) + 1 + (te == F2) : 0;
            const int ke = (dir == X3DIR) ? bounds.ke(IndexDomain::interior) + 1 + (te == F3) : 0;
            // Set/recall one face right on left side, left on right side.
            // This sets the last faces technically on domain
            const int ioff = (dir == X1DIR && te == F1) ? ((right) ? -1 : 1) : 0;
            const int joff = (dir == X2DIR && te == F2) ? ((right) ? -1 : 1) : 0;
            const int koff = (dir == X3DIR && te == F3) ? ((right) ? -1 : 1) : 0;
            pmb->par_for_bndry(
                "dirichlet_boundary_face", vars, domain, te, coarse,
                KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
                    if (set) {
                        if (right) {
                            bound(p, k - ke, j - je, i - ie) = q(p, k + koff, j + joff, i + ioff);
                        } else {
                            bound(p, k, j, i) = q(p, k + koff, j + joff, i + ioff);
                        }
                    } else {
                        if (right) {
                            q(p, k + koff, j + joff, i + ioff) = bound(p, k - ke, j - je, i - ie);
                        } else {
                            q(p, k + koff, j + joff, i + ioff) = bound(p, k, j, i);
                        }
                    }
                }
            );
        }
    } else {
        // Subtract off the starting index if we're on the right
        const int ie = (dir == X1DIR) ? bounds.ie(IndexDomain::interior) + 1 : 0;
        const int je = (dir == X2DIR) ? bounds.je(IndexDomain::interior) + 1 : 0;
        const int ke = (dir == X3DIR) ? bounds.ke(IndexDomain::interior) + 1 : 0;
        pmb->par_for_bndry(
            "dirichlet_boundary", vars, domain, CC, coarse,
            KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
                if (set) {
                    if (right) {
                        bound(p, k - ke, j - je, i - ie) = q(p, k, j, i);
                    } else {
                        bound(p, k, j, i) = q(p, k, j, i);
                    }
                } else {
                    if (right) {
                        q(p, k, j, i) = bound(p, k - ke, j - je, i - ie);
                    } else {
                        q(p, k, j, i) = bound(p, k, j, i);
                    }
                }
            }
        );
    }
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
                auto pmb = rc->GetBlockPointer();
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
