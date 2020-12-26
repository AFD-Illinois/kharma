/* 
 *  File: reconstruction.cpp
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

#include "reconstruction.hpp"

#include "decs.hpp"

#include "reconstruct/plm_inline.hpp"

// Honestly these are on the chopping block.
// Merged reconstruction works without playing questionable index games

namespace Reconstruction {

TaskStatus ReconstructLR(std::shared_ptr<MeshBlockData<Real>>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr, int dir, ReconstructionType recon)
{
    FLAG(string_format("Reconstuct X%d", dir));
    auto& P = rc->Get("c.c.bulk.prims").data;
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    // TODO take a hard look at this when using small meshes
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    // 1-zone halo in nontrivial dimensions. Don't calculate/allow fluxes in trivial dimensions
    ks = (ks == 0) ? 0 : ks - 1; ke = (ke == 0) ? 0 : ke + 1;
    if (ke == 0 && dir == X3DIR) return TaskStatus::complete;
    js = (js == 0) ? 0 : js - 1; je = (je == 0) ? 0 : je + 1;
    if (je == 0 && dir == X2DIR) return TaskStatus::complete;
    is = is - 1; ie = ie + 1;

    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(NPRIM, n1);

    pmb->par_for_outer(string_format("recon_x%d", dir), 2 * scratch_size_in_bytes, scratch_level, ks, ke, js, je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
            ScratchPad2D<Real> ql(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> qr(member.team_scratch(scratch_level), NPRIM, n1);

            // get reconstructed state on faces
            // TODO switch statements are fast, right? Should we move them outside the kernel?
            switch (recon) {
            case ReconstructionType::linear_mc:
                switch (dir) {
                case X1DIR:
                    PiecewiseLinearX1(member, k, j, is, ie, P, ql, qr);
                    break;
                case X2DIR:
                    PiecewiseLinearX2(member, k, j, is, ie, P, ql, qr);
                    break;
                case X3DIR:
                    PiecewiseLinearX3(member, k, j, is, ie, P, ql, qr);
                    break;
                }
                break;
            case ReconstructionType::weno5:
                switch (dir) {
                case X1DIR:
                    WENO5X1(member, k, j, is, ie, P, ql, qr);
                    break;
                case X2DIR:
                    WENO5X2(member, k, j, is, ie, P, ql, qr);
                    break;
                case X3DIR:
                    WENO5X3(member, k, j, is, ie, P, ql, qr);
                    break;
                }
                break;
            }

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            PLOOP {
                parthenon::par_for_inner(member, is, ie, [&](const int i) {
                    Pr(p, k, j, i) = qr(p, i);
                    Pl(p, k, j, i) = ql(p, i);
                });
            }
        }
    );

    return TaskStatus::complete;
}

TaskStatus ReconstructLRSimple(std::shared_ptr<MeshBlockData<Real>>& rc, ParArrayND<Real> Pl, ParArrayND<Real> Pr, int dir, ReconstructionType recon)
{
    FLAG(string_format("Reconstuct X%d", dir));
    auto& P = rc->Get("c.c.bulk.prims").data;
    auto pmb = rc->GetBlockPointer();
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    // 1-zone halo in nontrivial dimensions. Don't calculate/allow fluxes in trivial dimensions
    ks = (ks == 0) ? 0 : ks - 1; ke = (ke == 0) ? 0 : ke + 1;
    if (ke == 0 && dir == X3DIR) return TaskStatus::complete;
    js = (js == 0) ? 0 : js - 1; je = (je == 0) ? 0 : je + 1;
    if (je == 0 && dir == X2DIR) return TaskStatus::complete;
    is = is - 1; ie = ie + 1;

    pmb->par_for("recon", 0, NPRIM-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_VARS
        {
            switch(dir) {
            case X1DIR:
                weno5(P(p, k, j, i-2), P(p, k, j, i-1), P(p, k, j, i),
                      P(p, k, j, i+1), P(p, k, j, i+2),
                      Pl(p, k, j, i), Pr(p, k, j, i));
                break;
            case X2DIR:
                weno5(P(p, k, j-2, i), P(p, k, j-1, i), P(p, k, j, i),
                      P(p, k, j+1, i), P(p, k, j+2, i),
                      Pl(p, k, j, i),  Pr(p, k, j, i));
                break;
            case X3DIR:
                weno5(P(p, k-2, j, i), P(p, k-1, j, i), P(p, k, j, i),
                      P(p, k+1, j, i), P(p, k+2, j, i),
                      Pl(p, k, j, i),  Pr(p, k, j, i));
                break;
            }
        }
    );

    return TaskStatus::complete;
}

} // namespace Reconstruction
