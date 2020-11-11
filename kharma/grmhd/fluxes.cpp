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

using namespace parthenon;

// Functions for finding LLF fluxes.
// LR to flux is up for deletion now ReconandFlux works reliably
namespace LLF {

TaskStatus LRToFlux(std::shared_ptr<MeshBlockData<Real>>& rc, GridVars pl, GridVars pr, const int dir, GridVars flux)
{
    FLAG("LR to flux");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    // 1-zone halo in nontrivial dimensions. Don't calculate/allow fluxes in trivial dimensions
    ks = (ks == 0) ? 0 : ks - 2; ke = (ke == 0) ? 0 : ke + 2;
    if (ke == 0 && dir == X3DIR) return TaskStatus::complete;
    js = (js == 0) ? 0 : js - 2; je = (je == 0) ? 0 : je + 2;
    if (je == 0 && dir == X2DIR) return TaskStatus::complete;
    is = is - 2; ie = ie + 2;

    auto& G = pmb->coords;
    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");


    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;

    // So far we don't need fluxes that don't match faces
    Loci loc;
    switch (dir) {
    case 1:
        loc = Loci::face1;
        break;
    case 2:
        loc = Loci::face2;
        break;
    case 3:
        loc = Loci::face3;
        break;
    }

    //  LOOP FUSION BABY
    pmb->par_for("uber_flux", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA_3D {
                FourVectors Dtmp;
                Real cmaxL, cmaxR, cminL, cminR;
                Real cmin, cmax, ctop_loc;

                Real fluxL[8], fluxR[8];
                Real Ul[8], Ur[8];

                // All the following calls write to *local* temporaries.
                // That means we don't need an offset-left array to make indices line up
                // We can just *read* from a different spot
                // TODO WARNING TODO Reads metric from the wrong zone, too.  Read with offset & then use local versions!!
                int kl, jl, il;
                switch (dir) {
                case X1DIR:
                    kl = k; jl = j; il = i - 1;
                    break;
                case X2DIR:
                    kl = k; jl = j - 1; il = i;
                    break;
                case X3DIR:
                    kl = k - 1; jl = j; il = i;
                    break;
                }

                // Left
                get_state(G, pl, kl, jl, il, loc, Dtmp);
                prim_to_flux(G, pl, Dtmp, eos, kl, jl, il, loc, 0, Ul); // dir==0 -> U instead of F in direction
                prim_to_flux(G, pl, Dtmp, eos, kl, jl, il, loc, dir, fluxL);
                mhd_vchar(G, pl, Dtmp, eos, kl, jl, il, loc, dir, cmaxL, cminL);

                // Right
                get_state(G, pr, k, j, i, loc, Dtmp);
                // Note: these three can be done simultaneously if we want to get real fancy
                prim_to_flux(G, pr, Dtmp, eos, k, j, i, loc, 0, Ur);
                prim_to_flux(G, pr, Dtmp, eos, k, j, i, loc, dir, fluxR);
                mhd_vchar(G, pr, Dtmp, eos, k, j, i, loc, dir, cmaxR, cminR);

                cmax = fabs(max(max(0.,  cmaxL),  cmaxR));
                cmin = fabs(max(max(0., -cminL), -cminR));
                ctop_loc = max(cmax, cmin);

                ctop(dir, k, j, i) = ctop_loc;
                PLOOP flux(p, k, j, i) = 0.5 * (fluxL[p] + fluxR[p] - ctop_loc * (Ur[p] - Ul[p]));
            }
    );

    FLAG("Uber fluxcalc");
    return TaskStatus::complete;
}

TaskStatus ReconAndFlux(std::shared_ptr<MeshBlockData<Real>>& rc, const int& dir)
{
    FLAG(string_format("Recon and flux X%d", dir));
    auto pmb = rc->GetBlockPointer();
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

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    auto& P = rc->Get("c.c.bulk.prims").data;
    auto& flux = rc->Get("c.c.bulk.cons").flux[dir];

    auto& G = pmb->coords;
    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");
    ReconstructionType recon = pmb->packages["GRMHD"]->Param<ReconstructionType>("recon");

    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;

    // So far we don't need fluxes that don't match faces
    Loci loc;
    switch (dir) {
    case X1DIR:
        loc = Loci::face1;
        break;
    case X2DIR:
        loc = Loci::face2;
        break;
    case X3DIR:
        loc = Loci::face3;
        break;
    }

    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(NPRIM, n1);

    pmb->par_for_outer(string_format("uberkernel_x%d", dir), 3 * scratch_size_in_bytes, scratch_level, ks, ke, js, je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
            ScratchPad2D<Real> ql(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> qr(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), NPRIM, n1);

            // Get reconstructed state on faces
            // TODO switch statements are fast... right? This dispatch table is a pain, but so is another variant
            switch (recon) {
            case ReconstructionType::linear_mc:
                switch (dir) {
                case X1DIR:
                    Reconstruction::PiecewiseLinearX1(member, k, j, is, ie, P, ql, qr);
                    break;
                case X2DIR:
                    Reconstruction::PiecewiseLinearX2(member, k, j, is, ie, P, ql, q_unused);
                    Reconstruction::PiecewiseLinearX2(member, k, j - 1, is, ie, P, q_unused, qr);
                    break;
                case X3DIR:
                    Reconstruction::PiecewiseLinearX3(member, k, j, is, ie, P, ql, q_unused);
                    Reconstruction::PiecewiseLinearX3(member, k - 1, j, is, ie, P, q_unused, qr);
                    break;
                }
                break;
            case ReconstructionType::weno5:
                switch (dir) {
                case X1DIR:
                    Reconstruction::WENO5X1(member, k, j, is, ie, P, ql, qr);
                    break;
                case X2DIR:
                    Reconstruction::WENO5X2l(member, k, j, is, ie, P, ql);
                    Reconstruction::WENO5X2r(member, k, j - 1, is, ie, P, qr);
                    break;
                case X3DIR:
                    Reconstruction::WENO5X3l(member, k, j, is, ie, P, ql);
                    Reconstruction::WENO5X3r(member, k - 1, j, is, ie, P, qr);
                    break;
                }
                break;
            }

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            parthenon::par_for_inner(member, is, ie, [&](const int i) {
                // Reverse the fluxes so that "left" and "right" are w.r.t. the *faces*
                Real pl[NPRIM], pr[NPRIM];
                if (dir == X1DIR) {
                    PLOOP {
                        pr[p] = ql(p, i);
                        pl[p] = qr(p, i - 1);
                    }
                } else {
                    PLOOP {
                        pr[p] = ql(p, i);
                        pl[p] = qr(p, i);
                    }
                }

                // LR -> flux
                FourVectors Dtmp;
                Real cmaxL, cmaxR, cminL, cminR;
                Real cmin, cmax, ctop_loc;

                Real fluxL[8], fluxR[8];
                Real Ul[8], Ur[8];

                // Left
                get_state(G, pl, k, j, i, loc, Dtmp);
                prim_to_flux(G, pl, Dtmp, eos, k, j, i, loc, 0, Ul); // dir==0 -> U instead of F in direction
                prim_to_flux(G, pl, Dtmp, eos, k, j, i, loc, dir, fluxL);
                mhd_vchar(G, pl, Dtmp, eos, k, j, i, loc, dir, cmaxL, cminL);

                // Right
                get_state(G, pr, k, j, i, loc, Dtmp);
                // Note: these three can be done simultaneously if we want to get real fancy
                prim_to_flux(G, pr, Dtmp, eos, k, j, i, loc, 0, Ur);
                prim_to_flux(G, pr, Dtmp, eos, k, j, i, loc, dir, fluxR);
                mhd_vchar(G, pr, Dtmp, eos, k, j, i, loc, dir, cmaxR, cminR);

                cmax = fabs(max(max(0.,  cmaxL),  cmaxR));
                cmin = fabs(max(max(0., -cminL), -cminR));
                ctop_loc = max(cmax, cmin);

                ctop(dir, k, j, i) = ctop_loc;
                PLOOP flux(p, k, j, i) = 0.5 * (fluxL[p] + fluxR[p] - ctop_loc * (Ur[p] - Ul[p]));
            });
        }
    );

    FLAG(string_format("Finished recon and flux X%d", dir));
    return TaskStatus::complete;
}

} // namespace LLF