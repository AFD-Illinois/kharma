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

#include "debug.hpp"
#include "floors.hpp"

#include <parthenon/parthenon.hpp>
#include "reconstruct/dc_inline.hpp"
#include "reconstruct/plm_inline.hpp"

using namespace parthenon;

TaskStatus HLLE::GetFlux(std::shared_ptr<MeshBlockData<Real>>& rc, const int& dir)
{
    FLAG(string_format("Recon and flux X%d", dir));
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    // 1-zone halo in nontrivial dimensions. Don't calculate/allow fluxes in trivial dimensions
    // Leave is/ie, js/je, ks/ke with their usual definitions for consistency, and define the loop
    // bounds separately to include the appropriate halo
    int halo = 1;
    int ks_l = (ks == 0) ? 0 : ks - halo;
    int ke_l = (ke == 0) ? 0 : ke + halo;
    if (ke == 0 && dir == X3DIR) return TaskStatus::complete;
    int js_l = (js == 0) ? 0 : js - halo;
    int je_l = (je == 0) ? 0 : je + halo;
    if (je == 0 && dir == X2DIR) return TaskStatus::complete;
    int is_l = is - halo;
    int ie_l = ie + halo;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    auto& P = rc->Get("c.c.bulk.prims").data;
    auto& flux = rc->Get("c.c.bulk.cons").flux[dir];

    auto& G = pmb->coords;
    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");
    ReconstructionType recon = pmb->packages["GRMHD"]->Param<ReconstructionType>("recon");

    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;

    // Pull out a struct of just the actual floor values for speed
    FloorPrescription floors = FloorPrescription(pmb->packages["GRMHD"]->AllParams());

    // And cache whether we should reduce reconstruction order on the X2 bound
    bool is_inner_x2 = pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect;
    bool is_outer_x2 = pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect;

    // Calculate fluxes at matching face/direction
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

    pmb->par_for_outer(string_format("flux_x%d", dir), 7 * scratch_size_in_bytes, scratch_level,
        ks_l, ke_l, js_l, je_l,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
            ScratchPad2D<Real> ql(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> qr(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), NPRIM, n1);
            // Extra scratch space for Parthenon's VL limiter stuff
            ScratchPad2D<Real> qc(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> dql(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> dqr(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> dqm(member.team_scratch(scratch_level), NPRIM, n1);

            // Immediately enforce that fluxes through/beyond the pole are 0 (and skip computing them!)
            if (dir == X2DIR && ((is_inner_x2 && j <= js) || (is_outer_x2 && j >= je+1))) {
                parthenon::par_for_inner(member, is, ie,
                    [&](const int& i) {
                        PLOOP flux(p, k, j, i) = 0;
                        // Make ctop small enough not to drive the timestep, but not 0
                        // (since we use ctop <= 0 to detect bad things)
                        ctop(dir, k, j, i) = 1.e-20;
                    }
                );
                return;
            }

            // Get reconstructed state on faces
            // Note the switch statement here is in the outer loop, or it would be a performance concern
            switch (recon) {
            case ReconstructionType::donor_cell:
                switch (dir) {
                case X1DIR:
                    DonorCellX1(member, k, j, is_l, ie_l, P, ql, qr);
                    break;
                case X2DIR:
                    DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
                    DonorCellX2(member, k, j, is_l, ie_l, P, q_unused, qr);
                    break;
                case X3DIR:
                    DonorCellX3(member, k - 1, j, is_l, ie_l, P, ql, q_unused);
                    DonorCellX3(member, k, j, is_l, ie_l, P, q_unused, qr);
                    break;
                }
                break;
            case ReconstructionType::linear_vl:
                switch (dir) {
                case X1DIR:
                    PiecewiseLinearX1(member, k, j, is_l, ie_l, G, P, ql, qr, qc, dql, dqr, dqm);
                    break;
                case X2DIR:
                    PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, G, P, ql, q_unused, qc, dql, dqr, dqm);
                    PiecewiseLinearX2(member, k, j, is_l, ie_l, G, P, q_unused, qr, qc, dql, dqr, dqm);
                    break;
                case X3DIR:
                    PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, G, P, ql, q_unused, qc, dql, dqr, dqm);
                    PiecewiseLinearX3(member, k, j, is_l, ie_l, G, P, q_unused, qr, qc, dql, dqr, dqm);
                    break;
                }
                break;
            case ReconstructionType::linear_mc:
                switch (dir) {
                case X1DIR:
                    KReconstruction::PiecewiseLinearX1(member, k, j, is_l, ie_l, P, ql, qr);
                    break;
                case X2DIR:
                    KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
                    KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_unused, qr);
                    break;
                case X3DIR:
                    KReconstruction::PiecewiseLinearX3(member, k - 1, j, is_l, ie_l, P, ql, q_unused);
                    KReconstruction::PiecewiseLinearX3(member, k, j, is_l, ie_l, P, q_unused, qr);
                    break;
                }
                break;
            case ReconstructionType::weno5:
                switch (dir) {
                case X1DIR:
                    KReconstruction::WENO5X1(member, k, j, is_l, ie_l, P, ql, qr);
                    break;
                case X2DIR:
                    KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
                    KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
                    break;
                case X3DIR:
                    KReconstruction::WENO5X3l(member, k - 1, j, is_l, ie_l, P, ql);
                    KReconstruction::WENO5X3r(member, k, j, is_l, ie_l, P, qr);
                    break;
                }
                break;
            case ReconstructionType::weno5_lower_poles:
                switch (dir) {
                case X1DIR:
                    KReconstruction::WENO5X1(member, k, j, is_l, ie_l, P, ql, qr);
                    break;
                case X2DIR:
                    // This prioritizes calculating fluxes for the same *zone* with the same *algorithm*,
                    // mostly to mirror iharm3d.  One could imagine prioritizing matching faces, instead
                    if (is_inner_x2 && j == js + 1) {
                        DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
                        //DonorCellX2(member, k, j, is_l, ie_l, P, q_unused, qr);
                        KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_unused, qr);
                    } else if (is_inner_x2 && j == js + 2) {
                        KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
                        //KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_unused, qr);
                        KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
                    } else if (is_outer_x2 && j == je - 1) {
                        KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
                        //KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
                        KReconstruction::PiecewiseLinearX2(member, k, j, is_l, ie_l, P, q_unused, qr);
                    } else if (is_outer_x2 && j == je) {
                        KReconstruction::PiecewiseLinearX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
                        //DonorCellX2(member, k, j - 1, is_l, ie_l, P, ql, q_unused);
                        DonorCellX2(member, k, j, is_l, ie_l, P, q_unused, qr);
                    } else {
                        KReconstruction::WENO5X2l(member, k, j - 1, is_l, ie_l, P, ql);
                        KReconstruction::WENO5X2r(member, k, j, is_l, ie_l, P, qr);
                    }
                    break;
                case X3DIR:
                    KReconstruction::WENO5X3l(member, k - 1, j, is_l, ie_l, P, ql);
                    KReconstruction::WENO5X3r(member, k, j, is_l, ie_l, P, qr);
                    break;
                }
                break;
            }

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            parthenon::par_for_inner(member, is_l, ie_l,
                [&](const int& i) {
                    // TODO are immediates slow here?
                    Real pl[NPRIM], pr[NPRIM];
                    PLOOP {
                        pl[p] = ql(p, i);
                        pr[p] = qr(p, i);
                    }
                    // Apply floors to the *reconstructed* primitives, because we have no
                    // guarantee they remotely resemble the *centered* primitives
                    // TODO can we get away with less?  Doesn't seem horrible if we slip the ceilings,
                    // but consistent floors should mean NOF frame at our nominal values...
                    apply_floors(G, pl, eos, k, j, i, floors);
                    //apply_ceilings(G, pl, eos, k, j, i, floors);
                    apply_floors(G, pr, eos, k, j, i, floors);
                    //apply_ceilings(G, pr, eos, k, j, i, floors);

                    // LR -> flux
                    FourVectors Dtmp;
                    Real cmaxL, cmaxR, cminL, cminR;
                    Real cmin, cmax, ctop_loc;

                    Real fluxL[8], fluxR[8];
                    Real Ul[8], Ur[8];

                    // TODO Note that the only dependencies here are that get_state be done first.
                    // Otherwise the 6 prim_to_flux/mhd_vchar calls are all independent
                    // Could also perform these as individual par_for_inner calls over i

                    // Left
                    get_state(G, pl, k, j, i, loc, Dtmp);
                    prim_to_flux(G, pl, Dtmp, eos, k, j, i, loc, 0, Ul); // dir==0 -> U instead of F in direction
                    prim_to_flux(G, pl, Dtmp, eos, k, j, i, loc, dir, fluxL);
                    mhd_vchar(G, pl, Dtmp, eos, k, j, i, loc, dir, cmaxL, cminL);

                    // Right
                    get_state(G, pr, k, j, i, loc, Dtmp);
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
        }
    );

    if (pmb->packages["GRMHD"]->Param<int>("extra_checks") > 0) {
        CheckNaN(rc, dir);
    }

    FLAG(string_format("Finished recon and flux X%d", dir));
    return TaskStatus::complete;
}
