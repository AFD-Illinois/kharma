/* 
 *  File: fixup.cpp
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

#include "fixup.hpp"

void ClearCorners(std::shared_ptr<MeshBlock> pmb, GridInt pflag) {
    // TODO add back all but the physical corners

    FLAG("Clearing corners");
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // Is there a better way to check if I'm in a physical corner?
    bool inner_x1 = pmb->boundary_flag[BoundaryFace::inner_x1] != BoundaryFlag::block;
    bool inner_x2 = pmb->boundary_flag[BoundaryFace::inner_x2] != BoundaryFlag::block;
    bool inner_x3 = pmb->boundary_flag[BoundaryFace::inner_x3] != BoundaryFlag::block;
    bool outer_x1 = pmb->boundary_flag[BoundaryFace::outer_x1] != BoundaryFlag::block;
    bool outer_x2 = pmb->boundary_flag[BoundaryFace::outer_x2] != BoundaryFlag::block;
    bool outer_x3 = pmb->boundary_flag[BoundaryFace::outer_x3] != BoundaryFlag::block;
    
    pmb->par_for("clear_corners", 0, max(ks-1, 0), 0, max(js-1, 0), 0, is-1,
        KOKKOS_LAMBDA_3D {
            if (inner_x3 && inner_x2 && inner_x1) pflag(k, j, i) = -1;
            if (inner_x3 && inner_x2 && outer_x1) pflag(k, j, ie+i) = -1;
            if (inner_x3 && outer_x2 && inner_x1) pflag(k, je+j, i) = -1;
            if (outer_x3 && inner_x2 && inner_x1) pflag(ke+k, j, i) = -1;
            if (inner_x3 && outer_x2 && outer_x1) pflag(k, je+j, ie+i) = -1;
            if (outer_x3 && inner_x2 && outer_x1) pflag(ke+k, j, ie+i) = -1;
            if (outer_x3 && outer_x2 && inner_x1) pflag(ke+k, je+j, i) = -1;
            if (outer_x3 && outer_x2 && outer_x1) pflag(ke+k, je+j, ie+i) = -1;
        }
    );

    FLAG("Cleared corner flags");
}

void FixUtoP(std::shared_ptr<Container<Real>>& rc, GridInt pflag, GridInt fflag)
{
    // We expect primitives all the way out to 3 ghost zones on all sides.  But we can only fix primitives with their neighbors.
    // This may actually mean we require the 4 ghost zones Parthenon "wants" us to have, if we need to use only fixed zones.
    // TODO or do a bounds check in fix_U_to_P and average the available zones
    FLAG("Fixing U to P inversions");
    auto pmb = rc->GetBlockPointer();
    GRCoordinates G = pmb->coords;

    ClearCorners(pmb, pflag); // Don't use zones in physical corners. TODO persistent pflag would be faster...

    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // TODO This is a lot of if statements. Slow?
    pmb->par_for("fix_U_to_P", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Negative flags mark physical corners, which shouldn't be fixed
            if (pflag(k, j, i) > InversionStatus::success) {
                double wsum = 0.;
                double sum[NFLUID] = {0};
                // For all neighboring cells...
                for (int n = -1; n <= 1; n++) {
                    for (int m = -1; m <= 1; m++) {
                        for (int l = -1; l <= 1; l++) {
                            int ii = i + l, jj = j + m, kk = k + n;
                            // If in bounds and this cell is not flagged...
                            if (ii >= is && ii <= ie && jj >= js && jj <= je && kk >= ks && kk <= ke) {
                                if (pflag(kk, jj, ii) == InversionStatus::success) {
                                    // Weight by distance.  Note interpolated "fixed" cells stay flagged
                                    double w = 1./(abs(l) + abs(m) + abs(n) + 1);
                                    wsum += w;
                                    FLOOP sum[p] += w * P(p, kk, jj, ii);
                                }
                            }
                        }
                    }
                }

                if(wsum < 1.e-10) {
                    // TODO set a flag and handle it outside
#if 0
                    printf("No neighbors were available!\n");
#endif
                } else {
                    FLOOP P(p, k, j, i) = sum[p]/wsum;

                    // Make sure fixed values still abide by floors
                    fflag(k, j, i) |= gamma_ceiling(G, P, U, eos, k, j, i);
                    fflag(k, j, i) |= fixup_floor(G, P, U, eos, k, j, i);
                    // Make sure the original conserved variables match
                    FourVectors Dtmp;
                    get_state(G, P, k, j, i, Loci::center, Dtmp);
                    prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
                }
            }
        }
    );

    DelEOS(eos);

    FLAG("Fixed U to P inversions");
}