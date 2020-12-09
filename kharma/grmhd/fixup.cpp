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

void FixUtoP(MeshBlockData<Real> *rc, GridInt pflag, GridInt fflag)
{
    // We expect primitives all the way out to 3 ghost zones on all sides.  But we can only fix primitives with their neighbors.
    // This may actually mean we require the 4 ghost zones Parthenon "wants" us to have, if we need to use only fixed zones.
    FLAG("Fixing U to P inversions");
    auto pmb = rc->GetBlockPointer();
    auto& G = pmb->coords;

    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    FloorPrescription floors = FloorPrescription(pmb->packages["GRMHD"]->AllParams());

    int is = is_physical_bound(pmb->boundary_flag[BoundaryFace::inner_x1]) ?
                pmb->cellbounds.is(IndexDomain::interior) : pmb->cellbounds.is(IndexDomain::entire);
    int ie = is_physical_bound(pmb->boundary_flag[BoundaryFace::outer_x1]) ?
                pmb->cellbounds.ie(IndexDomain::interior) : pmb->cellbounds.ie(IndexDomain::entire);
    int js = is_physical_bound(pmb->boundary_flag[BoundaryFace::inner_x2]) ?
                pmb->cellbounds.js(IndexDomain::interior) : pmb->cellbounds.js(IndexDomain::entire);
    int je = is_physical_bound(pmb->boundary_flag[BoundaryFace::outer_x2]) ?
                pmb->cellbounds.je(IndexDomain::interior) : pmb->cellbounds.je(IndexDomain::entire);
    int ks = is_physical_bound(pmb->boundary_flag[BoundaryFace::inner_x3]) ?
                pmb->cellbounds.ks(IndexDomain::interior) : pmb->cellbounds.ks(IndexDomain::entire);
    int ke = is_physical_bound(pmb->boundary_flag[BoundaryFace::outer_x3]) ?
                pmb->cellbounds.ke(IndexDomain::interior) : pmb->cellbounds.ke(IndexDomain::entire);

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
                    //printf("No neighbors were available!\n");
                } else {
                    //printf("Original: %g %g %g %g %g\nReplacement: %g %g %g %g %g\n",
                    //P(0, k, j, i), P(1, k, j, i), P(2, k, j, i), P(3, k, j, i), P(4, k, j, i),
                    //sum[0]/wsum, sum[1]/wsum, sum[2]/wsum, sum[3]/wsum, sum[4]/wsum);
                    FLOOP P(p, k, j, i) = sum[p]/wsum;
                }
                // Make sure to keep lockstep
                p_to_u(G, P, eos, k, j, i, U);

                // Make sure fixed values still abide by floors (floors keep lockstep)
                fflag(k, j, i) |= apply_floors(G, P, U, eos, k, j, i, floors);
                fflag(k, j, i) |= apply_ceilings(G, P, U, eos, k, j, i, floors);
            }
        }
    );

    FLAG("Fixed U to P inversions");
}