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

#include "floors.hpp"
#include "pack.hpp"

// I'm undecided on reintroducing these more widely but they clearly make sense here
#define NPRIM 5
#define PLOOP for(int p=0; p < NPRIM; ++p)

TaskStatus GRMHD::FixUtoP(MeshBlockData<Real> *rc)
{
    // We expect primitives all the way out to 3 ghost zones on all sides.
    // But we can only fix primitives with their neighbors.
    // This may actually mean we require the 4 ghost zones Parthenon "wants" us to have,
    // if we need to use only fixed zones.
    FLAG("Fixing U to P inversions");
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    // TODO what should be averaged on a fixup? Just these core 5 prims?
    // Should there be a flag to do more?
    auto P = GRMHD::PackHDPrims(rc);

    GridScalar pflag = rc->Get("pflag").data;
    GridScalar fflag = rc->Get("fflag").data;

    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");
    const int verbose = pars.Get<int>("verbose");
    const FloorPrescription floors = FloorPrescription(pars);

    const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);

    // TODO attempt to recover from entropy here if it's present

    pmb->par_for("fix_U_to_P", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            // Negative flags mark physical corners, which shouldn't be fixed
            if (((int) pflag(k, j, i)) > InversionStatus::success) {
                // Luckily fixups are rare, so we don't have to worry about optimizing this too much
                double wsum = 0., wsum_x = 0.;
                double sum[NPRIM] = {0.}, sum_x[NPRIM] = {0.};
                // For all neighboring cells...
                for (int n = -1; n <= 1; n++) {
                    for (int m = -1; m <= 1; m++) {
                        for (int l = -1; l <= 1; l++) {
                            int ii = i + l, jj = j + m, kk = k + n;
                            // If in bounds...
                            if (ii >= ib.s && ii <= ib.e && jj >= jb.s && jj <= jb.e && kk >= kb.s && kk <= kb.e) {
                                // Weight by distance
                                double w = 1./(abs(l) + abs(m) + abs(n) + 1);

                                // Count only the good cells, if we can
                                if (((int) pflag(kk, jj, ii)) == InversionStatus::success) {
                                    // Weight by distance.  Note interpolated "fixed" cells stay flagged
                                    wsum += w;
                                    PLOOP sum[p] += w * P(p, kk, jj, ii);
                                }
                                // Just in case, keep a sum of even the bad ones
                                wsum_x += w;
                                PLOOP sum_x[p] += w * P(p, kk, jj, ii);
                            }
                        }
                    }
                }

                if(wsum < 1.e-10) {
                    // TODO probably should crash here.
#ifndef KOKKOS_ENABLE_SYCL
                    if (verbose >= 1) printf("No neighbors were available at %d %d %d!\n", i, j, k);
#endif
                    PLOOP P(p, k, j, i) = sum_x[p]/wsum_x;
                } else {
                    PLOOP P(p, k, j, i) = sum[p]/wsum;
                }
            }
        }
    );

    // We need the full packs of prims/cons for p_to_u
    // Pack new variables
    PackIndexMap prims_map, cons_map;
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    P = GRMHD::PackMHDPrims(rc, prims_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    // Get new sizes
    const int nvar = P.GetDim(4);

    pmb->par_for("fix_U_to_P_floors", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            if (((int) pflag(k, j, i)) > InversionStatus::success) {
                // Make sure to keep lockstep
                GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);

                // And make sure the fixed values still abide by floors (floors keep lockstep)
                int fflag_local = 0;
                fflag_local |= apply_floors(G, P, m_p, gam, k, j, i, floors, U, m_u);
                fflag_local |= apply_ceilings(G, P, m_p, gam, k, j, i, floors, U, m_u);
                fflag(k, j, i) = fflag_local;
            }
        }
    );

    FLAG("Fixed U to P inversions");
    return TaskStatus::complete;
}
