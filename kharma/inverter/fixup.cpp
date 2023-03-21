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

#include "inverter.hpp"

#include "floors.hpp"
#include "floors_functions.hpp"
#include "flux_functions.hpp"
#include "pack.hpp"

// Version of "PLOOP" guaranteeing specifically the 5 GRMHD fixup-amenable primitive vars
#define NPRIM 5
#define PRIMLOOP for(int p=0; p < NPRIM; ++p)

TaskStatus Inverter::FixUtoP(MeshBlockData<Real> *rc)
{
    // We expect primitives all the way out to 3 ghost zones on all sides.
    // But we can only fix primitives with their neighbors.
    // This may actually mean we require the 4 ghost zones Parthenon "wants" us to have,
    // if we need to use only fixed zones.
    auto pmb = rc->GetBlockPointer();
    // Bail if we're not enabled
    if (!pmb->packages.Get("Inverter")->Param<bool>("fix_average_neighbors")) {
        return TaskStatus::complete;
    }

    Flag(rc, "Fixing U to P inversions");
    // Only fixup the core 5 prims
    auto P = GRMHD::PackHDPrims(rc);

    GridScalar pflag = rc->Get("pflag").data;

    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");
    // Only yell about neighbors on extreme verbosity.
    // 
    const int flag_verbose = pmb->packages.Get("Globals")->Param<int>("flag_verbose");

    // UtoP is applied and fixed over all "Physical" zones -- anything in the domain,
    // OR in an MPI boundary.  This is because it is applied *after* the MPI sync,
    // but before physical boundary zones are computed (which it should never use anyway)

    const IndexRange3 b = GetPhysicalZones(pmb, pmb->cellbounds);

    const auto& G = pmb->coords;

    pmb->par_for("fix_U_to_P", b.kb.s, b.kb.e, b.jb.s, b.jb.e, b.ib.s, b.ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            if (failed(pflag(k, j, i))) {
                // Luckily fixups are rare, so we don't have to worry about optimizing this *too* much
                double wsum = 0., wsum_x = 0.;
                double sum[NPRIM] = {0.}, sum_x[NPRIM] = {0.};
                // For all neighboring cells...
                for (int n = -1; n <= 1; n++) {
                    for (int m = -1; m <= 1; m++) {
                        for (int l = -1; l <= 1; l++) {
                            int ii = i + l, jj = j + m, kk = k + n;
                            // If we haven't overstepped array bounds...
                            if (inside(kk, jj, ii, b.kb, b.jb, b.ib)) {
                                // Weight by distance
                                double w = 1./(m::abs(l) + m::abs(m) + m::abs(n) + 1);

                                // Count only the good cells (not failed AND not corner), if we can
                                if (!failed(pflag(kk, jj, ii))) {
                                    // Weight by distance.  Note interpolated "fixed" cells stay flagged
                                    wsum += w;
                                    PRIMLOOP sum[p] += w * P(p, kk, jj, ii);
                                }
                                // Just in case, keep a sum of even the bad ones
                                wsum_x += w;
                                PRIMLOOP sum_x[p] += w * P(p, kk, jj, ii);
                            }
                        }
                    }
                }

                if(wsum < 1.e-10) {
                    // TODO probably should crash here.
#ifndef KOKKOS_ENABLE_SYCL
                    if (flag_verbose >= 3)
                        printf("No neighbors were available at %d %d %d!\n", i, j, k);
#endif
                    // TODO is there a situation in which this shadow is useful, or do we ditch it?
                    PRIMLOOP P(p, k, j, i) = sum_x[p]/wsum_x;
                } else {
                    PRIMLOOP P(p, k, j, i) = sum[p]/wsum;
                }
            }
        }
    );

    // Re-apply floors to fixed zones
    if (pmb->packages.AllPackages().count("Floors")) {
        // Floor prescription from the package
        const Floors::Prescription floors(pmb->packages.Get("Floors")->AllParams());

        // We need the full packs of prims/cons for p_to_u
        // Pack new variables
        PackIndexMap prims_map, cons_map;
        auto U = GRMHD::PackMHDCons(rc, cons_map);
        P = GRMHD::PackMHDPrims(rc, prims_map);
        const VarMap m_u(cons_map, true), m_p(prims_map, false);
        // Get new sizes
        const int nvar = P.GetDim(4);

        // Get floor flag
        GridScalar fflag = rc->Get("fflag").data;

        pmb->par_for("fix_U_to_P_floors", b.kb.s, b.kb.e, b.jb.s, b.jb.e, b.ib.s, b.ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                if (failed(pflag(k, j, i))) {
                    // Make sure all fixed values still abide by floors (floors keep lockstep)
                    // TODO Full floors instead of just geo?
                    apply_geo_floors(G, P, m_p, gam, k, j, i, floors);

                    // Make sure to keep lockstep
                    // This will only be run for GRMHD, so we can call its p_to_u
                    GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
                }
            }
        );
    }

    Flag(rc, "Fixed U to P inversions");
    return TaskStatus::complete;
}
