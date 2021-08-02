/* 
 *  File: floors.cpp
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

// Fixups.  Apply limits and fix bad fluid values to maintain integrable state
// ApplyFloors, FixUtoP

#include "floors.hpp"

#include "debug.hpp"
#include "fixup.hpp"
#include "mhd_functions.hpp"

/**
 * Apply density and internal energy floors and ceilings
 * 
 * Note that apply_ceilings and apply_floors are called from some other places for most steps
 * This applies floors only to the physical zones, and is used in initialization just before the first boundary sync.
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
TaskStatus ApplyFloors(MeshBlockData<Real> *rc)
{
    FLAG("Apply floors");
    auto pmb = rc->GetBlockPointer();

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVector B_P = rc->Get("c.c.bulk.B_prim").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVector B_U = rc->Get("c.c.bulk.B_con").data;
    auto& G = pmb->coords;

    GridScalar pflag = rc->Get("c.c.bulk.pflag").data;
    GridScalar fflag = rc->Get("c.c.bulk.fflag").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    FloorPrescription floors = FloorPrescription(pmb->packages.Get("GRMHD")->AllParams());

    // Apply floors over the same zones we just updated with UtoP
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

    pmb->par_for("apply_floors", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            int fflag_local = 0;

            // Fixup_floor involves another U_to_P call.  Hide the pflag in bottom 5 bits and retrieve both
            int comboflag = apply_floors(G, P, B_P, U, B_U, gam, k, j, i, floors);
            fflag_local |= (comboflag / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;
            // The floors as they're written *guarantee* a consistent state in their cells
            // TODO still keep track of whether their inversions failed, without triggering a re-fix
            // if (fflag_local) pflag(k, j, i) = InversionStatus::success; //comboflag % HIT_FLOOR_GEOM_RHO;

            // Apply ceilings *after* floors, to make the temperature ceiling better-behaved
            // Ceilings never involve a U_to_P call
            fflag_local |= apply_ceilings(G, P, B_P, U, gam, k, j, i, floors);

            fflag(k, j, i) = fflag_local;
        }
    );

    FLAG("Applied");
    return TaskStatus::complete;
}
