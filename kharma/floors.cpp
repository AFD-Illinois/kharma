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
#include "phys.hpp"

/**
 * Apply density and internal energy floors and ceilings
 * 
 * Note that apply_ceilings and apply_floors are called from some other places for most steps
 * This applies floors only to the physical zones, and is used in initialization just before the first boundary sync.
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
TaskStatus ApplyFloors(std::shared_ptr<MeshBlockData<Real>>& rc)
{
    FLAG("Apply floors");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    auto& G = pmb->coords;

    GridInt fflag("fflag", n3, n2, n1);

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");
    FloorPrescription floors = FloorPrescription(pmb->packages["GRMHD"]->AllParams());

    // Note floors are applied only to physical zones
    pmb->par_for("apply_floors", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            fflag(k, j, i) = 0;
            fflag(k, j, i) |= (apply_floors(G, P, U, eos, k, j, i, floors) / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;
            fflag(k, j, i) |= apply_ceilings(G, P, U, eos, k, j, i, floors);
        }
    );

#if 0
    // Print some diagnostic info about which floors were hit
    CountFFlags(pmb, fflag.GetHostMirrorAndCopy());
#endif

    FLAG("Applied");
    return TaskStatus::complete;
}