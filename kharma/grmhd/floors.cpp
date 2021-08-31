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

// Floors.  Apply limits to fluid values to maintain integrable state

#include "floors.hpp"

#include "debug.hpp"
#include "fixup.hpp"
#include "mhd_functions.hpp"
#include "pack.hpp"

/**
 * Apply density and internal energy floors and ceilings
 * 
 * This function is called just after UtoP finishes, and
 * applies to the same subset of zones (anything "on" the grid,
 * i.e. not past a polar or outflow boundary)
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
TaskStatus GRMHD::ApplyFloors(MeshBlockData<Real> *rc)
{
    FLAG("Apply floors");
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc, prims_map);
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const auto& G = pmb->coords;

    //GridScalar pflag = rc->Get("pflag").data;
    GridScalar fflag = rc->Get("fflag").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const FloorPrescription floors = FloorPrescription(pmb->packages.Get("GRMHD")->AllParams());

    // Apply floors over the same zones we just updated with UtoP
    IndexRange ib = GetPhysicalZonesI(pmb->boundary_flag, pmb->cellbounds);
    IndexRange jb = GetPhysicalZonesJ(pmb->boundary_flag, pmb->cellbounds);
    IndexRange kb = GetPhysicalZonesK(pmb->boundary_flag, pmb->cellbounds);

    pmb->par_for("apply_floors", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            // apply_floors can involve another U_to_P call.  Hide the pflag in bottom 5 bits and retrieve both
            int comboflag = apply_floors(G, P, m_p, gam, k, j, i, floors, U, m_u);
            fflag(k, j, i) = (comboflag / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;

            // The floors as they're written guarantee a consistent state in their cells,
            // so we do not flag any additional cells, nor do we remove existing flags
            // (which might have only "needed floors" due to being left untouched by UtoP)
            // TODO record these flags separately, they are likely common depending on floor prescriptions
            // if (fflag_local) pflag(k, j, i) = InversionStatus::success; //comboflag % HIT_FLOOR_GEOM_RHO;

#if !FUSE_FLOOR_KERNELS
        }
    );
    pmb->par_for("apply_ceilings", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
#endif

            // Apply ceilings *after* floors, to make the temperature ceiling better-behaved
            // Ceilings never involve a U_to_P call
            int addflag = fflag(k, j, i);
            addflag |= apply_ceilings(G, P, m_p, gam, k, j, i, floors, U, m_u);
            fflag(k, j, i) = addflag;
        }
    );

    FLAG("Applied");
    return TaskStatus::complete;
}
