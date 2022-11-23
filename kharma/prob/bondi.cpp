/* 
 *  File: bondi.cpp
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

#include "bondi.hpp"

/**
 * Initialization of a Bondi problem with specified sonic point, BH mdot, and horizon radius
 * TODO mdot and rs are redundant and should be merged into one parameter. Uh, no.
 */
TaskStatus InitializeBondi(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag(rc, "Initializing Bondi problem");
    auto pmb = rc->GetBlockPointer();

    const Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
    const Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);

    // Add these to package properties, since they continue to be needed on boundaries
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("mdot")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("mdot", mdot);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rs")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("rs", rs);

    // Set the whole domain to the analytic solution to begin
    SetBondi(rc);

    Flag(rc, "Initialized");
    return TaskStatus::complete;
}

TaskStatus SetBondi(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "Setting Bondi zones");
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc, prims_map);
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const Real mdot = pmb->packages.Get("GRMHD")->Param<Real>("mdot");
    const Real rs = pmb->packages.Get("GRMHD")->Param<Real>("rs");
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // Just the X1 right boundary
    GRCoordinates G = pmb->coords;
    SphKSCoords ks = mpark::get<SphKSCoords>(G.coords.base);
    SphBLCoords bl = SphBLCoords(ks.a);
    CoordinateEmbedding cs = G.coords;

    // This function currently only handles "outer X1" and "entire" grid domains,
    // but is the special-casing here necessary?
    // Can we define outer_x1 w/priority more flexibly?
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int ibs, ibe;
    if (domain == IndexDomain::outer_x1) {
        ibs = bounds.GetBoundsI(IndexDomain::interior).e+1;
        ibe = bounds.GetBoundsI(IndexDomain::entire).e;
    } else {
        ibs = bounds.GetBoundsI(domain).s;
        ibe = bounds.GetBoundsI(domain).e;
    }
    IndexRange jb_e = bounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb_e = bounds.GetBoundsK(IndexDomain::entire);
    pmb->par_for("bondi_boundary", kb_e.s, kb_e.e, jb_e.s, jb_e.e, ibs, ibe,
        KOKKOS_LAMBDA_3D {
            get_prim_bondi(G, cs, P, m_p, gam, bl, ks, mdot, rs, k, j, i);
            // TODO all flux
            GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
        }
    );

    Flag(rc, "Set");
    return TaskStatus::complete;
}
