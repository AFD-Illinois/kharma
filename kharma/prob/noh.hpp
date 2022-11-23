/* 
 *  File: noh.hpp
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
#pragma once

#include "decs.hpp"

using namespace parthenon;

/**
 * Noh shock tube test.
 */
TaskStatus InitializeNoh(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag(rc, "Initializing 1D (Noh) Shock test");
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridScalar ktot = rc->Get("prims.Ktot").data;
    GridScalar kel_constant = rc->Get("prims.Kel_Constant").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
    const Real fel0 = pmb->packages.Get("Electrons")->Param<Real>("fel_0");
    const Real fel_constant = pmb->packages.Get("Electrons")->Param<Real>("fel_constant");
    
    const Real mach = pin->GetOrAddReal("noh", "mach", 49);
    const Real rhoL = pin->GetOrAddReal("noh", "rhoL", 1.0);
    const Real rhoR = pin->GetOrAddReal("noh", "rhoR", 1.0);
    const Real PL = pin->GetOrAddReal("noh", "PL", 0.1);
    const Real PR = pin->GetOrAddReal("noh", "PR", 0.1);
    bool set_tlim = pin->GetOrAddBoolean("noh", "set_tlim", false);

    const auto& G = pmb->coords;

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
    const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
    const Real center = (x1min + x1max) / 2.;

    // TODO relativistic sound speed
    Real cs2 = (gam * (gam - 1) * PL) / rhoL;
    Real v1 = mach * m::sqrt(cs2);

    if (set_tlim) {
        pin->SetReal("parthenon/time", "tlim", 0.6*(x1max - x1min)/v1);
    }

    double gamma = 1. / m::sqrt(1. - v1 * v1); // Since we are in flat space


    pmb->par_for("noh_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);

            const bool lhs = X[1] < center;
            rho(k, j, i) = (lhs) ? rhoL : rhoR;
            u(k, j, i) = ((lhs) ? PL : PR)/(gam - 1.);
            uvec(0, k, j, i) = ((lhs) ? v1 : -v1) * gamma;
            uvec(1, k, j, i) = 0.0;
            uvec(2, k, j, i) = 0.0;
        }
    );

    Flag(rc, "Initialized 1D (Noh) Shock test");
    return TaskStatus::complete;
}
