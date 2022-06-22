/* 
 *  File: hubble.cpp
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

#include "hubble.hpp"

#include "types.hpp"

TaskStatus InitializeHubble(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag("Initializing Hubble Flow Electron Heating problem");
    auto pmb = rc->GetBlockPointer();
    const Real fel0 = pmb->packages.Get("Electrons")->Param<Real>("fel_0");

    // Original problem definition:
    // max(v0*x) = 1e-3 (on domain 0->1)
    // max(rho*v0*x/ug) = 1
    // gam = 4/3, game = 5/3
    // TODO adapt these to fit other domain size or whatever
    Real v0 = pin->GetOrAddReal("hubble", "v0", 1.e-3);
    Real ug0 = pin->GetOrAddReal("hubble", "ug0", 1.e-3);
    Real rho0 = pin->GetOrAddReal("hubble", "rho0", 1.0);
    // Whether to stop after 1 dynamical time L/max(v0*x)
    bool set_tlim = pin->GetOrAddBoolean("hubble", "set_tlim", false);
    bool q_sign = pin->GetOrAddBoolean("hubble", "q_sign", true);

    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("q_sign")))
        pmb->packages.Get("GRMHD")->AddParam<bool>("q_sign", q_sign);

    // Add everything to package parameters, since they continue to be needed on boundaries
    Params& g_params = pmb->packages.Get("GRMHD")->AllParams();
    if(!g_params.hasKey("rho0")) g_params.Add("rho0", rho0);
    if(!g_params.hasKey("v0"))  g_params.Add("v0", v0);
    if(!g_params.hasKey("ug0")) g_params.Add("ug0", ug0);
    // This is how we will initialize kel values later
    if(!g_params.hasKey("ue0")) g_params.Add("ue0", fel0 * ug0);

    // Override end time to be 1 dynamical time L/max(v@t=0)
    if (set_tlim) {
        pin->SetReal("parthenon/time", "tlim", 1.0 / v0);
    }

    // Then call the general function to fill the grid
    SetHubble(rc);

    Flag("Initialized");
    return TaskStatus::complete;
}

TaskStatus SetHubble(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag("Setting zones to Hubble Flow");
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridScalar ktot = rc->Get("prims.Ktot").data;
    GridScalar kel_const = rc->Get("prims.Kel_Constant").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
    const Real fel0 = pmb->packages.Get("Electrons")->Param<Real>("fel_0");
    const Real fel_const = pmb->packages.Get("Electrons")->Param<Real>("fel_constant");
    const Real rho0 = pmb->packages.Get("GRMHD")->Param<Real>("rho0");
    const Real v0 = pmb->packages.Get("GRMHD")->Param<Real>("v0");
    const Real ug0 = pmb->packages.Get("GRMHD")->Param<Real>("ug0");
    const Real ue0 = pmb->packages.Get("GRMHD")->Param<Real>("ue0");
    const Real t = pmb->packages.Get("Globals")->Param<Real>("time");

    const auto& G = pmb->coords;

    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);

            // Setting as in equation 37
            rho(k, j, i) = rho0 / (1. + v0*t);
            u(k, j, i) = ug0 / pow(1 + v0*t, 2); // pow(_,gam) if not cooling
            uvec(0, k, j, i) = v0 * X[1] / (1 + v0*t);
            uvec(1, k, j, i) = 0.0;
            uvec(2, k, j, i) = 0.0;

            const Real k_e = (gam - 2) * (game - 1)/(game - 2) * ue0/pow(rho0, game) * pow(1 + v0*t, game-2);
            // printf("%.16f | %.16f", ue0, k_e);
            ktot(k, j, i) = k_e;
            kel_const(k, j, i) = k_e; //Since we are using fel = 1
        }
    );

    Flag("Set");
    return TaskStatus::complete;
}
