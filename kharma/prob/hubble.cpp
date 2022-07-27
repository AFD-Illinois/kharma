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

    // Original problem definition:
    // max(v0*x) = 1e-3 (on domain 0->1)
    // max(rho*v0*x/ug) = 1
    // gam = 4/3, game = 5/3
    // TODO adapt these to fit other domain size as 
        // v0*x should stay below or equal to 1
    Real v0 = pin->GetOrAddReal("hubble", "v0", 1.e-3);
    Real ug0 = pin->GetOrAddReal("hubble", "ug0", 1.e-3);
    Real rho0 = pin->GetOrAddReal("hubble", "rho0", 1.0);
    Real fcool = pin->GetOrAddReal("hubble", "fcool", 1.0);
    Real dyntimes = pin->GetOrAddReal("hubble", "dyntimes", 1.0);
    bool helecs = pin->GetOrAddBoolean("hubble", "electrons", true);
    // Whether to stop after 1 dynamical time L/max(v0*x)
    bool set_tlim = pin->GetOrAddBoolean("hubble", "set_tlim", false);

    // Add everything to package parameters, since they continue to be needed on boundaries
    Params& g_params = pmb->packages.Get("GRMHD")->AllParams();
    int counter = -5.0;
    if(!g_params.hasKey("counter")) g_params.Add("counter", counter, true);
    if(!g_params.hasKey("rho0")) g_params.Add("rho0", rho0);
    if(!g_params.hasKey("v0"))  g_params.Add("v0", v0);
    if(!g_params.hasKey("ug0")) g_params.Add("ug0", ug0);
    if(!g_params.hasKey("fcool")) g_params.Add("fcool", fcool);
    if(!g_params.hasKey("helecs")) g_params.Add("helecs", helecs);

    // This is how we will initialize kel values later
    if (helecs) {
        const Real fel0 = pmb->packages.Get("Electrons")->Param<Real>("fel_0");
        if(!g_params.hasKey("ue0")) g_params.Add("ue0", fel0 * ug0);
    }

    // Override end time to be 1 dynamical time L/max(v@t=0)
    if (set_tlim) {
        pin->SetReal("parthenon/time", "tlim", dyntimes / v0);
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

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real rho0 = pmb->packages.Get("GRMHD")->Param<Real>("rho0");
    const Real v0 = pmb->packages.Get("GRMHD")->Param<Real>("v0");
    const Real ug0 = pmb->packages.Get("GRMHD")->Param<Real>("ug0");
    const Real fcool = pmb->packages.Get("GRMHD")->Param<Real>("fcool");
    int counter = pmb->packages.Get("GRMHD")->Param<int>("counter");
    const Real tt = pmb->packages.Get("Globals")->Param<Real>("time");
    const Real dt = pmb->packages.Get("Globals")->Param<Real>("dt_last");
    const bool helecs = pmb->packages.Get("GRMHD")->Param<bool>("helecs");

    Real t = tt + 0.5*dt;
    if ((counter%4) > 1)   t = tt + dt;

    const auto& G = pmb->coords;

    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    
    // Setting as in equation 37
    Real toberho = rho0 / (1. + v0*t);
    Real tobeu  = fcool * ug0 / pow(1 + v0*t, 2);
    // Not cooling and not interested in trivial solution
    if (fcool == 0) {
        tobeu = ug0 / pow(1 + v0*t, gam);
    } else if (fcool == -1) {
        tobeu = ug0;
        toberho = rho0;
    }
    pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord_embed(k, j, i, Loci::center, X);

            rho(k, j, i) = toberho;
            u(k, j, i) = tobeu;
            if (fcool != -1)
                uvec(0, k, j, i) = v0 * X[1] / (1 + v0*t);
            else 
                uvec(0, k, j, i) = v0;
            uvec(1, k, j, i) = 0.0;
            uvec(2, k, j, i) = 0.0;
        }
    );

    if (helecs) {
        GridScalar ktot = rc->Get("prims.Ktot").data;
        GridScalar kel_const = rc->Get("prims.Kel_Constant").data;
        const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
        const Real ue0 = pmb->packages.Get("GRMHD")->Param<Real>("ue0");
        Real tobeke = (gam - 2) * (game - 1)/(game - 2) * ue0/pow(rho0, game) * pow(1 + v0*t, game-2);
        pmb->par_for("hubble_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA_3D {
                ktot(k, j, i) = tobeke;
                kel_const(k, j, i) = tobeke; //Since we are using fel = 1
            }
        );
    }
    pmb->packages.Get("GRMHD")->UpdateParam<int>("counter", ++counter);
    Flag("Set");
    return TaskStatus::complete;
}
