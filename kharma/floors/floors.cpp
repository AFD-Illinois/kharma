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
#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"

namespace Floors
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin)
{
    auto pkg = std::make_shared<StateDescriptor>("Floors");
    Params &params = pkg->AllParams();

    // Floor parameters: fill a "Prescription" struct we can pass to the
    // floor/ceiling functions
    Floors::Prescription p;
    p.spherical = pin->GetBoolean("coordinates", "spherical");
    if (p.spherical) {
        // In spherical systems, floors drop as r^2, so set them higher by default
        p.rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-6);
        p.u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1.e-8);
    } else {
        p.rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-8);
        p.u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1.e-10);
    }
    // Record things in the package parameters too
    params.Add("rho_min_geom", p.rho_min_geom);
    params.Add("u_min_geom", p.u_min_geom);

    // In iharm3d, overdensities would run away; one proposed solution was
    // to decrease the density floor more with radius.  However, in practice
    // 1. This proved to be a result of the floor vs bsq, not the geometric one
    // 2. interior density floors are dominated by the floor vs bsq
    // Also, this changes the internal energy floor pretty drastically --
    // newly interesting in light of increases to the UU floors
    p.use_r_char = pin->GetOrAddBoolean("floors", "use_r_char", false);
    params.Add("use_r_char", p.use_r_char);
    p.r_char = pin->GetOrAddReal("floors", "r_char", 10);
    params.Add("r_char", p.r_char);

    // Floors vs magnetic field.  Most commonly hit & most temperamental
    p.bsq_over_rho_max = pin->GetOrAddReal("floors", "bsq_over_rho_max", 1e20);
    params.Add("bsq_over_rho_max", p.bsq_over_rho_max);
    p.bsq_over_u_max = pin->GetOrAddReal("floors", "bsq_over_u_max", 1e20);
    params.Add("bsq_over_u_max", p.bsq_over_u_max);

    // Limit temperature or entropy, optionally by siphoning off extra rather
    // than by adding material.
    p.u_over_rho_max = pin->GetOrAddReal("floors", "u_over_rho_max", 1e20);
    params.Add("u_over_rho_max", p.u_over_rho_max);
    p.ktot_max = pin->GetOrAddReal("floors", "ktot_max", 1e20);
    params.Add("ktot_max", p.ktot_max);
    p.temp_adjust_u = pin->GetOrAddBoolean("floors", "temp_adjust_u", false);
    params.Add("temp_adjust_u", p.temp_adjust_u);
    // Adjust electron entropy values when applying density floors to conserve
    // internal energy, as in Ressler+ but not more recent implementations
    p.adjust_k = pin->GetOrAddBoolean("floors", "adjust_k", true);
    params.Add("adjust_k", p.adjust_k);

    // Limit fluid Lorentz factor gamma
    p.gamma_max = pin->GetOrAddReal("floors", "gamma_max", 50.);
    params.Add("gamma_max", p.gamma_max);

    // Frame to apply floors: usually we use normal observer frame, but
    // the option exists to use the fluid frame exclusively or outside a
    // certain radius.  This option adds fluid at speed, making results
    // less reliable but velocity reconstructions potentially more robust
    std::string frame = pin->GetOrAddString("floors", "frame", "normal");
    params.Add("frame", frame);
    if (frame == "normal" || frame == "nof") {
        p.frame = FloorFrame::normal_observer;
    } else if (frame == "fluid" || frame == "ff") {
        p.frame = FloorFrame::fluid;
    } else if (frame == "mixed") {
        p.frame = FloorFrame::mixed_nof_ff;
    } else {
        throw std::invalid_argument("Floor frame "+frame+" not supported");
    }
    params.Add("fluid_frame", p.frame);
    // We initialize this even if not using mixed frame, for constructing Prescription objs
    p.mixed_frame_switch = pin->GetOrAddReal("floors", "frame_switch", 50.);
    params.Add("frame_switch", p.mixed_frame_switch);

    // Add the whole prescription to the Params struct
    params.Add("prescription", p);
    

    // Option to disable all floors.  It is obviously tremendously inadvisable to
    // enable this option in production simulations.
    // However, it is useful in smaller tests where floors are not expected to be hit
    bool disable_floors = pin->GetOrAddBoolean("floors", "disable_floors", false);
    params.Add("disable_floors", disable_floors);

    // Temporary fix just for being able to save field values
    // Should switch these to "Integer" fields when Parthenon supports it
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("fflag", m);

    // Floors should be applied to primitive ("Derived") variables just after they are calculated.
    pkg->PostFillDerivedBlock = Floors::PostFillDerivedBlock;
    // Could print floor flags using this package, but they're very similar to pflag
    // so I'm leaving them together & printing in debug.cpp
    //pkg->PostStepDiagnosticsMesh = GRMHD::PostStepDiagnostics;

    return pkg;
}

TaskStatus PostFillDerivedBlock(MeshBlockData<Real> *rc)
{
    if (rc->GetBlockPointer()->packages.Get("Floors")->Param<bool>("disable_floors")
        || !rc->GetBlockPointer()->packages.Get("Globals")->Param<bool>("in_loop")) {
        return TaskStatus::complete;
    } else {
        return ApplyFloors(rc);
    }
}

TaskStatus ApplyFloors(MeshBlockData<Real> *rc)
{
    Flag(rc, "Apply floors");
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc, prims_map);
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const auto& G = pmb->coords;

    GridScalar pflag = rc->Get("pflag").data;
    GridScalar fflag = rc->Get("fflag").data;

    const Real& gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Floors::Prescription& floors = pmb->packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    

    // Apply floors over the same zones we just updated with UtoP
    // This selects the entire domain+ghosts, but we then require pflag >= 0,
    // which keeps us from covering completely uninitialized zones,
    // but still applies floors to zones with failed UtoP
    // (which we want in case there are no successfully integrated neighbors)
    const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
    pmb->par_for("apply_floors", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            if (((int) pflag(k, j, i)) >= InversionStatus::success) {
                // apply_floors can involve another U_to_P call.  Hide the pflag in bottom 5 bits and retrieve both
                int comboflag = apply_floors(G, P, m_p, gam, k, j, i, floors, U, m_u);
                int ifflag = (comboflag / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;
                int ipflag = comboflag % HIT_FLOOR_GEOM_RHO;

                // Apply ceilings *after* floors, to make the temperature ceiling better-behaved
                // Ceilings never involve a U_to_P call
                ifflag |= apply_ceilings(G, P, m_p, gam, k, j, i, floors, U, m_u);

                // Write flags to arrays
                fflag(k, j, i) = ifflag;
                if (((int) pflag(k, j, i)) == 0)
                    pflag(k, j, i) = ipflag;

                // Keep conserved variables updated
                if (ifflag != 0 || ipflag != 0) {
                    GRMHD::p_to_u(G, P, m_p, gam, k, j, i, U, m_u);
                }
            }
        }
    );

    Flag(rc, "Applied");
    return TaskStatus::complete;
}

} // namespace Floors
