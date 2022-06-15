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
    // TODO can I just build/add/use a Prescription here, rather than building one
    // before each call?
    auto pkg = std::make_shared<StateDescriptor>("Floors");
    Params &params = pkg->AllParams();

    // Floor parameters
    double rho_min_geom, u_min_geom;
    if (pin->GetBoolean("coordinates", "spherical")) {
        // In spherical systems, floors drop as r^2, so set them higher by default
        rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-6);
        u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1.e-8);
    } else {
        rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-8);
        u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1.e-10);
    }
    params.Add("rho_min_geom", rho_min_geom);
    params.Add("u_min_geom", u_min_geom);

    // In iharm3d, overdensities would run away; one proposed solution was
    // to decrease the density floor more with radius.  However, in practice
    // 1. This proved to be a result of the floor vs bsq, not the geometric one
    // 2. interior density floors are dominated by the floor vs bsq
    // Also, this changes the internal energy floor pretty drastically --
    // newly interesting in light of increases to the UU floors
    bool use_r_char = pin->GetOrAddBoolean("floors", "use_r_char", false);
    params.Add("use_r_char", use_r_char);
    double r_char = pin->GetOrAddReal("floors", "r_char", 10);
    params.Add("r_char", r_char);

    // Floors vs magnetic field.  Most commonly hit & most temperamental
    double bsq_over_rho_max = pin->GetOrAddReal("floors", "bsq_over_rho_max", 1e20);
    params.Add("bsq_over_rho_max", bsq_over_rho_max);
    double bsq_over_u_max = pin->GetOrAddReal("floors", "bsq_over_u_max", 1e20);
    params.Add("bsq_over_u_max", bsq_over_u_max);

    // Limit temperature or entropy, optionally by siphoning off extra rather
    // than by adding material.
    double u_over_rho_max = pin->GetOrAddReal("floors", "u_over_rho_max", 1e20);
    params.Add("u_over_rho_max", u_over_rho_max);
    double ktot_max = pin->GetOrAddReal("floors", "ktot_max", 1e20);
    params.Add("ktot_max", ktot_max);
    bool temp_adjust_u = pin->GetOrAddBoolean("floors", "temp_adjust_u", false);
    params.Add("temp_adjust_u", temp_adjust_u);
    // Adjust electron entropy values when applying density floors to conserve
    // internal energy, as in Ressler+ but not more recent implementations
    bool adjust_k = pin->GetOrAddBoolean("floors", "adjust_k", true);
    params.Add("adjust_k", adjust_k);

    // Limit 
    double gamma_max = pin->GetOrAddReal("floors", "gamma_max", 50.);
    params.Add("gamma_max", gamma_max);

    // Frame to apply floors: usually we use normal observer frame, but
    // the option exists to use the fluid frame exclusively or outside a
    // certain radius.  This option adds fluid at speed, making results
    // less reliable but velocity reconstructions potentially more robust
    std::string frame = pin->GetOrAddString("floors", "frame", "normal");
    params.Add("frame", frame);
    if (frame == "normal" || frame == "nof") {
        params.Add("fluid_frame", false);
        params.Add("mixed_frame", false);
    } else if (frame == "fluid" || frame == "ff") {
        params.Add("fluid_frame", true);
        params.Add("mixed_frame", false);
    } else if (frame == "mixed") {
        params.Add("fluid_frame", false);
        params.Add("mixed_frame", true);
    } else {
        throw std::invalid_argument("Floor frame "+frame+" not supported");
    }
    // We initialize this even if not using mixed frame, for constructing Prescription objs
    Real frame_switch = pin->GetOrAddReal("floors", "frame_switch", 50.);
    params.Add("frame_switch", frame_switch);
    

    // Disable all floors.  It is obviously tremendously inadvisable to
    // set this option to true
    bool disable_floors = pin->GetOrAddBoolean("floors", "disable_floors", false);
    params.Add("disable_floors", disable_floors);

    // Temporary fix just for being able to save field values
    // Should switch these to "Integer" fields when Parthenon supports it
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("fflag", m);

    // Floors should be applied to primitive ("Derived") variables just after they are calculated.
    pkg->PostFillDerivedBlock = Floors::PostFillDerivedBlock;
    // Could print floor flags using this package, but they're very similar to pflag
    // so I'm leaving them together
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

TaskStatus ApplyFloors(MeshBlockData<Real> *rc, IndexDomain domain)
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

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Floors::Prescription floors(pmb->packages.Get("Floors")->AllParams());

    // Apply floors over the same zones we just updated with UtoP
    // This selects the entire domain, but we then require pflag >= 0,
    // which keeps us from covering completely uninitialized zones
    // (but still applies to failed UtoP!)
    const IndexRange ib = rc->GetBoundsI(domain);
    const IndexRange jb = rc->GetBoundsJ(domain);
    const IndexRange kb = rc->GetBoundsK(domain);
    pmb->par_for("apply_floors", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            if (((int) pflag(k, j, i)) >= InversionStatus::success) {
                // apply_floors can involve another U_to_P call.  Hide the pflag in bottom 5 bits and retrieve both
                int comboflag = apply_floors(G, P, m_p, gam, k, j, i, floors, U, m_u);
                fflag(k, j, i) = (comboflag / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;

                // Record the pflag as well.  KHARMA did not traditionally do this,
                // because floors were run over uninitialized zones, and thus wrote
                // garbage pflags.  We now prevent this.
                // Note that the pflag is recorded only if inversion failed --
                // floors can paper over zones that really *should* be discarded,
                // even if they now technically correspond to a physical state.
                // Zones next to the sharp edge of the initial torus, for example,
                // can produce negative u when inverted, then magically stay invertible
                // after floors when they should be diffused.
                if (comboflag % HIT_FLOOR_GEOM_RHO) {
                    pflag(k, j, i) = comboflag % HIT_FLOOR_GEOM_RHO;
                }

#if !FUSE_FLOOR_KERNELS
            }
        }
    );
    pmb->par_for("apply_ceilings", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            if (((int) pflag(k, j, i)) >= InversionStatus::success) {
#endif
                // Apply ceilings *after* floors, to make the temperature ceiling better-behaved
                // Ceilings never involve a U_to_P call
                int addflag = fflag(k, j, i);
                addflag |= apply_ceilings(G, P, m_p, gam, k, j, i, floors, U, m_u);
                fflag(k, j, i) = addflag;
            }
        }
    );

    Flag(rc, "Applied");
    return TaskStatus::complete;
}

} // namespace Floors
