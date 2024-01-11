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

#include "boundaries.hpp"
#include "floors.hpp"
#include "flux_functions.hpp"

/**
 * Initialization of a Bondi problem with specified sonic point & accretion rate
 */
TaskStatus InitializeBondi(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();

    const Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
    const Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);
    const Real ur_frac = pin->GetOrAddReal("bondi", "ur_frac", 1.);
    const Real uphi = pin->GetOrAddReal("bondi", "uphi", 0.); 

    // Set the innermost radius to apply the Bondi problem initialization
    // By default, stay away from the outer BL coordinate singularity
    const Real a = pin->GetReal("coordinates", "a");
    const Real rin_bondi_default = 1 + m::sqrt(1 - a*a) + 0.1;
    // Prefer parameter bondi/r_in_bondi vs bondi/r_shell
    Real rin_bondi_tmp;
    if (pin->DoesParameterExist("bondi", "r_in_bondi")) {
        rin_bondi_tmp = pin->GetReal("bondi", "r_in_bondi");
    } else {
        rin_bondi_tmp = pin->GetOrAddReal("bondi", "r_shell", rin_bondi_default);
    }
    const Real rin_bondi = rin_bondi_tmp;

    const bool fill_interior = pin->GetOrAddBoolean("bondi", "fill_interior", false);
    const bool zero_velocity = pin->GetOrAddBoolean("bondi", "zero_velocity", false);
    const bool diffinit = pin->GetOrAddBoolean("bondi", "diffinit", false); // uses r^-1 density initialization instead

    // Add these to package properties, since they continue to be needed on boundaries
    // TODO Problems NEED params
    if(! pmb->packages.Get("GRMHD")->AllParams().hasKey("mdot"))
        pmb->packages.Get("GRMHD")->AddParam<Real>("mdot", mdot);
    if(! pmb->packages.Get("GRMHD")->AllParams().hasKey("rs"))
        pmb->packages.Get("GRMHD")->AddParam<Real>("rs", rs);
    if(! pmb->packages.Get("GRMHD")->AllParams().hasKey("rin_bondi"))
        pmb->packages.Get("GRMHD")->AddParam<Real>("rin_bondi", rin_bondi);
    if(! pmb->packages.Get("GRMHD")->AllParams().hasKey("fill_interior_bondi"))
        pmb->packages.Get("GRMHD")->AddParam<Real>("fill_interior_bondi", fill_interior);
    if(! pmb->packages.Get("GRMHD")->AllParams().hasKey("zero_velocity_bondi"))
        pmb->packages.Get("GRMHD")->AddParam<Real>("zero_velocity_bondi", zero_velocity);
    if(! pmb->packages.Get("GRMHD")->AllParams().hasKey("diffinit_bondi"))
        pmb->packages.Get("GRMHD")->AddParam<Real>("diffinit_bondi", diffinit);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("ur_frac")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("ur_frac", ur_frac);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("uphi")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("uphi", uphi);

    // Set this problem to control the outer X1 boundary by default
    // remember to disable inflow_check in parameter file!
    // "dirichlet" here means specifically KHARMA's cached boundaries (see boundaries.cpp)
    // The boudaries below are technically Dirichlet boundaries, too, but
    // aren't called that for our purposes
    auto outer_dirichlet = pin->GetString("boundaries", "outer_x1") == "dirichlet";
    auto inner_dirichlet = pin->GetString("boundaries", "inner_x1") == "dirichlet";
    if (outer_dirichlet || inner_dirichlet) {
        SetBondi<IndexDomain::entire>(rc); // TODO iterate & set any bounds specifically?
    } else {
        // Generally, we only set the interior domain, not the ghost zones.
        // This tests that PostInitialize will correctly fill all ghosts
        SetBondi<IndexDomain::interior>(rc);
    }

    // Default Bondi boundary conditions: reset the outer boundary using our set function.
    // Register the callback to replace value from boundaries.cpp, & record the change in pin.
    auto bound_pkg = pmb->packages.Get<KHARMAPackage>("Boundaries");
    if (pin->GetOrAddBoolean("bondi", "set_outer_bound", !outer_dirichlet)) {
        pin->SetString("boundaries", "outer_x1", "bondi");
        bound_pkg->KBoundaries[BoundaryFace::outer_x1] = SetBondi<IndexDomain::outer_x1>;
    }
    // Option to set the inner boundary too.  Ruins convergence
    if (pin->GetOrAddBoolean("bondi", "set_inner_bound", false)) {
        pin->SetString("boundaries", "inner_x1", "bondi");
        bound_pkg->KBoundaries[BoundaryFace::inner_x1] = SetBondi<IndexDomain::inner_x1>;
    }

    // Apply floors to initialize the any part of the domain we didn't
    // Bondi's BL coordinates do not like the EH, so we replace the zeros with something reasonable
    // Note this ignores the "disable_floors" parameter, since it's necessary for initialization
    if (rin_bondi > pin->GetReal("coordinates", "r_in") && !(fill_interior)) {
        Floors::ApplyInitialFloors(pin, rc.get(), IndexDomain::interior);
    }

    return TaskStatus::complete;
}

TaskStatus SetBondiImpl(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    //std::cerr << "Bondi on domain: " << KBoundaries::BoundaryName(KBoundaries::BoundaryFaceOf(domain)) << "coarse: " << coarse << std::endl;

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map);
    auto U = GRMHD::PackMHDCons(rc.get(), cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    // This gets called over various packs -- exit if the pack has no primitive vars
    if (P.GetDim(4) == 0) {
        return TaskStatus::complete;
    }

    const Real mdot = pmb->packages.Get("GRMHD")->Param<Real>("mdot");
    const Real rs = pmb->packages.Get("GRMHD")->Param<Real>("rs");
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real ur_frac = pmb->packages.Get("GRMHD")->Param<Real>("ur_frac");
    const Real uphi = pmb->packages.Get("GRMHD")->Param<Real>("uphi");
    const Real rin_bondi = pmb->packages.Get("GRMHD")->Param<Real>("rin_bondi");
    const bool fill_interior = pmb->packages.Get("GRMHD")->Param<Real>("fill_interior_bondi");
    const bool zero_velocity = pmb->packages.Get("GRMHD")->Param<Real>("zero_velocity_bondi");
    const bool diffinit = pmb->packages.Get("GRMHD")->Param<Real>("diffinit_bondi");

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Just the X1 right boundary
    GRCoordinates G = pmb->coords;

    // Set the Bondi conditions wherever we're asked
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);

    pmb->par_for("bondi_boundary", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {

            Real rho, u;
            Real u_prim[NVEC];
            get_prim_bondi(G, diffinit, rs, mdot, gam, ur_frac, uphi, rin_bondi, fill_interior, rho, u, u_prim, k, j, i);

            // Note that NaN guards, including these, are ignored (!) under -ffast-math flag.
            // Thus we stay away from initializing at EH where this could happen
            if(!isnan(rho)) P(m_p.RHO, k, j, i) = rho;
            if(!isnan(u)) P(m_p.UU, k, j, i) = u;
            if(!isnan(u_prim[0])) P(m_p.U1, k, j, i) = u_prim[0];
            if(!isnan(u_prim[1])) P(m_p.U2, k, j, i) = u_prim[1];
            if(!isnan(u_prim[2])) P(m_p.U3, k, j, i) = u_prim[2];
        }
    );

    // Generally I avoid this, but the viscous Bondi test problem has very unique
    // boundary requirements to converge.  The GRMHD vars must be held constant,
    // but the pressure anisotropy allowed to change as necessary with outflow conditions
    // TODO(BSP) this doesn't properly support face_ct, yell if enabled?
    if (pmb->packages.Get("Globals")->Param<std::string>("problem") == "bondi_viscous") {
        BoundaryFace bface = KBoundaries::BoundaryFaceOf(domain);
        bool inner = KBoundaries::BoundaryIsInner(bface);
        IndexRange ib_i = bounds.GetBoundsI(domain);
        int ref = inner ? ib_i.s : ib_i.e;
        pmb->par_for("bondi_viscous_boundary", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                GReal Xembed[GR_DIM];
                G.coord_embed(k, j, i, Loci::center, Xembed);
                GReal r = Xembed[1];
                // TODO more general?
                if (m_p.B1 >= 0) {
                    P(m_p.B1, k, j, i) = 1/(r*r*r);
                    P(m_p.B2, k, j, i) = 0.;
                    P(m_p.B3, k, j, i) = 0.;
                }
                if (m_p.DP >= 0) {
                    P(m_p.DP, k, j, i) = P(m_p.DP, k, j, ref);
                }
            }
        );
    }

    return TaskStatus::complete;
}
