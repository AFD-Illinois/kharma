/* 
 *  File: boundaries.cpp
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

#include "decs.hpp"

#include "boundaries.hpp"

#include "kharma.hpp"
#include "flux.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "types.hpp"

// Problem-specific boundaries
#include "bondi.hpp"
#include "emhd/conducting_atmosphere.hpp"
#include "emhd/bondi_viscous.hpp"
//#include "hubble.hpp"

// Going to need all modules' headers here
#include "b_flux_ct.hpp"
#include "b_cd.hpp"

#include "basic_types.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

// Single outflow boundary function for inner and outer bounds
// Lots of shared code and only a few indices different
void OutflowX1(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse)
{
    Flag(rc.get(), "Applying KHARMA outflow X1 bound");
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    bool check_inner = pmb->packages.Get("GRMHD")->Param<bool>("check_inflow_inner");
    bool check_outer = pmb->packages.Get("GRMHD")->Param<bool>("check_inflow_outer");
    const bool check_inflow = ((check_inner && domain == IndexDomain::inner_x1)
                            || (check_outer && domain == IndexDomain::outer_x1));

    // q will actually have *both* cons & prims (unless using imex driver)
    // We'll only need cons.B specifically tho
    PackIndexMap prims_map, ghosts_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map, coarse);
    auto q = rc->PackVariables({Metadata::FillGhost}, ghosts_map, coarse);
    const VarMap m_u(ghosts_map, true), m_p(prims_map, false);
    // If we're running imex, q is just the *primitive* variables
    bool prim_ghosts = pmb->packages.Get("GRMHD")->Param<std::string>("driver_type") == "imex";

    // KHARMA is very particular about corner boundaries.
    // In particular, we apply the outflow boundary over ALL X2, X3,
    // Then the polar bound only where outflow is not applied,
    // and periodic bounds only where neither other bound applies.
    // The latter is accomplished regardless of Parthenon's definitions,
    // since these functions are run after Parthenon's MPI boundary syncs &
    // replace whatever they've done.
    IndexDomain ldomain = IndexDomain::interior;
    int is = bounds.is(ldomain), ie = bounds.ie(ldomain);
    int js = bounds.js(ldomain), je = bounds.je(ldomain);
    int ks = bounds.ks(ldomain), ke = bounds.ke(ldomain);
    ldomain = IndexDomain::entire;
    int is_e = bounds.is(ldomain), ie_e = bounds.ie(ldomain);
    int js_e = bounds.js(ldomain), je_e = bounds.je(ldomain);
    int ks_e = bounds.ks(ldomain), ke_e = bounds.ke(ldomain);

    int ref_tmp, ibs, ibe;
    if (domain == IndexDomain::inner_x1) {
        ref_tmp = is;
        ibs = is_e;
        ibe = is - 1;
    } else if (domain == IndexDomain::outer_x1) {
        ref_tmp = ie;
        ibs = ie + 1;
        ibe = ie_e;
    } else {
        throw std::invalid_argument("KHARMA Outflow boundaries only implemented in X1!");
    }
    const int ref = ref_tmp;

    // This first loop copies all variables with the "FillGhost" tag into the outer zones
    // This includes some we may replace below
    pmb->par_for("OutflowX1", 0, q.GetDim(4) - 1, ks_e, ke_e, js_e, je_e, ibs, ibe,
        KOKKOS_LAMBDA_VARS {
            q(p, k, j, i) = q(p, k, j, ref);
        }
    );
    // Inflow check
    if (check_inflow) {
        pmb->par_for("OutflowX1_check", ks_e, ke_e, js_e, je_e, ibs, ibe,
            KOKKOS_LAMBDA_3D {
                KBoundaries::check_inflow(G, P, domain, m_p.U1, k, j, i);
            }
        );
    }
    if (!prim_ghosts) {
        // Normal operation: We copied both both prim & con GRMHD variables, but we want to apply
        // the boundaries based on just the former, so we run P->U
        pmb->par_for("OutflowX1_PtoU", ks_e, ke_e, js_e, je_e, ibs, ibe,
            KOKKOS_LAMBDA_3D {
                // TODO move these steps into FillDerivedDomain, make a GRMHD::PtoU call the last in that series
                // Correct primitive B
                if (m_p.B1 >= 0)
                    VLOOP P(m_p.B1 + v, k, j, i) = q(m_u.B1 + v, k, j, i) / G.gdet(Loci::center, j, i);
                // Recover conserved vars.  Must be only GRMHD.
                GRMHD::p_to_u(G, P, m_p, gam, k, j, i, q, m_u);
            }
        );
    }

    Flag(rc.get(), "Applied");
}

// Single reflecting boundary function for inner and outer bounds
// See above for comments
void ReflectX2(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse) {
    Flag(rc.get(), "Applying KHARMA reflecting X2 bound");
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // q will actually have *both* cons & prims (unless using imex driver)
    // We'll only need cons.B specifically tho
    PackIndexMap prims_map, ghosts_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map, coarse);
    auto q = rc->PackVariables({Metadata::FillGhost}, ghosts_map, coarse);
    const VarMap m_u(ghosts_map, true), m_p(prims_map, false);
    // If we're running imex, q is the *primitive* variables
    bool prim_ghosts = pmb->packages.Get("GRMHD")->Param<std::string>("driver_type") == "imex";

    // KHARMA is very particular about corner boundaries, see above
    IndexDomain ldomain = IndexDomain::interior;
    int is = bounds.is(ldomain), ie = bounds.ie(ldomain);
    int js = bounds.js(ldomain), je = bounds.je(ldomain);
    int ks = bounds.ks(ldomain), ke = bounds.ke(ldomain);
    ldomain = IndexDomain::entire;
    int is_e = bounds.is(ldomain), ie_e = bounds.ie(ldomain);
    int js_e = bounds.js(ldomain), je_e = bounds.je(ldomain);
    int ks_e = bounds.ks(ldomain), ke_e = bounds.ke(ldomain);

    // So. Parthenon wants us to do our thing over is_e to ie_e
    // BUT if we're at the interior bound on X1, that's gonna blow things up
    // (for reasons unknown, inflow bounds must take precedence)
    // so we have to be smart.
    // Side note: this *lags* the X1/X2 corner zones by one step, since X1 is applied first.
    // this is potentially bad
    int ics = (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) ? is : is_e;
    int ice = (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) ? ie : ie_e;
    //int ics = is_e;
    //int ice = ie_e;

    int ref_tmp, add_tmp, jbs, jbe;
    if (domain == IndexDomain::inner_x2) {
        add_tmp = -1;
        ref_tmp = bounds.GetBoundsJ(IndexDomain::interior).s;
        jbs = js_e;
        jbe = js - 1;
    } else if (domain == IndexDomain::outer_x2) {
        add_tmp = 1;
        ref_tmp = bounds.GetBoundsJ(IndexDomain::interior).e;
        jbs = je + 1;
        jbe = je_e;
    } else {
        throw std::invalid_argument("KHARMA Reflecting boundaries only implemented in X2!");
    }
    const int ref = ref_tmp;
    const int add = add_tmp;

    // This first loop copies all variables with the "FillGhost" tag into the outer zones
    // This includes some we may replace below
    pmb->par_for("ReflectX2", 0, q.GetDim(4) - 1, ks_e, ke_e, jbs, jbe, ics, ice,
        KOKKOS_LAMBDA_VARS {
            Real reflect = q.VectorComponent(p) == X2DIR ? -1.0 : 1.0;
            q(p, k, j, i) = reflect * q(p, k, (ref + add) + (ref - j), i);
        }
    );
    if (!prim_ghosts) {
        // Normal operation: see above
        pmb->par_for("ReflectX2_PtoU", ks_e, ke_e, jbs, jbe, ics, ice,
            KOKKOS_LAMBDA_3D {
                if (m_p.B1 >= 0)
                    VLOOP P(m_p.B1 + v, k, j, i) = q(m_u.B1 + v, k, j, i) / G.gdet(Loci::center, j, i);
                GRMHD::p_to_u(G, P, m_p, gam, k, j, i, q, m_u);
            }
        );
    }
}

// Interface calls into the preceding functions
void KBoundaries::InnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse)
{
    // TODO implement as named callback, give combo start/bound problems their own "packages"
    auto pmb = rc->GetBlockPointer();
    std::string prob = pmb->packages.Get("GRMHD")->Param<std::string>("problem");
    if (prob == "hubble") {
       //SetHubble(rc.get(), IndexDomain::inner_x1, coarse);
    } else if (prob == "conducting_atmosphere"){
        dirichlet_bc(rc.get(), IndexDomain::inner_x1, coarse);
    } else {
        OutflowX1(rc, IndexDomain::inner_x1, coarse);
    }
    // If we're in KHARMA/HARM driver, we need primitive versions of all the
    // non-GRMHD vars
    bool prim_ghosts = pmb->packages.Get("GRMHD")->Param<std::string>("driver_type") == "imex";
    if (!prim_ghosts) KHARMA::FillDerivedDomain(rc, IndexDomain::inner_x1, coarse);
}
void KBoundaries::OuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    std::string prob = pmb->packages.Get("GRMHD")->Param<std::string>("problem");
    if (prob == "hubble") {
       //SetHubble(rc.get(), IndexDomain::outer_x1, coarse);
    } else if (prob == "bondi") {
        SetBondi(rc.get(), IndexDomain::outer_x1, coarse);
    } else if (prob == "conducting_atmosphere"){
        dirichlet_bc(rc.get(), IndexDomain::outer_x1, coarse);
    } else if (prob == "bondi_viscous") {
        SetBondiViscous(rc.get(), IndexDomain::outer_x1, coarse);
    } else {
        OutflowX1(rc, IndexDomain::outer_x1, coarse);
    }
    // If we're in KHARMA/HARM driver, we need primitive versions of all the
    // non-GRMHD vars
    bool prim_ghosts = pmb->packages.Get("GRMHD")->Param<std::string>("driver_type") == "imex";
    if (!prim_ghosts) KHARMA::FillDerivedDomain(rc, IndexDomain::outer_x1, coarse);
}
void KBoundaries::InnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    ReflectX2(rc, IndexDomain::inner_x2, coarse);
    bool prim_ghosts = pmb->packages.Get("GRMHD")->Param<std::string>("driver_type") == "imex";
    if (!prim_ghosts) KHARMA::FillDerivedDomain(rc, IndexDomain::inner_x2, coarse);
}
void KBoundaries::OuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    ReflectX2(rc, IndexDomain::outer_x2, coarse);
    bool prim_ghosts = pmb->packages.Get("GRMHD")->Param<std::string>("driver_type") == "imex";
    if (!prim_ghosts) KHARMA::FillDerivedDomain(rc, IndexDomain::outer_x2, coarse);
}

/**
 * Zero flux of mass through inner and outer boundaries, and everything through the pole
 * TODO Both may be unnecessary...
 */
TaskStatus KBoundaries::FixFlux(MeshData<Real> *md)
{
    Flag("Fixing fluxes");
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    bool check_inflow_inner = pmb0->packages.Get("GRMHD")->Param<bool>("check_inflow_inner");
    bool check_inflow_outer = pmb0->packages.Get("GRMHD")->Param<bool>("check_inflow_outer");
    bool fix_flux_pole = pmb0->packages.Get("GRMHD")->Param<bool>("fix_flux_pole");

    IndexDomain domain = IndexDomain::interior;
    const int is = pmb0->cellbounds.is(domain), ie = pmb0->cellbounds.ie(domain);
    const int js = pmb0->cellbounds.js(domain), je = pmb0->cellbounds.je(domain);
    const int ks = pmb0->cellbounds.ks(domain), ke = pmb0->cellbounds.ke(domain);
    const int ndim = pmesh->ndim;

    // Fluxes are defined at faces, so there is one more valid flux than
    // valid cell in the face direction.  That is, e.g. F1 is valid on
    // an (N1+1)xN2xN3 grid, F2 on N1x(N2+1)xN3, etc
    const int ie_l = ie + 1;
    const int je_l = (ndim > 1) ? je + 1 : je;
    //const int ke_l = (ndim > 2) ? ke + 1 : ke;

    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();

        PackIndexMap cons_map;
        auto& F = rc->PackVariablesAndFluxes({Metadata::WithFluxes}, cons_map);
        const int m_rho = cons_map["cons.rho"].first;

        if (check_inflow_inner) {
            if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
                pmb->par_for("fix_flux_in_l", ks, ke, js, je, is, is,
                    KOKKOS_LAMBDA_3D {
                        F.flux(X1DIR, m_rho, k, j, i) = m::min(F.flux(X1DIR, m_rho, k, j, i), 0.);
                    }
                );
            }
        }
        if (check_inflow_outer) {
            if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
                pmb->par_for("fix_flux_in_r", ks, ke, js, je, ie_l, ie_l,
                    KOKKOS_LAMBDA_3D {
                        F.flux(X1DIR, m_rho, k, j, i) = m::max(F.flux(X1DIR, m_rho, k, j, i), 0.);
                    }
                );
            }
        }

        // This is a lot of zero fluxes!
        if (fix_flux_pole) {
            if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
                // This loop covers every flux we need
                pmb->par_for("fix_flux_pole_l", 0, F.GetDim(4) - 1, ks, ke, js, js, is, ie,
                    KOKKOS_LAMBDA_VARS {
                        F.flux(X2DIR, p, k, j, i) = 0.;
                    }
                );
            }

            if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
                pmb->par_for("fix_flux_pole_r", 0, F.GetDim(4) - 1, ks, ke, je_l, je_l, is, ie,
                    KOKKOS_LAMBDA_VARS {
                        F.flux(X2DIR, p, k, j, i) = 0.;
                    }
                );
            }
        }
    }

    Flag("Fixed fluxes");
    return TaskStatus::complete;
}

TaskID KBoundaries::AddBoundarySync(TaskID t_start, TaskList &tl, std::shared_ptr<MeshData<Real>> mc1)
{
    // Readability
    const auto local = parthenon::BoundaryType::local;
    const auto nonlocal = parthenon::BoundaryType::nonlocal;
    // Send all, receive/set local after sending
    auto send =
        tl.AddTask(t_start, parthenon::cell_centered_bvars::SendBoundBufs<nonlocal>, mc1);

    auto t_send_local =
        tl.AddTask(t_start, parthenon::cell_centered_bvars::SendBoundBufs<local>, mc1);
    auto t_recv_local =
        tl.AddTask(t_start, parthenon::cell_centered_bvars::ReceiveBoundBufs<local>, mc1);
    auto t_set_local =
        tl.AddTask(t_recv_local, parthenon::cell_centered_bvars::SetBounds<local>, mc1);

    // Receive/set nonlocal
    auto t_recv = tl.AddTask(
        t_start, parthenon::cell_centered_bvars::ReceiveBoundBufs<nonlocal>, mc1);
    auto t_set = tl.AddTask(t_recv, parthenon::cell_centered_bvars::SetBounds<nonlocal>, mc1);

    // TODO add AMR prolongate/restrict here (and/or maybe option not to?)

    return t_set | t_set_local;
}

void KBoundaries::SyncAllBounds(std::shared_ptr<MeshData<Real>> md, bool apply_domain_bounds)
{
    Flag("Syncing all bounds");
    TaskID t_none(0);

    // If we're using the ImEx driver, where primitives are fundamental, "AddBoundarySync"
    // will only sync those, and we can call PtoU over everything after.
    // If "AddBoundarySync" means syncing conserved variables, we have to call PtoU *before*
    // the MPI sync operation, then recover the primitive vars *again* afterward.
    auto pmesh = md->GetMeshPointer();
    bool sync_prims = pmesh->packages.Get("GRMHD")->Param<std::string>("driver_type") == "imex";

    // TODO un-meshblock the rest of this
    auto &block_list = md.get()->GetMeshPointer()->block_list;

    if (sync_prims) {
        // If we're syncing the primitive vars, we just sync once
        TaskCollection tc;
        auto tr = tc.AddRegion(1);
        AddBoundarySync(t_none, tr[0], md);
        while (!tr.Execute());

        // Then PtoU
        for (auto &pmb : block_list) {
            auto& rc = pmb->meshblock_data.Get();

            Flag("Block fill Conserved");
            Flux::PtoU(rc.get(), IndexDomain::entire);

            if (apply_domain_bounds) {
                Flag("Block physical bounds");
                // Physical boundary conditions
                parthenon::ApplyBoundaryConditions(rc);
            }
        }
    } else {
        // If we're syncing the conserved vars...
        // Honestly, the easiest way through this sync is:
        // 1. PtoU everywhere
        for (auto &pmb : block_list) {
            auto& rc = pmb->meshblock_data.Get();
            Flag("Block fill conserved");
            Flux::PtoU(rc.get(), IndexDomain::entire);
        }

        // 2. Sync MPI bounds like a normal step
        TaskCollection tc;
        auto tr = tc.AddRegion(1);
        AddBoundarySync(t_none, tr[0], md);
        while (!tr.Execute());

        // 3. UtoP everywhere
        for (auto &pmb : block_list) {
            auto& rc = pmb->meshblock_data.Get();

            Flag("Block fill Derived");
            // Fill P again, including ghost zones
            // But, sice we sync'd GRHD primitives already,
            // leave those off by calling *Domain
            // (like we do in a normal boundary sync)
            KHARMA::FillDerivedDomain(rc, IndexDomain::entire, false);

            if (apply_domain_bounds) {
                Flag("Block physical bounds");
                // Physical boundary conditions
                parthenon::ApplyBoundaryConditions(rc);
            }
        }
    }

    Kokkos::fence();
    Flag("Sync'd");
}
