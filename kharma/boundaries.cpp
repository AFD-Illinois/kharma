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

#include "bondi.hpp"
#include "kharma.hpp"
#include "mhd_functions.hpp"
#include "pack.hpp"

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
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map, coarse);
    auto q = rc->PackVariables({Metadata::FillGhost}, cons_map, coarse);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    // KHARMA is very particular about corner boundaries.
    // In particular, we apply the outflow boundary over ALL X2, X3,
    // Then the polar bound only where outflow is not applied,
    // and periodic bounds only where neither other bound applies.
    // The latter is accomplished regardless of Parthenon's definitions,
    // since these functions are run after Parthenon's MPI boundary
    // syncs
    IndexDomain ldomain = IndexDomain::interior;
    int is = bounds.is(ldomain), ie = bounds.ie(ldomain);
    ldomain = IndexDomain::entire;
    int is_e = bounds.is(ldomain), ie_e = bounds.ie(ldomain);
    int js_e = bounds.js(ldomain), je_e = bounds.je(ldomain);
    int ks_e = bounds.ks(ldomain), ke_e = bounds.ke(ldomain);

    int ref_tmp, dir_tmp, ibs, ibe;
    if (domain == IndexDomain::inner_x1) {
        ref_tmp = bounds.GetBoundsI(IndexDomain::interior).s;
        dir_tmp = 0;
        ibs = is_e;
        ibe = is - 1;
    } else if (domain == IndexDomain::outer_x1) {
        ref_tmp = bounds.GetBoundsI(IndexDomain::interior).e;
        dir_tmp = 1;
        ibs = ie + 1;
        ibe = ie_e;
    } else {
        throw std::invalid_argument("KHARMA Outflow boundaries only implemented in X1!");
    }
    const int ref = ref_tmp;
    const int dir = dir_tmp;

    // This first loop copies all conserved variables into the outer zones
    // This includes some we will replace below, but it would be harder
    // to figure out where they were in the pack than just replace them
    pmb->par_for("OutflowX1", 0, q.GetDim(4) - 1, ks_e, ke_e, js_e, je_e, ibs, ibe,
        KOKKOS_LAMBDA_VARS {
            q(p, k, j, i) = q(p, k, j, ref);
        }
    );
    // Apply KHARMA boundary to the primitive values
    // TODO currently this includes B, which we then replace.
    pmb->par_for("OutflowX1_prims", 0, P.GetDim(4) - 1, ks_e, ke_e, js_e, je_e, ibs, ibe,
        KOKKOS_LAMBDA_VARS {
            P(p, k, j, i) = P(p, k, j, ref);
        }
    );
    // Zone-by-zone recovery of U from P
    pmb->par_for("OutflowX1_PtoU", ks_e, ke_e, js_e, je_e, ibs, ibe,
        KOKKOS_LAMBDA_3D {
            // Inflow check
            KBoundaries::check_inflow(G, P, m_p.U1, k, j, i, dir);
            // TODO move these steps into FillDerivedDomain, make a GRMHD::PrimToFlux call the last in that series
            // Correct primitive B
            VLOOP P(m_p.B1 + v, k, j, i) = q(m_u.B1 + v, k, j, i) / G.gdet(Loci::center, j, i);
            // Recover conserved vars
            GRMHD::p_to_u(G, P, m_p, gam, k, j, i, q, m_u);
        }
    );

    KHARMA::FillDerivedDomain(rc, domain, coarse);
}

// Single reflecting boundary function for inner and outer bounds
// See above for comments
void ReflectX2(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse) {
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    PackIndexMap prims_map, cons_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map, coarse);
    auto q = rc->PackVariables({Metadata::FillGhost}, cons_map, coarse);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    // KHARMA is very particular about corner boundaries, see above
    IndexDomain ldomain = IndexDomain::interior;
    int is = bounds.is(ldomain), ie = bounds.ie(ldomain);
    int js = bounds.js(ldomain), je = bounds.je(ldomain);
    ldomain = IndexDomain::entire;
    int js_e = bounds.js(ldomain), je_e = bounds.je(ldomain);
    int ks_e = bounds.ks(ldomain), ke_e = bounds.ke(ldomain);

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

    pmb->par_for("ReflectX2", 0, q.GetDim(4) - 1, ks_e, ke_e, jbs, jbe, is, ie,
        KOKKOS_LAMBDA_VARS {
            Real reflect = q.VectorComponent(p) == X2DIR ? -1.0 : 1.0;
            q(p, k, j, i) = reflect * q(p, k, (ref + add) + (ref - j), i);
        }
    );
    pmb->par_for("ReflectX2_prims", 0, P.GetDim(4) - 1, ks_e, ke_e, jbs, jbe, is, ie,
        KOKKOS_LAMBDA_VARS {
            Real reflect = P.VectorComponent(p) == X2DIR ? -1.0 : 1.0;
            P(p, k, j, i) = reflect * P(p, k, (ref + add) + (ref - j), i);
        }
    );
    pmb->par_for("ReflectX2_PtoU", ks_e, ke_e, jbs, jbe, is, ie,
        KOKKOS_LAMBDA_3D {
            VLOOP P(m_p.B1 + v, k, j, i) = q(m_u.B1 + v, k, j, i) / G.gdet(Loci::center, j, i);
            GRMHD::p_to_u(G, P, m_p, gam, k, j, i, q, m_u);
        }
    );

    KHARMA::FillDerivedDomain(rc, domain, coarse);
}

// Interface calls into the preceding functions
void KBoundaries::OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse)
{
    OutflowX1(rc, IndexDomain::inner_x1, coarse);
}
void KBoundaries::OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse)
{
    OutflowX1(rc, IndexDomain::outer_x1, coarse);
}
void KBoundaries::ReflectInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse)
{
    ReflectX2(rc, IndexDomain::inner_x2, coarse);
}
void KBoundaries::ReflectOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse)
{
    ReflectX2(rc, IndexDomain::outer_x2, coarse);
}

/**
 * Zero flux of mass through inner and outer boundaries, and everything through the pole
 * TODO Both may be unnecessary...
 */
TaskStatus KBoundaries::FixFlux(MeshBlockData<Real> *rc)
{
    FLAG("Fixing fluxes");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const int ndim = pmb->pmy_mesh->ndim;

    PackIndexMap cons_map;
    const auto& F = rc->PackVariablesAndFluxes({Metadata::WithFluxes}, cons_map);
    const int m_rho = cons_map["cons.rho"].first;


    // For Parthenon's boundary stuff
    const auto nb_0 = IndexRange{0, 0};
    const auto nb_1 = IndexRange{0, F.GetDim(4) - 1};
    const int coarse = 0;

    if (pmb->packages.Get("GRMHD")->Param<bool>("fix_flux_inflow")) {
        if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user)
        {
            pmb->par_for_bndry("fix_flux_in_l", nb_0, IndexDomain::inner_x1, coarse,
                KOKKOS_LAMBDA_VARS {
                    F.flux(X1DIR, m_rho, k, j, i) = min(F.flux(X1DIR, m_rho, k, j, i), 0.);
                }
            );
        }

        if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user &&
            !(pmb->packages.Get("GRMHD")->Param<std::string>("problem") == "bondi"))
        {
            pmb->par_for_bndry("fix_flux_in_r", nb_0, IndexDomain::outer_x1, coarse,
                KOKKOS_LAMBDA_VARS {
                    F.flux(X1DIR, m_rho, k, j, i) = max(F.flux(X1DIR, m_rho, k, j, i), 0.);
                }
            );
        }
    }

    // This is a lot of zero fluxes!
    if (pmb->packages.Get("GRMHD")->Param<bool>("fix_flux_pole")) {
        if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user)
        {
            pmb->par_for_bndry("fix_flux_pole_l", nb_1, IndexDomain::inner_x2, coarse,
                KOKKOS_LAMBDA_VARS {
                    F.flux(X2DIR, p, k, j, i) = 0.;
                }
            );
        }

        if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user)
        {
            pmb->par_for_bndry("fix_flux_pole_r", nb_1, IndexDomain::outer_x2, coarse,
                KOKKOS_LAMBDA_VARS {
                    F.flux(X2DIR, p, k, j, i) = 0.;
                }
            );
        }
    }

    FLAG("Fixed fluxes");
    return TaskStatus::complete;
}
