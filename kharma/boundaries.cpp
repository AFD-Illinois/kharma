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
#include "mhd_functions.hpp"

// Going to need all modules' headers here
#include "b_flux_ct.hpp"
#include "b_cd.hpp"

#include "basic_types.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

KOKKOS_INLINE_FUNCTION void check_inflow(const GRCoordinates &G, GridVars P, const int& k, const int& j, const int& i, int type);

void FillDerivedDomain(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, int coarse)
{
    // We need to re-fill the "derived" (primitive) variables from everything
    // except GRMHD
    auto pm = rc->GetParentPointer();
    for (const auto &pkg : pm->packages.AllPackages()) {
        if (pkg.first == "B_FluxCT")
            B_FluxCT::UtoP(rc.get(), domain, coarse);
        if (pkg.first == "B_CD")
            B_CD::UtoP(rc.get(), domain, coarse);
        // TODO ADD ELECTRONS, PASSIVES, WHATEVER
    }
}

void OutflowInnerX1_KHARMA(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    PackIndexMap cons_map;
    auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, cons_map, coarse);
    const int B_start = cons_map["c.c.bulk.B_con"].first;

    int ref = bounds.GetBoundsI(IndexDomain::interior).s;

    // This first loop copies all conserved variables into the outer zones
    // This includes some we will replace below, but it would be harder
    // to figure out where they were in the pack than just replace them
    auto nb = IndexRange{0, q.GetDim(4) - 1};
    pmb->par_for_bndry("OutflowInnerX1", nb, IndexDomain::inner_x1, coarse,
        KOKKOS_LAMBDA_VARS {
            q(p, k, j, i) = q(p, k, j, ref);
        }
    );
    // Parthenon uses the last index, but we're going to be treating the primitives all
    // at once since we'll be calling p_to_u
    nb = IndexRange{0, 0};
    pmb->par_for_bndry("OutflowInnerX1_KHARMA", nb, IndexDomain::inner_x1, coarse,
        KOKKOS_LAMBDA (const int &z, const int &k, const int &j, const int &i) {
            // Apply boundary on primitives
            PLOOP P(p, k, j, i) = P(p, k, j, ref);
            // Inflow check
            check_inflow(G, P, k, j, i, 0);
            // Recover conserved vars
            Real B_P[NVEC];
            VLOOP B_P[v] = q(B_start + v, k, j, i) / G.gdet(Loci::center, j, i);
            GRMHD::p_to_u(G, P, B_P, gam, k, j, i, U);
        }
    );

    FillDerivedDomain(rc, IndexDomain::inner_x1, coarse);
}
void OutflowOuterX1_KHARMA(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    PackIndexMap cons_map;
    auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, cons_map, coarse);
    const int B_start = cons_map["c.c.bulk.B_con"].first;

    int ref = bounds.GetBoundsI(IndexDomain::interior).e;

    auto nb = IndexRange{0, q.GetDim(4) - 1};
    pmb->par_for_bndry("OutflowOuterX1", nb, IndexDomain::outer_x1, coarse,
        KOKKOS_LAMBDA_VARS {
            q(p, k, j, i) = q(p, k, j, ref);
        }
    );
    if (pmb->packages.Get("GRMHD")->Param<std::string>("problem") == "bondi") {
        FLAG("Bondi outer boundary");
        ApplyBondiBoundary(rc.get());
    } else {
        nb = IndexRange{0, 0};
        pmb->par_for_bndry("OutflowOuterX1_KHARMA", nb, IndexDomain::outer_x1, coarse,
            KOKKOS_LAMBDA (const int &z, const int &k, const int &j, const int &i) {
                // Apply boundary on primitives
                PLOOP P(p, k, j, i) = P(p, k, j, ref);
                // Inflow check
                check_inflow(G, P, k, j, i, 1);
                // Recover conserved vars
                Real B_P[NVEC];
                VLOOP B_P[v] = q(B_start + v, k, j, i) / G.gdet(Loci::center, j, i);
                GRMHD::p_to_u(G, P, B_P, gam, k, j, i, U);
            }
        );
    }

    FillDerivedDomain(rc, IndexDomain::outer_x1, coarse);
}
void ReflectInnerX2_KHARMA(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    PackIndexMap cons_map;
    auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, cons_map, coarse);
    const int B_start = cons_map["c.c.bulk.B_con"].first;

    int ref = bounds.GetBoundsJ(IndexDomain::interior).s;

    auto nb = IndexRange{0, q.GetDim(4) - 1};
    pmb->par_for_bndry("ReflectInnerX2", nb, IndexDomain::inner_x2, coarse,
        KOKKOS_LAMBDA_VARS {
            Real reflect = q.VectorComponent(p) == X2DIR ? -1.0 : 1.0;
            q(p, k, j, i) = reflect * q(p, k, (ref - 1) + (ref - j), i);
        }
    );
    nb = IndexRange{0, 0};
    pmb->par_for_bndry("ReflectInnerX2_KHARMA", nb, IndexDomain::inner_x2, coarse,
        KOKKOS_LAMBDA (const int &z, const int &k, const int &j, const int &i) {
            PLOOP {
                Real reflect = ((p == prims::u2) ? -1.0 : 1.0);
                P(p, k, j, i) = reflect * P(p, k, (ref - 1) + (ref - j), i);
            }
            // Recover conserved vars
            Real B_P[NVEC];
            VLOOP B_P[v] = q(B_start + v, k, j, i) / G.gdet(Loci::center, j, i);
            GRMHD::p_to_u(G, P, B_P, gam, k, j, i, U);
        }
    );

    FillDerivedDomain(rc, IndexDomain::inner_x2, coarse);
}
void ReflectOuterX2_KHARMA(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    PackIndexMap cons_map;
    auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, cons_map, coarse);
    const int B_start = cons_map["c.c.bulk.B_con"].first;

    int ref = bounds.GetBoundsJ(IndexDomain::interior).e;

    auto nb = IndexRange{0, q.GetDim(4) - 1};
    pmb->par_for_bndry("ReflectOuterX2", nb, IndexDomain::outer_x2, coarse,
        KOKKOS_LAMBDA_VARS {
            Real reflect = q.VectorComponent(p) == X2DIR ? -1.0 : 1.0;
            q(p, k, j, i) = reflect * q(p, k, (ref + 1) + (ref - j), i);
        }
    );
    nb = IndexRange{0, 0};
    pmb->par_for_bndry("ReflectOuterX2_KHARMA", nb, IndexDomain::outer_x2, coarse,
        KOKKOS_LAMBDA (const int &z, const int &k, const int &j, const int &i) {

            PLOOP {
                Real reflect = ((p == prims::u2) ? -1.0 : 1.0);
                P(p, k, j, i) = reflect * P(p, k, (ref + 1) + (ref - j), i);
            }
            // Recover conserved vars
            Real B_P[NVEC];
            VLOOP B_P[v] = q(B_start + v, k, j, i) / G.gdet(Loci::center, j, i);
            GRMHD::p_to_u(G, P, B_P, gam, k, j, i, U);
        }
    );

    FillDerivedDomain(rc, IndexDomain::outer_x2, coarse);
}

/**
 * Check for flow into simulation and reset velocity to eliminate it
 * TODO does Parthenon do something like this for outflow bounds already?
 *
 * @param type: 0 to check outflow from EH, 1 to check inflow from outer edge
 */
KOKKOS_INLINE_FUNCTION void check_inflow(const GRCoordinates &G, GridVars P, const int& k, const int& j, const int& i, int type)
{
    Real ucon[GR_DIM];
    GRMHD::calc_ucon(G, P, k, j, i, Loci::center, ucon);

    if (((ucon[1] > 0.) && (type == 0)) ||
        ((ucon[1] < 0.) && (type == 1)))
    {
        // Find gamma and remove it from primitive velocity
        // TODO check failures?
        double gamma = lorentz_calc(G, P, k, j, i, Loci::center);
        P(prims::u1, k, j, i) /= gamma;
        P(prims::u2, k, j, i) /= gamma;
        P(prims::u3, k, j, i) /= gamma;

        // Reset radial velocity so radial 4-velocity is zero
        Real alpha = 1. / sqrt(-G.gcon(Loci::center, j, i, 0, 0));
        Real beta1 = G.gcon(Loci::center, j, i, 0, 1) * alpha * alpha;
        P(prims::u1, k, j, i) = beta1 / alpha;

        // Now find new gamma and put it back in
        Real vsq = G.gcov(Loci::center, j, i, 1, 1) * P(prims::u1, k, j, i) * P(prims::u1, k, j, i) +
                   G.gcov(Loci::center, j, i, 2, 2) * P(prims::u2, k, j, i) * P(prims::u2, k, j, i) +
                   G.gcov(Loci::center, j, i, 3, 3) * P(prims::u3, k, j, i) * P(prims::u3, k, j, i) +
        2. * (G.gcov(Loci::center, j, i, 1, 2) * P(prims::u1, k, j, i) * P(prims::u2, k, j, i) +
              G.gcov(Loci::center, j, i, 1, 3) * P(prims::u1, k, j, i) * P(prims::u3, k, j, i) +
              G.gcov(Loci::center, j, i, 2, 3) * P(prims::u2, k, j, i) * P(prims::u3, k, j, i));
        
        if (fabs(vsq) < 1.e-13) vsq = 1.e-13;
        if (vsq >= 1.) {
            vsq = 1. - 1./(50.*50.);  // TODO DEFINE as max gamma
        }

        gamma = 1./sqrt(1. - vsq);

        P(prims::u1, k, j, i) *= gamma;
        P(prims::u2, k, j, i) *= gamma;
        P(prims::u3, k, j, i) *= gamma;
    }
}

/**
 * Zero mass flux at inner and outer boundaries,
 * and through the pole.
 * TODO Both may be unnecessary...
 */
TaskStatus FixFlux(MeshBlockData<Real> *rc)
{
    FLAG("Fixing fluxes");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const int ndim = pmb->pmy_mesh->ndim;

    GridVars F1, F2;
    F1 = rc->Get("c.c.bulk.cons").flux[X1DIR];
    if (ndim > 1) F2 = rc->Get("c.c.bulk.cons").flux[X2DIR];
    int je_e = (ndim > 1) ? je + 1 : je;
    int ke_e = (ndim > 2) ? ke + 1 : ke;
    const int nprim = F1.GetDim(4);

    if (pmb->packages.Get("GRMHD")->Param<bool>("fix_flux_inflow")) {
        if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::outflow)
        {
            pmb->par_for("fix_flux_in_l", ks, ke_e, js, je_e, is, is,
                KOKKOS_LAMBDA_3D {
                    F1(prims::rho, k, j, i) = min(F1(prims::rho, k, j, i), 0.);
                }
            );
        }

        if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow &&
            !(pmb->packages.Get("GRMHD")->Param<std::string>("problem") == "bondi"))
        {
            pmb->par_for("fix_flux_in_r", ks, ke_e, js, je_e, ie+1, ie+1,
                KOKKOS_LAMBDA_3D {
                    F1(prims::rho, k, j, i) = max(F1(prims::rho, k, j, i), 0.);
                }
            );
        }
    }

    if (pmb->packages.Get("GRMHD")->Param<bool>("fix_flux_pole")) {
        if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect)
        {
            pmb->par_for("fix_flux_pole_l", 0, nprim-1, ks, ke_e, js, js, is, ie+1,
                KOKKOS_LAMBDA_VARS {
                    F2(p, k, j, i) = 0.;
                }
            );
        }

        if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect)
        {
            pmb->par_for("fix_flux_pole_r", 0, nprim-1, ks, ke_e, je_e, je_e, is, ie+1,
                KOKKOS_LAMBDA_VARS {
                    F2(p, k, j, i) = 0.;
                }
            );
        }
    }

    FLAG("Fixed fluxes");
    return TaskStatus::complete;
}
