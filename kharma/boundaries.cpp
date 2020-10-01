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
#include "phys.hpp"

#include "basic_types.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

KOKKOS_INLINE_FUNCTION void check_inflow(const GRCoordinates &G, GridVars P, const int& k, const int& j, const int& i, int type);


TaskStatus ApplyCustomBoundaries(std::shared_ptr<Container<Real>>& rc)
{
    auto pmb = rc->GetBlockPointer();
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    // Implement the outflow boundaries on the primitives, since the inflow check needs that
    if(pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::outflow) {
        pmb->par_for("inner_x1_outflow", kb.s, kb.e, jb.s, jb.e, 0, ib.s-1,
            KOKKOS_LAMBDA_3D {
                // Apply boundary on primitives
                PLOOP {
                    P(p, k, j, i) = P(p, k, j, ib.s);
                    if(p == prims::B1 || p == prims::B2 || p == prims::B3) {
                        P(p, k, j, i) *= G.gdet(Loci::center, j, ib.s) / G.gdet(Loci::center, j, i);
                    }
                }
                // Inflow check
                check_inflow(G, P, k, j, i, 0);
                // Recover conserved vars
                FourVectors Dtmp;
                get_state(G, P, k, j, i, Loci::center, Dtmp);
                prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
            }
        );
    }
    if(pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow) {
        pmb->par_for("outer_x1_outflow", kb.s, kb.e, jb.s, jb.e, ib.e+1, n1-1,
            KOKKOS_LAMBDA_3D {
                // Apply boundary on primitives
                PLOOP {
                    P(p, k, j, i) = P(p, k, j, ib.e);
                    if(p == prims::B1 || p == prims::B2 || p == prims::B3) {
                        P(p, k, j, i) *= G.gdet(Loci::center, j, ib.e) / G.gdet(Loci::center, j, i);
                    }
                }
                // Inflow check
                check_inflow(G, P, k, j, i, 1);
                // Recover conserved vars
                FourVectors Dtmp;
                get_state(G, P, k, j, i, Loci::center, Dtmp);
                prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
            }
        );
    }

    // Implement our own reflecting boundary for our variables. TODO does this work in conserved?
    if(pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect) {
        pmb->par_for("inner_x2_reflect", 0, NPRIM-1, kb.s, kb.e, 0, jb.s-1, 0, n1-1,
            KOKKOS_LAMBDA_VARS {
                Real reflect = ((p == prims::u2 || p == prims::B2) ? -1.0 : 1.0);
                P(p, k, j, i) = reflect * P(p, k, jb.s + (jb.s - j - 1), i);
                // Recover conserved vars
                FourVectors Dtmp;
                get_state(G, P, k, j, i, Loci::center, Dtmp);
                prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
            }
        );
    }
    if(pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect) {
        pmb->par_for("outer_x2_reflect", 0, NPRIM-1, kb.s, kb.e, jb.e+1, n2-1, 0, n1-1,
            KOKKOS_LAMBDA_VARS {
                Real reflect = ((p == prims::u2 || p == prims::B2) ? -1.0 : 1.0);
                P(p, k, j, i) = reflect * P(p, k, jb.e + (jb.e - j + 1), i);
                // Recover conserved vars
                FourVectors Dtmp;
                get_state(G, P, k, j, i, Loci::center, Dtmp);
                prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
            }
        );
    }

    if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow &&
        pmb->packages["GRMHD"]->Param<std::string>("problem") == "bondi") {
        FLAG("Applying Bondi problem boundary");
        ApplyBondiBoundary(rc);
    }

    DelEOS(eos);
    return TaskStatus::complete;
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
    ucon_calc(G, P, k, j, i, Loci::center, ucon);

    if (((ucon[1] > 0.) && (type == 0)) ||
        ((ucon[1] < 0.) && (type == 1)))
    {
        // Find gamma and remove it from primitive velocity
        // TODO check failures?
        double gamma = mhd_gamma_calc(G, P, k, j, i, Loci::center);
        P(prims::u1, k, j, i) /= gamma;
        P(prims::u2, k, j, i) /= gamma;
        P(prims::u3, k, j, i) /= gamma;

        // Reset radial velocity so radial 4-velocity is zero
        Real alpha = 1 / G.gcon(Loci::center, j, i, 0, 0);
        Real beta1 = G.gcon(Loci::center, j, i, 0, 1) * alpha * alpha;
        P(prims::u1, k, j, i) = beta1 / alpha;

        // Now find new gamma and put it back in
        gamma = mhd_gamma_calc(G, P, k, j, i, Loci::center);

        P(prims::u1, k, j, i) *= gamma;
        P(prims::u2, k, j, i) *= gamma;
        P(prims::u3, k, j, i) *= gamma;
    }
}

/**
 * Fix fluxes on domain boundaries. No inflow, correct B fields on reflecting conditions.
 * TODO Parthenon does this, if given to understand B is a vector
 */
TaskStatus FixFlux(std::shared_ptr<Container<Real>>& rc)
{
    FLAG("Fixing boundary fluxes");
    auto pmb = rc->GetBlockPointer();
    GridVars F1 = rc->Get("c.c.bulk.cons").flux[X1DIR];
    GridVars F2 = rc->Get("c.c.bulk.cons").flux[X2DIR];
    GridVars F3 = rc->Get("c.c.bulk.cons").flux[X3DIR];

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    domain = IndexDomain::entire;
    int is_e = pmb->cellbounds.is(domain), ie_e = pmb->cellbounds.ie(domain);
    int js_e = pmb->cellbounds.js(domain), je_e = pmb->cellbounds.je(domain);
    int ks_e = pmb->cellbounds.ks(domain), ke_e = pmb->cellbounds.ke(domain);

    // TODO option to allow inflow?
    if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::outflow)
    {
        pmb->par_for("fix_flux_in_l", ks_e, ke_e, js_e, je_e,
            KOKKOS_LAMBDA (const int& k, const int& j) {
                F1(prims::rho, k, j, is) = min(F1(prims::rho, k, j, is), 0.);
            }
        );
    }

    if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow &&
        !(pmb->packages["GRMHD"]->Param<std::string>("problem") == "bondi"))
    {
        pmb->par_for("fix_flux_in_r", ks_e, ke_e, js_e, je_e,
            KOKKOS_LAMBDA (const int& k, const int& j) {
                F1(prims::rho, k, j, ie+1) = max(F1(prims::rho, k, j, ie+1), 0.);
            }
        );
    }

    if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect)
    {
        pmb->par_for("fix_flux_b_l", ks_e, ke_e, is_e, ie_e,
            KOKKOS_LAMBDA (const int& k, const int& i) {
                F1(prims::B2, k, js - 1, i) = -F1(prims::B2, k, js, i);
                F3(prims::B2, k, js - 1, i) = -F3(prims::B2, k, js, i);
                PLOOP F2(p, k, js, i) = 0.;
            }
        );
    }

    if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect)
    {
        pmb->par_for("fix_flux_b_r", ks_e, ke_e, is_e, ie_e,
            KOKKOS_LAMBDA (const int& k, const int& i) {
                F1(prims::B2, k, je + 1, i) = -F1(prims::B2, k, je, i);
                F3(prims::B2, k, je + 1, i) = -F3(prims::B2, k, je, i);
                PLOOP F2(p, k, je + 1, i) = 0.;
            }
        );
    }

    FLAG("Fixed boundary fluxes");
    return TaskStatus::complete;
}