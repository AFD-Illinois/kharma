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
#include "b_flux_ct_functions.hpp"
#include "b_cd_glm_functions.hpp"

#include "basic_types.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

KOKKOS_INLINE_FUNCTION void check_inflow(const GRCoordinates &G, GridVars P, const int& k, const int& j, const int& i, int type);


TaskStatus ApplyCustomBoundaries(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVector B_P = rc->Get("c.c.bulk.B_prim").data;
    GridVector B_U = rc->Get("c.c.bulk.B_con").data;
    const bool use_b_cd_glm = pmb->packages.AllPackages().count("B_CD_GLM") > 0;
    const bool use_b_flux_ct = pmb->packages.AllPackages().count("B_FluxCT") > 0;
    auto& G = pmb->coords;

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    domain = IndexDomain::entire;
    int is_e = pmb->cellbounds.is(domain), ie_e = pmb->cellbounds.ie(domain);
    int js_e = pmb->cellbounds.js(domain), je_e = pmb->cellbounds.je(domain);
    int ks_e = pmb->cellbounds.ks(domain), ke_e = pmb->cellbounds.ke(domain);

    EOS* eos = pmb->packages.Get("GRMHD")->Param<EOS*>("eos");

    // Big TODO: how many of these can be implemented in the *conserved* variables?
    // It would save much headache & wailing

    // Put the reflecting condition into the corners tentatively, for non-border processes
    if(pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect) {
        FLAG("Inner X2 reflect");
        pmb->par_for("inner_x2_reflect", ks_e, ke_e, js_e, js-1, is, ie,
            KOKKOS_LAMBDA_3D {
                PLOOP {
                    Real reflect = ((p == prims::u2) ? -1.0 : 1.0);
                    P(p, k, j, i) = reflect * P(p, k, (js - 1) + (js - j), i);
                }
                VLOOP B_P(v, k, j, i) = ((v == 1) ? -1.0 : 1.0) * B_P(v, k, (js - 1) + (js - j), i);
                // Recover conserved vars
                GRMHD::p_to_u(G, P, B_P, eos, k, j, i, U);
                if (use_b_flux_ct) B_FluxCT::p_to_u(G, B_P, k, j, i, B_U);
            }
        );
    }
    if(pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect) {
        FLAG("Outer X2 reflect");
        pmb->par_for("outer_x2_reflect", ks_e, ke_e, je+1, je_e, is, ie,
            KOKKOS_LAMBDA_3D {
                PLOOP {
                    Real reflect = ((p == prims::u2) ? -1.0 : 1.0);
                    P(p, k, j, i) = reflect * P(p, k, (je + 1) + (je - j), i);
                }
                VLOOP B_P(v, k, j, i) = ((v == 1) ? -1.0 : 1.0) * B_P(v, k, (je + 1) + (je - j), i);
                // Recover conserved vars
                GRMHD::p_to_u(G, P, B_P, eos, k, j, i, U);
                if (use_b_flux_ct) B_FluxCT::p_to_u(G, B_P, k, j, i, B_U);
            }
        );
    }

    // Implement the outflow boundaries on the primitives, since the inflow check needs that
    if(pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::outflow) {
        FLAG("Inner X1 outflow");
        pmb->par_for("inner_x1_outflow", ks_e, ke_e, js_e, je_e, is_e, is-1,
            KOKKOS_LAMBDA_3D {
                // Apply boundary on primitives
                PLOOP P(p, k, j, i) = P(p, k, j, is);
                VLOOP B_P(v, k, j, i) = B_P(v, k, j, is) * G.gdet(Loci::center, j, is) / G.gdet(Loci::center, j, i);
                // Inflow check
                check_inflow(G, P, k, j, i, 0);
                // Recover conserved vars
                GRMHD::p_to_u(G, P, B_P, eos, k, j, i, U);
                if (use_b_flux_ct) B_FluxCT::p_to_u(G, B_P, k, j, i, B_U);
            }
        );
    }
    if(pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow) {
        if (pmb->packages.Get("GRMHD")->Param<std::string>("problem") == "bondi") {
            FLAG("Bondi outer boundary");
            ApplyBondiBoundary(rc);
        } else {
            FLAG("Outer X1 outflow");
            pmb->par_for("outer_x1_outflow", ks_e, ke_e, js_e, je_e, ie+1, ie_e,
                KOKKOS_LAMBDA_3D {
                    // Apply boundary on primitives
                    PLOOP P(p, k, j, i) = P(p, k, j, ie);
                    VLOOP B_P(v, k, j, i) = B_P(v, k, j, ie) * G.gdet(Loci::center, j, ie) / G.gdet(Loci::center, j, i);
                    // Inflow check
                    check_inflow(G, P, k, j, i, 1);
                    // Recover conserved vars
                    GRMHD::p_to_u(G, P, B_P, eos, k, j, i, U);
                    if (use_b_flux_ct) B_FluxCT::p_to_u(G, B_P, k, j, i, B_U);
                }
            );
        }
    }

    // This is a bit dumb for scoping
    if (use_b_cd_glm) {
        GridScalar psi_p = rc->Get("c.c.bulk.psi_cd_prim").data;
        GridScalar psi_u = rc->Get("c.c.bulk.psi_cd_con").data;

        if(pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect) {
            pmb->par_for("inner_x2_psi", ks_e, ke_e, js_e, js-1, is, ie,
                KOKKOS_LAMBDA_3D {
                    //psi_p(k, j, i) = -psi_p(k, (js - 1) + (js - j), i);
                    psi_p(k, j, i) = psi_p(k, js, i);
                    B_CD_GLM::p_to_u(G, B_P, psi_p, k, j, i, B_U, psi_u);
                }
            );
        }
        if(pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect) {
            pmb->par_for("outer_x2_psi", ks_e, ke_e, je+1, je_e, is, ie,
                KOKKOS_LAMBDA_3D {
                    //psi_p(k, j, i) = -psi_p(k, (je + 1) + (je - j), i);
                    psi_p(k, j, i) = psi_p(k, je, i);
                    B_CD_GLM::p_to_u(G, B_P, psi_p, k, j, i, B_U, psi_u);
                }
            );
        }if(pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::outflow) {
            pmb->par_for("inner_x1_psi", ks_e, ke_e, js_e, je_e, is_e, is-1,
                KOKKOS_LAMBDA_3D {
                    psi_p(k, j, i) = psi_p(k, j, is);
                    B_CD_GLM::p_to_u(G, B_P, psi_p, k, j, i, B_U, psi_u);
                }
            );
        }
        if(pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow) {
            pmb->par_for("outer_x1_psi", ks_e, ke_e, js_e, je_e, ie+1, ie_e,
                KOKKOS_LAMBDA_3D {
                    psi_p(k, j, i) = psi_p(k, j, ie);
                    B_CD_GLM::p_to_u(G, B_P, psi_p, k, j, i, B_U, psi_u);
                }
            );
        }
    }

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
        // TODO abridge? Delete or record special cases? Are loops faster?
        Real vsq = G.gcov(Loci::center, j, i, 1, 1) * P(prims::u1, k, j, i) * P(prims::u1, k, j, i) +
                   G.gcov(Loci::center, j, i, 2, 2) * P(prims::u2, k, j, i) * P(prims::u2, k, j, i) +
                   G.gcov(Loci::center, j, i, 3, 3) * P(prims::u3, k, j, i) * P(prims::u3, k, j, i) +
        2. * (G.gcov(Loci::center, j, i, 1, 2) * P(prims::u1, k, j, i) * P(prims::u2, k, j, i) +
              G.gcov(Loci::center, j, i, 1, 3) * P(prims::u1, k, j, i) * P(prims::u3, k, j, i) +
              G.gcov(Loci::center, j, i, 2, 3) * P(prims::u2, k, j, i) * P(prims::u3, k, j, i));
        
        if (fabs(vsq) < 1.e-13) vsq = 1.e-13;
        if (vsq >= 1.) {
            vsq = 1. - 1./(50.*50.);
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
