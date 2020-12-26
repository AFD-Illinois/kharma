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


TaskStatus ApplyCustomBoundaries(std::shared_ptr<MeshBlockData<Real>>& rc)
{
    auto pmb = rc->GetBlockPointer();
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    domain = IndexDomain::entire;
    int is_e = pmb->cellbounds.is(domain), ie_e = pmb->cellbounds.ie(domain);
    int js_e = pmb->cellbounds.js(domain), je_e = pmb->cellbounds.je(domain);
    int ks_e = pmb->cellbounds.ks(domain), ke_e = pmb->cellbounds.ke(domain);

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    // Big TODO: can these be implemented in the *conserved* variables?
    // It would save much headache & wailing


    // Put the reflecting condition into the corners tentatively, for non-border processes
    if(pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect) {
        FLAG("Inner X2 reflect");
        pmb->par_for("inner_x2_reflect", ks_e, ke_e, js_e, js-1, is_e, ie_e,
            KOKKOS_LAMBDA_3D {
                PLOOP {
                    Real reflect = ((p == prims::u2 || p == prims::B2) ? -1.0 : 1.0);
                    P(p, k, j, i) = reflect * P(p, k, (js - 1) + (js - j), i);
                }
                // Recover conserved vars
                p_to_u(G, P, eos, k, j, i, U);
            }
        );
    }
    if(pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect) {
        FLAG("Outer X2 reflect");
        pmb->par_for("outer_x2_reflect", ks_e, ke_e, je+1, je_e, is_e, ie_e,
            KOKKOS_LAMBDA_3D {
                PLOOP {
                    Real reflect = ((p == prims::u2 || p == prims::B2) ? -1.0 : 1.0);
                    P(p, k, j, i) = reflect * P(p, k, (je + 1) + (je - j), i);
                }
                // Recover conserved vars
                p_to_u(G, P, eos, k, j, i, U);
            }
        );
    }

    // Implement the outflow boundaries on the primitives, since the inflow check needs that
    if(pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::outflow) {
        FLAG("Inner X1 outflow");
        pmb->par_for("inner_x1_outflow", ks_e, ke_e, js_e, je_e, is_e, is-1,
            KOKKOS_LAMBDA_3D {
                // Apply boundary on primitives
                PLOOP {
                    P(p, k, j, i) = P(p, k, j, is);
                    if(p == prims::B1 || p == prims::B2 || p == prims::B3) {
                        P(p, k, j, i) *= G.gdet(Loci::center, j, is) / G.gdet(Loci::center, j, i);
                    }
                }
                // Inflow check
                check_inflow(G, P, k, j, i, 0);
                // Recover conserved vars
                p_to_u(G, P, eos, k, j, i, U);
            }
        );
    }
    if(pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow) {
        FLAG("Outer X1 outflow");
        pmb->par_for("outer_x1_outflow", ks_e, ke_e, js_e, je_e, ie+1, ie_e,
            KOKKOS_LAMBDA_3D {
                // Apply boundary on primitives
                PLOOP {
                    P(p, k, j, i) = P(p, k, j, ie);
                    if(p == prims::B1 || p == prims::B2 || p == prims::B3) {
                        P(p, k, j, i) *= G.gdet(Loci::center, j, ie) / G.gdet(Loci::center, j, i);
                    }
                }
                // Inflow check
                check_inflow(G, P, k, j, i, 1);
                // Recover conserved vars
                p_to_u(G, P, eos, k, j, i, U);
            }
        );
    }

    if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow &&
        pmb->packages["GRMHD"]->Param<std::string>("problem") == "bondi") {
        FLAG("Bondi outer boundary");
        ApplyBondiBoundary(rc);
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
 * Zero any flux through the pole, and ensure flux CT preserves divB there
 */
TaskStatus FixFlux(std::shared_ptr<MeshBlockData<Real>>& rc)
{
    FLAG("Fixing boundary fluxes");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    int ndim = 3;
    if (js == je) {
        ndim = 1;
    } else if (ks == ke) {
        ndim = 2;
    }

    GridVars F1, F2, F3;
    F1 = rc->Get("c.c.bulk.cons").flux[X1DIR];
    if (ndim > 1) F2 = rc->Get("c.c.bulk.cons").flux[X2DIR];
    if (ndim > 2) F3 = rc->Get("c.c.bulk.cons").flux[X3DIR];
    int je_e = (ndim > 1) ? je + 1 : je;
    int ke_e = (ndim > 2) ? ke + 1 : ke;

    if (pmb->packages["GRMHD"]->Param<bool>("fix_flux_inflow") == true) {
        if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::outflow)
        {
            pmb->par_for("fix_flux_in_l", ks, ke_e, js, je_e, is, is,
                KOKKOS_LAMBDA_3D {
                    F1(prims::rho, k, j, i) = min(F1(prims::rho, k, j, i), 0.);
                }
            );
        }

        if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow &&
            !(pmb->packages["GRMHD"]->Param<std::string>("problem") == "bondi"))
        {
            pmb->par_for("fix_flux_in_r", ks, ke_e, js, je_e, ie+1, ie+1,
                KOKKOS_LAMBDA_3D {
                    F1(prims::rho, k, j, i) = max(F1(prims::rho, k, j, i), 0.);
                }
            );
        }
    }

    if (pmb->packages["GRMHD"]->Param<bool>("fix_flux_B") == true) {
        if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect)
        {
            pmb->par_for("fix_flux_b_l", ks, ke_e, js, js, is, ie+1,
                KOKKOS_LAMBDA_3D {
                    PLOOP F2(p, k, j, i) = 0.;
                    // Make sure the emfs are also 0, for flux-ct
                    F1(prims::B2, k, j-1, i) = -F1(prims::B2, k, js, i);
                    if (ke_e > 0) F3(prims::B2, k, j-1, i) = -F3(prims::B2, k, js, i);
                }
            );
        }

        if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect)
        {
            pmb->par_for("fix_flux_b_r", ks, ke_e, je_e, je_e, is, ie+1,
                KOKKOS_LAMBDA_3D {
                    PLOOP F2(p, k, j, i) = 0.;
                    // Make sure the emfs are also 0, for flux-ct
                    F1(prims::B2, k, j, i) = -F1(prims::B2, k, je, i);
                    if (ke_e > 0) F3(prims::B2, k, j, i) = -F3(prims::B2, k, je, i);
                }
            );
        }
    }

    FLAG("Fixed boundary fluxes");
    return TaskStatus::complete;
}
