/*
 *  File: seed_B.hpp
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
#pragma once

#include "seed_B.hpp"

#include "b_flux_ct.hpp"
#include "b_ct.hpp"
#include "boundaries.hpp"
#include "domain.hpp"
#include "fm_torus.hpp"

template <BSeedType Seed>
TaskStatus SeedBFieldType(MeshBlockData<Real> *rc, ParameterInput *pin, IndexDomain domain = IndexDomain::entire)
{
    auto pmb = rc->GetBlockPointer();
    auto pkgs = pmb->packages.AllPackages();

    // Fields
    GridScalar rho = rc->Get("prims.rho").data;
    const auto &G = pmb->coords;

    // Parameters
    std::string b_field_type = pin->GetString("b_field", "type");
    auto prob = pin->GetString("parthenon/job", "problem_id");
    bool is_torus = (prob == "torus");

    // Indices
    IndexRange3 b = KDomain::GetRange(rc, domain);
    int ndim = pmb->pmy_mesh->ndim;

    // Shortcut to field values for easy fields
    if constexpr (Seed == BSeedType::constant ||
                  Seed == BSeedType::monopole ||
                  Seed == BSeedType::monopole_cube)
    {
        if (pkgs.count("B_CT"))
        {
            auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
            Real b10 = pin->GetOrAddReal("b_field", "b10", 0.);
            Real b20 = pin->GetOrAddReal("b_field", "b20", 0.);
            Real b30 = pin->GetOrAddReal("b_field", "b30", 0.);
            // Fill at 3 different locations
            // TODO this would need to be extended for domain < entire
            pmb->par_for(
                "B_field_B", b.ks, b.ke, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                    GReal Xembed[GR_DIM];
                    G.coord_embed(k, j, i, Loci::face1, Xembed);
                    GReal gdet = G.gdet(Loci::face1, j, i);
                    double tmp1, tmp2;
                    seed_b<Seed>(Xembed, gdet, b10, b20, b30,
                                 B_Uf(F1, 0, k, j, i), tmp1, tmp2);

                    G.coord_embed(k, j, i, Loci::face2, Xembed);
                    gdet = G.gdet(Loci::face2, j, i);
                    seed_b<Seed>(Xembed, gdet, b10, b20, b30,
                                 tmp1, B_Uf(F2, 0, k, j, i), tmp2);

                    G.coord_embed(k, j, i, Loci::face3, Xembed);
                    gdet = G.gdet(Loci::face3, j, i);
                    seed_b<Seed>(Xembed, gdet, b10, b20, b30,
                                 tmp1, tmp2, B_Uf(F3, 0, k, j, i));
                });
            // Update primitive variables
            B_CT::BlockUtoP(rc, domain);
        }
        else if (pkgs.count("B_FluxCT"))
        {
            GridVector B_P = rc->Get("prims.B").data;
            Real b10 = pin->GetOrAddReal("b_field", "b10", 0.);
            Real b20 = pin->GetOrAddReal("b_field", "b20", 0.);
            Real b30 = pin->GetOrAddReal("b_field", "b30", 0.);
            pmb->par_for(
                "B_field_B", b.ks, b.ke, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                    GReal Xembed[GR_DIM];
                    G.coord_embed(k, j, i, Loci::center, Xembed);
                    const GReal gdet = G.gdet(Loci::center, j, i);
                    seed_b<Seed>(Xembed, gdet, b10, b20, b30,
                                 B_P(V1, k, j, i),
                                 B_P(V2, k, j, i),
                                 B_P(V3, k, j, i));
                });
            // We still need to update conserved flux values, but then we're done
            B_FluxCT::BlockPtoU(rc, domain);
        }
        return TaskStatus::complete;
    }

    // Require and load what we need if necessary
    // TODO this seems very inelegant. Also most of these should support non-FM-torii
    // as long as we don't call fm_torus_rho below
    Real a, rin, rmax, gam, kappa, rho_norm;
    Real tilt = 0; // Needs to be initialized
    switch (Seed)
    {
    case BSeedType::sane:
    case BSeedType::mad:
    case BSeedType::mad_quadrupole:
    case BSeedType::r3s3:
    case BSeedType::r5s5:
    case BSeedType::gaussian:
        if (!is_torus)
            throw std::invalid_argument("Magnetic field seed " + b_field_type + " supports only torus problems!");
        // Torus parameters
        rin = pin->GetReal("torus", "rin");
        rmax = pin->GetReal("torus", "rmax");
        kappa = pin->GetReal("torus", "kappa");
        tilt = pin->GetReal("torus", "tilt") / 180. * M_PI;
        // Other things we need only for torus evaluation
        gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
        rho_norm = pmb->packages.Get("GRMHD")->Param<Real>("rho_norm");
        a = G.coords.get_a();
        break;
    default:
        break;
    }

    Real A0 = pin->GetOrAddReal("b_field", "A0", 0.);
    Real min_A = pin->GetOrAddReal("b_field", "min_A", 0.2); // TODO back compat?  Doubtful was used

    // For all other fields...
    // Find the magnetic vector potential.  In X3 symmetry only A_phi is non-zero,
    // But for tilted conditions we must keep track of all components
    IndexSize3 sz = KDomain::GetBlockSize(rc);
    ParArrayND<double> A("A", NVEC, sz.n3, sz.n2, sz.n1);
    pmb->par_for(
        "B_field_A", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
            GReal Xnative[GR_DIM];
            GReal Xembed[GR_DIM], Xmidplane[GR_DIM];
            G.coord(k, j, i, Loci::corner, Xnative);
            G.coord_embed(k, j, i, Loci::corner, Xembed);
            // What are our corresponding "midplane" values for evaluating the function?
            rotate_polar(Xembed, tilt, Xmidplane);
            const GReal r = Xmidplane[1], th = Xmidplane[2];

            // This is written under the assumption re-computed rho is more accurate than a bunch
            // of averaging in a meaningful way.  Just use the average if not.
            Real rho_av;
            if (is_torus)
            {
                // Find rho at corner directly for torii
                rho_av = fm_torus_rho(a, rin, rmax, gam, kappa, r, th) / rho_norm;
            }
            else
            {
                // Use averages for anything else
                // This loop runs over every corner. Centers do not exist before the first
                // or after the last, so use the last (ghost) zones available.
                const int ii = clip((uint)i, b.is + 1, b.ie);
                const int jj = clip((uint)j, b.js + 1, b.je);
                const int kk = clip((uint)k, b.ks + 1, b.ke);
                if (ndim > 2)
                {
                    rho_av = (rho(kk, jj, ii) + rho(kk, jj, ii - 1) +
                              rho(kk, jj - 1, ii) + rho(kk, jj - 1, ii - 1) +
                              rho(kk - 1, jj, ii) + rho(kk - 1, jj, ii - 1) +
                              rho(kk - 1, jj - 1, ii) + rho(kk - 1, jj - 1, ii - 1)) /
                             8;
                }
                else
                {
                    rho_av = (rho(kk, jj, ii) + rho(kk, jj, ii - 1) +
                              rho(kk, jj - 1, ii) + rho(kk, jj - 1, ii - 1)) /
                             4;
                }
            }

            Real Aphi = seed_a<Seed>(Xmidplane, rho_av, rin, min_A, A0);

            if (tilt != 0.0)
            {
                // This is *covariant* A_mu of an untilted disk
                const double A_untilt_lower[GR_DIM] = {0., 0., 0., Aphi};
                // Raise to contravariant vector, since rotate_polar_vec will need that.
                // Note we have to do this in the midplane!
                // The coord_to_native calculation involves an iterative solve for MKS/FMKS
                GReal Xnative_midplane[GR_DIM] = {0}, gcon_midplane[GR_DIM][GR_DIM] = {0};
                G.coords.coord_to_native(Xmidplane, Xnative_midplane);
                G.coords.gcon_native(Xnative_midplane, gcon_midplane);
                double A_untilt[GR_DIM] = {0};
                DLOOP2 A_untilt[mu] += gcon_midplane[mu][nu] * A_untilt_lower[nu];

                // Then rotate
                double A_tilt[GR_DIM] = {0};
                double A_untilt_embed[GR_DIM] = {0}, A_tilt_embed[GR_DIM] = {0};
                G.coords.con_vec_to_embed(Xnative_midplane, A_untilt, A_untilt_embed);
                rotate_polar_vec(Xmidplane, A_untilt_embed, -tilt, Xembed, A_tilt_embed);
                G.coords.con_vec_to_native(Xnative, A_tilt_embed, A_tilt);

                // Lower the result as we need curl(A_mu).  Done at local zone.
                double A_tilt_lower[GR_DIM] = {0};
                G.lower(A_tilt, A_tilt_lower, k, j, i, Loci::corner);
                VLOOP A(v, k, j, i) = A_tilt_lower[1 + v];
            }
            else
            {
                // Some problems rely on a very accurate A->B, which the rotation lacks.
                // So, we preserve exact values in the no-tilt case.
                A(V3, k, j, i) = Aphi;
            }
        });

    if (pkgs.count("B_CT"))
    {
        auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
        // This fills a couple zones outside the exact interior with bad data
        // Careful of that w/e.g. Dirichlet bounds.
        IndexRange3 bB = KDomain::GetRange(rc, domain, 0, -1);
        if (ndim > 2)
        {
            pmb->par_for(
                "ot_B", bB.ks, bB.ke, bB.js, bB.je, bB.is, bB.ie,
                KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                    B_CT::curl_3D(G, A, B_Uf, k, j, i);
                });
        }
        else if (ndim > 1)
        {
            pmb->par_for(
                "ot_B", bB.ks, bB.ke, bB.js, bB.je, bB.is, bB.ie,
                KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                    B_CT::curl_2D(G, A, B_Uf, k, j, i);
                });
        }
        else
        {
            throw std::runtime_error("Must initialize 1D field directly!");
        }
        B_CT::BlockUtoP(rc, domain);
    }
    else if (pkgs.count("B_FluxCT"))
    {
        // Calculate B-field
        GridVector B_U = rc->Get("cons.B").data;
        IndexRange3 bl = KDomain::GetRange(rc, domain, 0, -1); // TODO will need changes if domain < entire
        if (ndim > 2)
        {
            pmb->par_for(
                "B_field_B_3D", bl.ks, bl.ke, bl.js, bl.je, bl.is, bl.ie,
                KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                    B_FluxCT::averaged_curl_3D(G, A, B_U, k, j, i);
                });
        }
        else if (ndim > 1)
        {
            pmb->par_for(
                "B_field_B_2D", bl.ks, bl.ke, bl.js, bl.je, bl.is, bl.ie,
                KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                    B_FluxCT::averaged_curl_2D(G, A, B_U, k, j, i);
                });
        }
        else
        {
            throw std::runtime_error("Must initialize 1D field directly!");
        }
        // Finally, make sure we initialize the primitive field too
        B_FluxCT::BlockUtoP(rc, domain);
    }

    return TaskStatus::complete;
}