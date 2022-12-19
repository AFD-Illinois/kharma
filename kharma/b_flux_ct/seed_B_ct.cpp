/* 
 *  File: seed_B_ct.cpp
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

// Seed a torus of some type with a magnetic field according to its density

#include "seed_B_ct.hpp"

#include "b_field_tools.hpp"
#include "b_flux_ct.hpp"
#include "fm_torus.hpp"
#include "grmhd_functions.hpp"
#include "prob_common.hpp"

using namespace parthenon;

TaskStatus B_FluxCT::SeedBField(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();

    const auto& G = pmb->coords;
    GridScalar rho = rc->Get("prims.rho").data;
    GridVector B_P = rc->Get("prims.B").data;
    GridVector B_U = rc->Get("cons.B").data;

    Real min_rho_q = pin->GetOrAddReal("b_field", "min_rho_q", 0.2);
    std::string b_field_type = pin->GetString("b_field", "type");
    // Translate the type to an enum so we can avoid string comp inside,
    // as well as for good errors, many->one maps, etc.
    BSeedType b_field_flag = ParseBSeedType(b_field_type);

    // Other parameters we need
    auto prob = pin->GetString("parthenon/job", "problem_id");
    bool is_torus = (prob == "torus");

    // Require and load what we need if necessary
    Real a, rin, rmax, gam, kappa, rho_norm;
    Real tilt = 0; // Needs to be initialized
    Real b10 = 0, b20 = 0, b30 = 0, bz = 0;
    switch (b_field_flag)
    {
    case BSeedType::constant:
        b10 = pin->GetOrAddReal("b_field", "b10", 0.);
        b20 = pin->GetOrAddReal("b_field", "b20", 0.);
        b30 = pin->GetOrAddReal("b_field", "b30", 0.);
        break;
    case BSeedType::monopole:
        b10 = pin->GetReal("b_field", "b10");
        break;
    case BSeedType::sane:
    case BSeedType::ryan:
    case BSeedType::ryan_quadrupole:
    case BSeedType::r3s3:
    case BSeedType::steep:
    case BSeedType::gaussian:
        if (!is_torus)
            throw std::invalid_argument("Magnetic field seed "+b_field_type+" supports only torus problems!");
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
    case BSeedType::bz_monopole:
        break;
    case BSeedType::vertical:
        bz = pin->GetOrAddReal("b_field", "bz", 0.);
        break;
    }

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    int ndim = pmb->pmy_mesh->ndim;

    // Shortcut to field values for easy fields
    if (b_field_flag == BSeedType::constant) {
        pmb->par_for("B_field_B", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA_3D {
                // Set B1 directly
                B_P(V1, k, j, i) = b10;
                B_P(V2, k, j, i) = b20;
                B_P(V3, k, j, i) = b30;
            }
        );
        B_FluxCT::PtoU(rc);
        return TaskStatus::complete;
    } else if (b_field_flag == BSeedType::monopole) {
        pmb->par_for("B_field_B", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA_3D {
                // Set B1 directly by normalizing
                B_P(V1, k, j, i) = b10 / G.gdet(Loci::center, j, i);
                B_P(V2, k, j, i) = 0.;
                B_P(V3, k, j, i) = 0.;
            }
        );
        B_FluxCT::PtoU(rc);
        return TaskStatus::complete;
    }

    // Find the magnetic vector potential.  In X3 symmetry only A_phi is non-zero,
    // But for tilted conditions we must keep track of all components
    // TODO there should be an ncornersi,j,k
    ParArrayND<double> A("A", NVEC, n3+1, n2+1, n1+1);
    pmb->par_for("B_field_A", ks, ke+1, js, je+1, is, ie+1,
        KOKKOS_LAMBDA_3D {
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
            if (is_torus) {
                // Find rho (later u?) at corner directly for torii
                rho_av = fm_torus_rho(a, rin, rmax, gam, kappa, r, th) / rho_norm;
            } else {
                // Use averages for anything else
                // This loop runs over every corner. Centers do not exist before the first
                // or after the last, so use the last (ghost) zones available.
                const int ii = clip(i, is+1, ie);
                const int jj = clip(j, js+1, je);
                const int kk = clip(k, ks+1, ke);
                if (ndim > 2) {
                    rho_av = (rho(kk, jj, ii)     + rho(kk, jj, ii - 1) +
                              rho(kk, jj - 1, ii) + rho(kk, jj - 1, ii - 1) +
                              rho(kk - 1, jj, ii)     + rho(kk - 1, jj, ii - 1) +
                              rho(kk - 1, jj - 1, ii) + rho(kk - 1, jj - 1, ii - 1)) / 8;
                } else {
                    rho_av = (rho(ks, jj, ii)     + rho(ks, jj, ii - 1) +
                              rho(ks, jj - 1, ii) + rho(ks, jj - 1, ii - 1)) / 4;
                }
            }

            Real q;
            switch (b_field_flag)
            {
            case BSeedType::sane:
                q = m::max(rho_av - min_rho_q, 0.);
                break;
            case BSeedType::bz_monopole:
                // used in testing to exactly agree with harmpi
                q = 1. - m::cos(th);
                break;
            case BSeedType::ryan:
                // BR's smoothed poloidal in-torus, EHT standard MAD
                q = m::max(m::pow(r / rin, 3) * m::pow(sin(th), 3) * m::exp(-r / 400) * rho_av - min_rho_q, 0.);
                break;
            case BSeedType::ryan_quadrupole:
                // BR's smoothed poloidal in-torus, but turned into a quadrupole
                q = m::max(pow(r / rin, 3) * m::pow(sin(th), 3) * m::exp(-r / 400) * rho_av - min_rho_q, 0.) * m::cos(th);
                break;
            case BSeedType::r3s3:
                // Just the r^3 sin^3 th term
                q = m::max(m::pow(r / rin, 3) * m::pow(m::sin(th), 3) * rho_av - min_rho_q, 0.);
                break;
            case BSeedType::steep:
                // Bump power to r^5 sin^5 th term, quieter MAD
                q = m::max(m::pow(r / rin, 5) * m::pow(m::sin(th), 5) * rho_av - min_rho_q, 0.);
                break;
            case BSeedType::gaussian:
                // Pure vertical threaded field of gaussian strength with FWHM 2*rin (i.e. HM@rin)
                // centered at BH center
                // Block is to avoid compiler whinging about initialization
                {
                    Real x = (r / rin) * m::sin(th);
                    Real sigma = 2 / m::sqrt(2 * m::log(2));
                    Real u = x / m::abs(sigma);
                    q = (1 / (m::sqrt(2 * M_PI) * m::abs(sigma))) * m::exp(-u * u / 2);
                }
                break;
            case BSeedType::vertical:
                q = bz * r * m::sin(th);
            default:
                // This shouldn't be reached. Squawk here?
                break;
            }

            if (tilt != 0.0) {
                // This is *covariant* A_mu of an untilted disk
                const double A_untilt_lower[GR_DIM] = {0., 0., 0., q};
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
                VLOOP A(v, k, j, i) = A_tilt_lower[1+v];
            } else {
                // Some problems rely on a very accurate A->B, which the 
				A(V3, k, j, i) = q;
            }
        }
    );

    // Calculate B-field
    if (ndim > 2) {
        pmb->par_for("B_field_B_3D", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA_3D {
                // Take a flux-ct step from the corner potentials.
                // This needs to be 3D because post-tilt A may not point in the phi direction only

                // A3,2 derivative
                const Real A3c2f = (A(V3, k, j + 1, i)     + A(V3, k, j + 1, i + 1) + 
                                    A(V3, k + 1, j + 1, i) + A(V3, k + 1, j + 1, i + 1)) / 4;
                const Real A3c2b = (A(V3, k, j, i)     + A(V3, k, j, i + 1) +
                                    A(V3, k + 1, j, i) + A(V3, k + 1, j, i + 1)) / 4;
                // A2,3 derivative
                const Real A2c3f = (A(V2, k + 1, j, i)     + A(V2, k + 1, j, i + 1) +
                                    A(V2, k + 1, j + 1, i) + A(V2, k + 1, j + 1, i + 1)) / 4;
                const Real A2c3b = (A(V2, k, j, i)     + A(V2, k, j, i + 1) +
                                    A(V2, k, j + 1, i) + A(V2, k, j + 1, i + 1)) / 4;
                B_U(V1, k, j, i) = (A3c2f - A3c2b) / G.dx2v(j) - (A2c3f - A2c3b) / G.dx3v(k);

                // A1,3 derivative
                const Real A1c3f = (A(V1, k + 1, j, i)     + A(V1, k + 1, j, i + 1) + 
                                    A(V1, k + 1, j + 1, i) + A(V1, k + 1, j + 1, i + 1)) / 4;
                const Real A1c3b = (A(V1, k, j, i)     + A(V1, k, j, i + 1) +
                                    A(V1, k, j + 1, i) + A(V1, k, j + 1, i + 1)) / 4;
                // A3,1 derivative
                const Real A3c1f = (A(V3, k, j, i + 1)     + A(V3, k + 1, j, i + 1) +
                                    A(V3, k, j + 1, i + 1) + A(V3, k + 1, j + 1, i + 1)) / 4;
                const Real A3c1b = (A(V3, k, j, i)     + A(V3, k + 1, j, i) +
                                    A(V3, k, j + 1, i) + A(V3, k + 1, j + 1, i)) / 4;
                B_U(V2, k, j, i) = (A1c3f - A1c3b) / G.dx3v(k) - (A3c1f - A3c1b) / G.dx1v(i);

                // A2,1 derivative
                const Real A2c1f = (A(V2, k, j, i + 1)     + A(V2, k, j + 1, i + 1) + 
                                    A(V2, k + 1, j, i + 1) + A(V2, k + 1, j + 1, i + 1)) / 4;
                const Real A2c1b = (A(V2, k, j, i)     + A(V2, k, j + 1, i) +
                                    A(V2, k + 1, j, i) + A(V2, k + 1, j + 1, i)) / 4;
                // A1,2 derivative
                const Real A1c2f = (A(V1, k, j + 1, i)     + A(V1, k, j + 1, i + 1) +
                                    A(V1, k + 1, j + 1, i) + A(V1, k + 1, j + 1, i + 1)) / 4;
                const Real A1c2b = (A(V1, k, j, i)     + A(V1, k, j, i + 1) +
                                    A(V1, k + 1, j, i) + A(V1, k + 1, j, i + 1)) / 4;
                B_U(V3, k, j, i) = (A2c1f - A2c1b) / G.dx1v(i) - (A1c2f - A1c2b) / G.dx2v(j);
            }
        );
    } else {
        pmb->par_for("B_field_B_2D", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA_3D {
                // A3,2 derivative
                const Real A3c2f = (A(V3, k, j + 1, i) + A(V3, k, j + 1, i + 1)) / 2;
                const Real A3c2b = (A(V3, k, j, i)     + A(V3, k, j, i + 1)) / 2;
                B_U(V1, k, j, i) = (A3c2f - A3c2b) / G.dx2v(j);

                // A3,1 derivative
                const Real A3c1f = (A(V3, k, j, i + 1) + A(V3, k, j + 1, i + 1)) / 2;
                const Real A3c1b = (A(V3, k, j, i)     + A(V3, k, j + 1, i)) / 2;
                B_U(V2, k, j, i) = - (A3c1f - A3c1b) / G.dx1v(i);

                B_U(V3, k, j, i) = 0;
            }
        );
    }

    // Then make sure the primitive versions are updated, too
    B_FluxCT::UtoP(rc);

    return TaskStatus::complete;
}
