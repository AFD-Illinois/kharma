/* 
 *  File: seed_B_cd.cpp
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

#include "seed_B_cd.hpp"

#include "b_field_tools.hpp"

#include "b_flux_ct.hpp"
#include "grmhd_functions.hpp"

using namespace parthenon;

TaskStatus B_CD::SeedBField(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);

    const auto& G = pmb->coords;
    GridScalar rho = rc->Get("prims.rho").data;
    GridVector B_P = rc->Get("prims.B").data;
    GridVector B_U = rc->Get("cons.B").data;

    Real min_rho_q = pin->GetOrAddReal("b_field", "min_rho_q", 0.2);
    std::string b_field_type = pin->GetString("b_field", "type");

    // Translate to an enum so we can avoid string comp inside,
    // as well as for good errors, many->one maps, etc.
    BSeedType b_field_flag = ParseBSeedType(b_field_type);

    // Require and load what we need if necessary
    Real rin, b10, b20, b30;
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
        break;
    case BSeedType::ryan:
    case BSeedType::r3s3:
    case BSeedType::gaussian:
        rin = pin->GetReal("torus", "rin");
        break;
    default:
        break;
    }

    // Shortcut to field values for easy fields
    if (b_field_flag == BSeedType::constant) {
        pmb->par_for("B_field_B", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                // Set B1 directly
                B_P(0, k, j, i) = b10;
                B_P(1, k, j, i) = b20;
                B_P(2, k, j, i) = b30;
            }
        );
        return TaskStatus::complete;
    } else if (b_field_flag == BSeedType::monopole) {
        pmb->par_for("B_field_B", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                // Set B1 directly by normalizing
                B_P(0, k, j, i) = b10 / G.gdet(Loci::center, j, i);
                B_P(1, k, j, i) = 0.;
                B_P(2, k, j, i) = 0.;
            }
        );
        return TaskStatus::complete;
    }

    // Find the cell-centered magnetic vector potential.  In X3 symmetry only A_phi is non-zero, so we keep track of that.
    ParArrayND<Real> A3("A", n2, n1);
    // TODO figure out double vs Real here
    pmb->par_for("B_field_A", js+1, je, is+1, ie,
        KOKKOS_LAMBDA (const int& j, const int& i) {
            GReal Xembed[GR_DIM];
            G.coord_embed(0, j, i, Loci::center, Xembed);
            GReal r = Xembed[1], th = Xembed[2];

            // Use rho at cell centers
            Real rho_av = rho(ks, j, i);

            Real q;
            switch (b_field_flag)
            {
            case BSeedType::sane:
                q = rho_av - min_rho_q;
                break;
            case BSeedType::ryan:
                // BR's smoothed poloidal in-torus
                q = m::pow(sin(th), 3) * m::pow(r / rin, 3) * m::exp(-r / 400) * rho_av - min_rho_q;
                break;
            case BSeedType::r3s3:
                // Just the r^3 sin^3 th term, proposed EHT standard MAD
                // TODO split r3 here and r3s3
                q = m::pow(r / rin, 3) * rho_av - min_rho_q;
                break;
            case BSeedType::gaussian:
                // Pure vertical threaded field of gaussian strength with FWHM 2*rin (i.e. HM@rin)
                // centered at BH center
                // Block is to avoid compiler whinging about initialization
                {
                    Real x = (r / rin) * sin(th);
                    Real sigma = 2 / m::sqrt(2 * log(2));
                    Real u = x / m::abs(sigma);
                    q = (1 / (m::sqrt(2 * M_PI) * m::abs(sigma))) * m::exp(-u * u / 2);
                }
                break;
            default:
                // This shouldn't be reached.  Could squawk here?
                break;
            }

            A3(j, i) = m::max(q, 0.);
        }
    );

    // Calculate B-field
    pmb->par_for("B_field_B", ks, ke, js+1, je-1, is+1, ie-1,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            // Take the curl
            B_P(0, k, j, i) = (A3(j + 1, i) - A3(j-1, i)) / (2 * G.Dxc<2>(j) * G.gdet(Loci::center, j, i));
            B_P(1, k, j, i) = -(A3(j, i + 1) - A3(j, i-1)) / (2 * G.Dxc<1>(i) * G.gdet(Loci::center, j, i));
            B_P(2, k, j, i) = 0.;
        }
    );

    return TaskStatus::complete;
}
