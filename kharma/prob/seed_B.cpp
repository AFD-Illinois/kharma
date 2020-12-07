/* 
 *  File: seed_B.cpp
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

#include "seed_B.hpp"

#include "phys.hpp"

// Internal representation of the field initialization preference for quick switch
// Avoids string comparsion in kernels
enum BSeedType{sane, ryan, r3s3, gaussian};

TaskStatus SeedBField(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);

    auto& G = pmb->coords;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    Real rin;
    Real min_rho_q = pin->GetOrAddReal("b_field", "min_rho_q", 0.2);
    std::string b_field_type = pin->GetString("b_field", "type");

    // Translate to an enum so we can avoid string comp inside,
    // as well as for good errors, many->one maps, etc.
    BSeedType b_field_flag = BSeedType::sane;
    if (b_field_type == "none") {
        return TaskStatus::complete;
    } else if (b_field_type == "sane") {
        b_field_flag = BSeedType::sane;
    } else if (b_field_type == "mad" || b_field_type == "ryan") {
        b_field_flag = BSeedType::ryan;
    } else if (b_field_type == "r3s3") {
        b_field_flag = BSeedType::r3s3;
    } else if (b_field_type == "gaussian") {
        b_field_flag = BSeedType::gaussian;
    } else {
        throw std::invalid_argument("Magnetic field seed type not supported: " + b_field_type);
    }

    // Require and load rin if necessary
    switch (b_field_flag)
    {
    case BSeedType::sane:
        break;
    case BSeedType::ryan:
    case BSeedType::r3s3:
    case BSeedType::gaussian:
        rin = pin->GetReal("torus", "rin");
        break;
    }

    // Find the magnetic vector potential.  In X3 symmetry only A_phi is non-zero, so we keep track of that.
    ParArrayND<Real> A("A", n2, n1);
    // TODO figure out double vs Real here
    pmb->par_for("B_field_A", js+1, je, is+1, ie,
        KOKKOS_LAMBDA_2D {
            GReal Xembed[GR_DIM];
            G.coord_embed(0, j, i, Loci::center, Xembed);
            GReal r = Xembed[1], th = Xembed[2];

            // Find rho (later u?) at corners by averaging from adjacent centers
            Real rho_av = 0.25 * (P(prims::rho, ks, j, i)     + P(prims::rho, ks, j, i - 1) +
                                  P(prims::rho, ks, j - 1, i) + P(prims::rho, ks, j - 1, i - 1));

            Real q;
            switch (b_field_flag)
            {
            case BSeedType::sane:
                q = rho_av - min_rho_q;
                break;
            case BSeedType::ryan:
                // BR's smoothed poloidal in-torus
                q = pow(sin(th), 3) * pow(r / rin, 3) * exp(-r / 400) * rho_av - min_rho_q;
                break;
            case BSeedType::r3s3:
                // Just the r^3 sin^3 th term, proposed EHT standard MAD
                // TODO split r3 here and r3s3
                q = pow(r / rin, 3) * rho_av - min_rho_q;
                break;
            case BSeedType::gaussian:
                // Pure vertical threaded field of gaussian strength with FWHM 2*rin (i.e. HM@rin)
                // centered at BH center
                Real x = (r / rin) * sin(th);
                Real sigma = 2 / sqrt(2 * log(2));
                Real u = x / fabs(sigma);
                q = (1 / (sqrt(2 * M_PI) * fabs(sigma))) * exp(-u * u / 2);
                break;
            }

            A(j, i) = max(q, 0.);
        }
    );

    // Calculate B-field
    pmb->par_for("B_field_B", ks, ke, js, je-1, is, ie-1,
        KOKKOS_LAMBDA_3D {
            // Take a flux-ct step from the corner potentials
            P(prims::B1, k, j, i) = -(A(j, i) - A(j + 1, i) + A(j, i + 1) - A(j + 1, i + 1)) /
                                (2. * G.dx2v(j) * G.gdet(Loci::center, j, i));
            P(prims::B2, k, j, i) =  (A(j, i) + A(j + 1, i) - A(j, i + 1) - A(j + 1, i + 1)) /
                                (2. * G.dx1v(i) * G.gdet(Loci::center, j, i));
            P(prims::B3, k, j, i) = 0.;

            // We don't need to update U here, since we're always going to normalize straightaway
        }
    );

    return TaskStatus::complete;
}

TaskStatus NormalizeBField(std::shared_ptr<MeshBlockData<Real>>& rc, Real norm)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    auto& G = pmb->coords;

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    pmb->par_for("B_field_normalize", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            P(prims::B1, k, j, i) *= norm;
            P(prims::B2, k, j, i) *= norm;
            P(prims::B3, k, j, i) *= norm;

            p_to_u(G, P, eos, k, j, i, U);
        }
    );

    return TaskStatus::complete;
}

Real GetLocalBetaMin(std::shared_ptr<MeshBlockData<Real>>& rc)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto& G = pmb->coords;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    Real beta_min;
    Kokkos::Min<Real> min_reducer(beta_min);
    pmb->par_reduce("B_field_betamin", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            FourVectors Dtmp;
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            double bsq_ij = dot(Dtmp.bcon, Dtmp.bcov);

            Real rho = P(prims::rho, k, j, i);
            Real u = P(prims::u, k, j, i);
            Real beta_ij = (eos->p(rho, u))/(0.5*(bsq_ij + TINY_NUMBER));

            if(beta_ij < local_result) local_result = beta_ij;
        }
    , min_reducer);
    return beta_min;
}
Real GetLocalBsqMax(std::shared_ptr<MeshBlockData<Real>>& rc)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto& G = pmb->coords;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    Real bsq_max;
    Kokkos::Max<Real> bsq_max_reducer(bsq_max);
    pmb->par_reduce("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            FourVectors Dtmp;
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            double bsq_ij = dot(Dtmp.bcon, Dtmp.bcov);
            if(bsq_ij > local_result) local_result = bsq_ij;
        }
    , bsq_max_reducer);
    return bsq_max;
}
Real GetLocalPMax(std::shared_ptr<MeshBlockData<Real>>& rc)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto& G = pmb->coords;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    Real p_max;
    Kokkos::Max<Real> p_max_reducer(p_max);
    pmb->par_reduce("B_field_pmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            Real rho = P(prims::rho, k, j, i);
            Real u = P(prims::u, k, j, i);
            Real p_ij = eos->p(rho, u);
            if(p_ij > local_result) local_result = p_ij;
        }
    , p_max_reducer);
    return p_max;
}