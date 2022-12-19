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

#include "b_field_tools.hpp"

#include "grmhd_functions.hpp"

// TODO KHARMA now has good reduction tooling, use that instead of these

TaskStatus NormalizeBField(MeshBlockData<Real> *rc, Real norm)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    GridVector B_P = rc->Get("prims.B").data;
    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    pmb->par_for("B_field_normalize", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            VLOOP B_P(v, k, j, i) *= norm;
        }
    );

    return TaskStatus::complete;
}

Real GetLocalBetaMin(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const auto& G = pmb->coords;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    Real beta_min;
    Kokkos::Min<Real> min_reducer(beta_min);
    pmb->par_reduce("B_field_betamin", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
            double bsq_ij = dot(Dtmp.bcon, Dtmp.bcov);

            Real beta_ij = ((gam - 1) * u(k, j, i))/(0.5*(bsq_ij + TINY_NUMBER));

            if(beta_ij < local_result) local_result = beta_ij;
        }
    , min_reducer);
    return beta_min;
}

Real GetLocalBsqMax(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const auto& G = pmb->coords;

    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;

    Real bsq_max;
    Kokkos::Max<Real> bsq_max_reducer(bsq_max);
    pmb->par_reduce("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
            double bsq_ij = dot(Dtmp.bcon, Dtmp.bcov);
            if(bsq_ij > local_result) local_result = bsq_ij;
        }
    , bsq_max_reducer);
    return bsq_max;
}

Real GetLocalBsqMin(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const auto& G = pmb->coords;

    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;

    Real bsq_min;
    Kokkos::Min<Real> bsq_min_reducer(bsq_min);
    pmb->par_reduce("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, uvec, B_P, k, j, i, Loci::center, Dtmp);
            double bsq_ij = dot(Dtmp.bcon, Dtmp.bcov);
            if(bsq_ij < local_result) local_result = bsq_ij;
        }
    , bsq_min_reducer);
    return bsq_min;
}

Real GetLocalPMax(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const auto& G = pmb->coords;
    GridScalar u = rc->Get("prims.u").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    Real p_max;
    Kokkos::Max<Real> p_max_reducer(p_max);
    pmb->par_reduce("B_field_pmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            Real p_ij = (gam - 1) * u(k, j, i);
            if(p_ij > local_result) local_result = p_ij;
        }
    , p_max_reducer);
    return p_max;
}
