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

#include "mhd_functions.hpp"

TaskStatus NormalizeBField(MeshBlockData<Real> *rc, Real norm)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVector B_P = rc->Get("c.c.bulk.B_prim").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    auto& G = pmb->coords;

    EOS* eos = pmb->packages.Get("GRMHD")->Param<EOS*>("eos");

    pmb->par_for("B_field_normalize", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            VLOOP B_P(v, k, j, i) *= norm;
            // We do need this for lockstep. If this were per-step I'd be mad.
            GRMHD::p_to_u(G, P, B_P, eos, k, j, i, U);
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
    auto& G = pmb->coords;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVector B_P = rc->Get("c.c.bulk.B_prim").data;

    EOS* eos = pmb->packages.Get("GRMHD")->Param<EOS*>("eos");

    Real beta_min;
    Kokkos::Min<Real> min_reducer(beta_min);
    pmb->par_reduce("B_field_betamin", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, P, B_P, k, j, i, Loci::center, Dtmp);
            double bsq_ij = dot(Dtmp.bcon, Dtmp.bcov);

            Real rho = P(prims::rho, k, j, i);
            Real u = P(prims::u, k, j, i);
            Real beta_ij = (eos->p(rho, u))/(0.5*(bsq_ij + TINY_NUMBER));

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
    auto& G = pmb->coords;

    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars B_P = rc->Get("c.c.bulk.B_prim").data;

    Real bsq_max;
    Kokkos::Max<Real> bsq_max_reducer(bsq_max);
    pmb->par_reduce("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, P, B_P, k, j, i, Loci::center, Dtmp);
            double bsq_ij = dot(Dtmp.bcon, Dtmp.bcov);
            if(bsq_ij > local_result) local_result = bsq_ij;
        }
    , bsq_max_reducer);
    return bsq_max;
}

Real GetLocalPMax(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto& G = pmb->coords;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    EOS* eos = pmb->packages.Get("GRMHD")->Param<EOS*>("eos");

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
