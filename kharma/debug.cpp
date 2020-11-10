/* 
 *  File: debug.cpp
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

#include "debug.hpp"

#include "decs.hpp"

#include "mesh/mesh.hpp"

using namespace Kokkos;

double MaxDivB(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain)
{
    FLAG("Calculating divB");
    auto pmb = rc->GetBlockPointer();
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    auto& G = pmb->coords;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    double max_divb;
    Kokkos::Max<double> max_reducer(max_divb);
    pmb->par_reduce("divB", ks+1, ke, js+1, je, is+1, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            double local_divb = fabs(0.25*(
                              P(prims::B1, k, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B1, k, j-1, i) * G.gdet(Loci::center, j-1, i)
                            + P(prims::B1, k-1, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B1, k-1, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B1, k, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B1, k, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            - P(prims::B1, k-1, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B1, k-1, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            )/G.dx1v(i) +
                            0.25*(
                              P(prims::B2, k, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B2, k, j, i-1) * G.gdet(Loci::center, j, i-1)
                            + P(prims::B2, k-1, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B2, k-1, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B2, k, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B2, k, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            - P(prims::B2, k-1, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B2, k-1, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            )/G.dx2v(j) +
                            0.25*(
                              P(prims::B3, k, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B3, k, j-1, i) * G.gdet(Loci::center, j-1, i)
                            + P(prims::B3, k, j, i-1) * G.gdet(Loci::center, j, i-1)
                            + P(prims::B3, k, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            - P(prims::B3, k-1, j, i) * G.gdet(Loci::center, j, i)
                            - P(prims::B3, k-1, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B3, k-1, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B3, k-1, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            )/G.dx3v(k));
            if (local_divb > local_result) local_result = local_divb;
        }
    , max_reducer);

    return max_divb;
}

int Diagnostic(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain)
{
    FLAG("Summing bad cells");
    auto pmb = rc->GetBlockPointer();
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    auto& G = pmb->coords;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;

    int nless;
    Kokkos::Sum<int> sum_reducer(nless);
    pmb->par_reduce("count_negative", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE_INT {
            if (U(prims::rho, k, j, i) <= 0.) ++local_result;
        }
    , sum_reducer);

    cerr << "Number of negative conserved rho,u: " << nless << endl;

    return nless;
}

int CountPFlags(std::shared_ptr<MeshBlock> pmb, ParArrayNDIntHost pflag, IndexDomain domain, bool print)
{
    int n_tot = 0, n_neg_in = 0, n_max_iter = 0;
    int n_utsq = 0, n_gamma = 0, n_neg_u = 0, n_neg_rho = 0, n_neg_both = 0;

    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // TODO parallel reduces on device-side.  Need to separate each flag into a new reduce...
    for(int k=ks; k <= ke; ++k)
        for(int j=js; j <= je; ++j)
            for(int i=is; i <= ie; ++i)
    {
        int flag = pflag(k, j, i);
        if (flag > InversionStatus::success) ++n_tot; // Corner regions use negative flags.  They aren't "failures"
        if (flag == InversionStatus::neg_input) ++n_neg_in;
        if (flag == InversionStatus::max_iter) ++n_max_iter;
        if (flag == InversionStatus::bad_ut) ++n_utsq;
        if (flag == InversionStatus::bad_gamma) ++n_gamma;
        if (flag == InversionStatus::neg_rho) ++n_neg_rho;
        if (flag == InversionStatus::neg_u) ++n_neg_u;
        if (flag == InversionStatus::neg_rhou) ++n_neg_both;
    }

    n_tot = MPISumInt(n_tot);
    if (print) {
        n_neg_in = MPISumInt(n_neg_in);
        n_max_iter = MPISumInt(n_max_iter);
        n_utsq = MPISumInt(n_utsq);
        n_gamma = MPISumInt(n_gamma);
        n_neg_rho = MPISumInt(n_neg_rho);
        n_neg_u = MPISumInt(n_neg_u);
        n_neg_both = MPISumInt(n_neg_both);

        cerr << "PFLAGS: " << n_tot << " (" << ((double) n_tot)/((ke-ks+1)*(je-js+1)*(ie-is+1))*100 << "% of all cells)" << endl;
        if (n_neg_in > 0) cerr << "Negative input: " << n_neg_in << endl;
        if (n_max_iter > 0) cerr << "Hit max iter: " << n_max_iter << endl;
        if (n_utsq > 0) cerr << "Velocity invalid: " << n_utsq << endl;
        if (n_gamma > 0) cerr << "Gamma invalid: " << n_gamma << endl;
        if (n_neg_rho > 0) cerr << "Negative rho: " << n_neg_rho << endl;
        if (n_neg_u > 0) cerr << "Negative U: " << n_neg_u << endl;
        if (n_neg_both > 0) cerr << "Negative rho & U: " << n_neg_both << endl;
        cerr << endl;
    }
    return n_tot;
}

int CountFFlags(std::shared_ptr<MeshBlock> pmb, ParArrayNDIntHost fflag, IndexDomain domain, bool print)
{
    int n_tot = 0, n_geom_rho = 0, n_geom_u = 0, n_b_rho = 0, n_b_u = 0, n_temp = 0, n_gamma = 0, n_ktot = 0;

    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // TODO par_for
    for(int k=ks; k <= ke; ++k)
        for(int j=js; j <= je; ++j)
            for(int i=is; i <= ie; ++i)
    {
        int flag = fflag(k, j, i);
        if (flag != 0) n_tot++; // TODO allow a pflag to be set in low bits wihtout triggering this
        if (flag & HIT_FLOOR_GEOM_RHO) n_geom_rho++;
        if (flag & HIT_FLOOR_GEOM_U) n_geom_u++;
        if (flag & HIT_FLOOR_B_RHO) n_b_rho++;
        if (flag & HIT_FLOOR_B_U) n_b_u++;
        if (flag & HIT_FLOOR_TEMP) n_temp++;
        if (flag & HIT_FLOOR_GAMMA) n_gamma++;
        if (flag & HIT_FLOOR_KTOT) n_ktot++;
    }

    n_tot = MPISumInt(n_tot);
    if (print) {
        n_geom_rho = MPISumInt(n_geom_rho);
        n_geom_u = MPISumInt(n_geom_u);
        n_b_rho = MPISumInt(n_b_rho);
        n_b_u = MPISumInt(n_b_u);
        n_temp = MPISumInt(n_temp);
        n_gamma = MPISumInt(n_gamma);
        n_ktot = MPISumInt(n_ktot);

        cerr << "FLOORS: " << n_tot << " (" << ((double) n_tot)/((ke-ks+1)*(je-js+1)*(ie-is+1))*100 << "% of all cells)" << endl;
        if (n_geom_rho > 0) cerr << "GEOM_RHO: " << n_geom_rho << endl;
        if (n_geom_u > 0) cerr << "GEOM_U: " << n_geom_u << endl;
        if (n_b_rho > 0) cerr << "B_RHO: " << n_b_rho << endl;
        if (n_b_u > 0) cerr << "B_U: " << n_b_u << endl;
        if (n_temp > 0) cerr << "TEMPERATURE: " << n_temp << endl;
        if (n_gamma > 0) cerr << "GAMMA: " << n_gamma << endl;
        if (n_ktot > 0) cerr << "KTOT: " << n_ktot << endl;
        cerr << endl;
    }
    return n_tot;
}