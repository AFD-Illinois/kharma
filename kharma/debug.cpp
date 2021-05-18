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

#include "eos.hpp"
#include "mhd_functions.hpp"

using namespace Kokkos;

// TODO rewrite for KHARMA...
// void print_a_geom_tensor(GeomTensor2 g, const Loci loc, const int& i, const int& j)
// {
//     auto h = g.GetHostMirrorAndCopy();
//     int ii = i+NGHOST; int jj = j+NGHOST;
//     std::cout << h.label() << string_format(" element %d %d diagonal is [%f %f %f %f]", i, j,
//                             h(loc,jj,ii,0,0), h(loc,jj,ii,1,1), h(loc,jj,ii,2,2),
//                             h(loc,jj,ii,3,3)) << std::endl;
// }
// void print_a_geom_tensor3(GeomTensor3 g, const int& i, const int& j)
// {
//     auto h = g.GetHostMirrorAndCopy();
//     int ii = i+NGHOST; int jj = j+NGHOST;
//     std::cout << h.label() << string_format(" element %d %d lam=1 diagonal is [%f %f %f %f]", i, j,
//                             h(jj,ii,0,1,0), h(jj,ii,1,1,1), h(jj,ii,2,1,2),
//                             h(jj,ii,3,1,3)) << std::endl;
// }

void compare_P_U(MeshBlockData<Real> *rc, const int& k, const int& j, const int& i)
{
    auto pmb = rc->GetBlockPointer();
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVector B_P = rc->Get("c.c.bulk.B_prim").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    auto& G = pmb->coords;
    EOS* eos = pmb->packages.Get("GRMHD")->Param<EOS*>("eos");

    pmb->par_for("compare_P_U", k, k, j, j, i, i,
        KOKKOS_LAMBDA_3D {
            Real Utmp[NPRIM];
            GRMHD::p_to_u(G, P, B_P, eos, k, j, i, Utmp);
            //printf("U(P) = %g %g %g %g\nU(U) = %g %g %g %g\n",
            //        Utmp[prims::u], Utmp[prims::u1], Utmp[prims::u2], Utmp[prims::u3],
            //        U(prims::u, k, j, i), U(prims::u1, k, j, i), U(prims::u2, k, j, i), U(prims::u3, k, j, i));
        }
    );

}

TaskStatus CheckNaN(MeshBlockData<Real> *rc, int dir, IndexDomain domain)
{
    FLAG("Checking ctop for NaNs");
    auto pmb = rc->GetBlockPointer();
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    auto& G = pmb->coords;
    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;

    int nzero, nnan;
    Kokkos::Sum<int> zero_reducer(nzero);
    Kokkos::Sum<int> nan_reducer(nnan);

    pmb->par_reduce("ctop_zeros", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE_INT {
            if (ctop(dir, k, j, i) <= 0.) {
                //printf("Ctop zero at %d %d %d\n", k, j, i);
                //printf("Local P: %g %g %g %g %g\n", 
                //        P(prims::rho, k, j, i), P(prims::u, k, j, i), P(prims::u1, k, j, i), P(prims::u2, k, j, i),
                //        P(prims::u3, k, j, i));
                //printf("Local U: %g %g %g %g %g\n",
                //        U(prims::rho, k, j, i), U(prims::u, k, j, i), U(prims::u1, k, j, i), U(prims::u2, k, j, i),
                //        U(prims::u3, k, j, i));
                ++local_result;
            }
        }
    , zero_reducer);
    pmb->par_reduce("ctop_nans", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE_INT {
            if (isnan(ctop(dir, k, j, i))) {
                //printf("Ctop NaN at %d %d %d\n", k, j, i);
                ++local_result;
            }
        }
    , nan_reducer);

    if (nzero > 0 || nnan > 0) {
        throw std::runtime_error(string_format("Max signal speed ctop was 0 or NaN (%d zero, %d NaN)", nzero, nnan));
    }

    return TaskStatus::complete;
}

int CountPFlags(std::shared_ptr<MeshBlock> pmb, ParArrayNDHost pflag, IndexDomain domain, int verbose)
{
    int n_tot = 0, n_neg_in = 0, n_max_iter = 0;
    int n_utsq = 0, n_gamma = 0, n_neg_u = 0, n_neg_rho = 0, n_neg_both = 0;

    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // TODO parallel reduce on device-side.  Need a vector reducer for flags so it'll be a pain
    for(int k=ks; k <= ke; ++k)
        for(int j=js; j <= je; ++j)
            for(int i=is; i <= ie; ++i)
    {
        int flag = (int) pflag(k, j, i);
        if (flag > InversionStatus::success) ++n_tot; // Corner regions use negative flags.  They aren't "failures"
        if (flag == InversionStatus::neg_input) ++n_neg_in;
        if (flag == InversionStatus::max_iter) ++n_max_iter;
        if (flag == InversionStatus::bad_ut) ++n_utsq;
        if (flag == InversionStatus::bad_gamma) ++n_gamma;
        if (flag == InversionStatus::neg_rho) ++n_neg_rho;
        if (flag == InversionStatus::neg_u) ++n_neg_u;
        if (flag == InversionStatus::neg_rhou) ++n_neg_both;

        if (flag > InversionStatus::success && verbose > 2) {
            cout << "Bad inversion (" << flag << ") at i,j,k: " << i << " " << j << " " << k << endl;
            compare_P_U(pmb->meshblock_data.Get().get(), k, j, i);
        }
    }

    n_tot = MPISumInt(n_tot);
    if (verbose > 0 && n_tot > 0) {
        n_neg_in = MPISumInt(n_neg_in);
        n_max_iter = MPISumInt(n_max_iter);
        n_utsq = MPISumInt(n_utsq);
        n_gamma = MPISumInt(n_gamma);
        n_neg_rho = MPISumInt(n_neg_rho);
        n_neg_u = MPISumInt(n_neg_u);
        n_neg_both = MPISumInt(n_neg_both);

        cout << "PFLAGS: " << n_tot << " (" << ((double) n_tot)/((ke-ks+1)*(je-js+1)*(ie-is+1))*100 << "% of all cells)" << endl;
        if (verbose > 1) {
            if (n_neg_in > 0) cout << "Negative input: " << n_neg_in << endl;
            if (n_max_iter > 0) cout << "Hit max iter: " << n_max_iter << endl;
            if (n_utsq > 0) cout << "Velocity invalid: " << n_utsq << endl;
            if (n_gamma > 0) cout << "Gamma invalid: " << n_gamma << endl;
            if (n_neg_rho > 0) cout << "Negative rho: " << n_neg_rho << endl;
            if (n_neg_u > 0) cout << "Negative U: " << n_neg_u << endl;
            if (n_neg_both > 0) cout << "Negative rho & U: " << n_neg_both << endl;
            cout << endl;
        }
    }
    return n_tot;
}

int CountFFlags(std::shared_ptr<MeshBlock> pmb, ParArrayNDHost fflag, IndexDomain domain, int verbose)
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
        int flag = (int) fflag(k, j, i);
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
    if (verbose > 0 && n_tot > 0) {
        n_geom_rho = MPISumInt(n_geom_rho);
        n_geom_u = MPISumInt(n_geom_u);
        n_b_rho = MPISumInt(n_b_rho);
        n_b_u = MPISumInt(n_b_u);
        n_temp = MPISumInt(n_temp);
        n_gamma = MPISumInt(n_gamma);
        n_ktot = MPISumInt(n_ktot);

        cout << "FLOORS: " << n_tot << " (" << ((double) n_tot)/((ke-ks+1)*(je-js+1)*(ie-is+1))*100 << "% of all cells)" << endl;
        if (verbose > 1) {
            if (n_geom_rho > 0) cout << "GEOM_RHO: " << n_geom_rho << endl;
            if (n_geom_u > 0) cout << "GEOM_U: " << n_geom_u << endl;
            if (n_b_rho > 0) cout << "B_RHO: " << n_b_rho << endl;
            if (n_b_u > 0) cout << "B_U: " << n_b_u << endl;
            if (n_temp > 0) cout << "TEMPERATURE: " << n_temp << endl;
            if (n_gamma > 0) cout << "GAMMA: " << n_gamma << endl;
            if (n_ktot > 0) cout << "KTOT: " << n_ktot << endl;
            cout << endl;
        }
    }
    return n_tot;
}
