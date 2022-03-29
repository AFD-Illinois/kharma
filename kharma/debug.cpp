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

#include "floors.hpp"
#include "grmhd_functions.hpp"
#include "types.hpp"

using namespace Kokkos;

// TODO have nice ways to print vectors, areas, geometry, etc for debugging new modules
// TODO device-side total flag counts, for logging numbers even when not printing

void PrintCorner(MeshBlockData<Real> *rc, std::string name)
{
    auto var = rc->Get("prims.uvec").data.GetHostMirrorAndCopy();
    cerr << name;
    for (int j=0; j<8; j++) {
        cout << endl;
        for (int i=0; i<8; i++) {
            fprintf(stderr, "%.2g ", var(1, 0, j, i));
        }
    }
    cerr << endl;
}

TaskStatus CheckNaN(MeshData<Real> *md, int dir, IndexDomain domain)
{
    Flag("Checking ctop for NaNs");
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // TODO verbose option?

    // Pack variables
    auto& ctop = md->PackVariables(std::vector<std::string>{"ctop"});

    // Get sizes
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    IndexRange block = IndexRange{0, ctop.GetDim(5) - 1};

    // TODO these two kernels can be one with some Kokkos magic
    int nzero = 0, nnan = 0;
    Kokkos::Sum<int> zero_reducer(nzero);
    Kokkos::Sum<int> nan_reducer(nnan);
    pmb0->par_reduce("ctop_zeros", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE_INT {
            if (ctop(b, dir-1, k, j, i) <= 0.) {
                ++local_result;
            }
        }
    , zero_reducer);
    pmb0->par_reduce("ctop_nans", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE_INT {
            if (isnan(ctop(b, dir-1, k, j, i))) {
                ++local_result;
            }
        }
    , nan_reducer);

    nzero = MPISum(nzero);
    nnan = MPISum(nnan);

    if (MPIRank0() && (nzero > 0 || nnan > 0)) {
        // TODO string formatting in C++ that doesn't suck
        fprintf(stderr, "Max signal speed ctop was 0 or NaN, direction %d (%d zero, %d NaN)", dir, nzero, nnan);
        throw std::runtime_error("Bad ctop!");
    }

    // TODO reimplement printing *where* these values were hit?
    // May not even be that useful, as the cause is usually much earlier

    Flag("Checked");
    return TaskStatus::complete;
}

TaskStatus CheckNegative(MeshData<Real> *md, IndexDomain domain)
{
    Flag("Counting negative values");
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Pack variables
    auto rho_p = md->PackVariables(std::vector<std::string>{"prims.rho"});
    auto u_p = md->PackVariables(std::vector<std::string>{"prims.u"});
    auto rho_c = md->PackVariables(std::vector<std::string>{"cons.rho"});
    // Get sizes
    IndexRange ib = md->GetBoundsI(domain);
    IndexRange jb = md->GetBoundsJ(domain);
    IndexRange kb = md->GetBoundsK(domain);
    IndexRange block = IndexRange{0, rho_p.GetDim(5)-1};

    // Check for negative values in the conserved vars
    int nless = 0;
    Kokkos::Sum<int> sum_reducer(nless);
    pmb0->par_reduce("count_negative_U", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE_INT {
            if (rho_c(b, 0, k, j, i) < 0.) ++local_result;
        }
    , sum_reducer);
    nless = MPISum(nless);
    if (MPIRank0() && nless > 0) {
        cout << "Number of negative conserved rho: " << nless << endl;
    }

    int nless_rho = 0, nless_u = 0;
    Kokkos::Sum<int> sum_reducer_rho(nless_rho);
    Kokkos::Sum<int> sum_reducer_u(nless_u);
    pmb0->par_reduce("count_negative_RHO", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE_INT {
            if (rho_p(b, 0, k, j, i) < 0.) ++local_result;
        }
    , sum_reducer_rho);
    pmb0->par_reduce("count_negative_UU", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE_INT {
            if (u_p(b, 0, k, j, i) < 0.) ++local_result;
        }
    , sum_reducer_u);
    nless_rho = MPISum(nless_rho);
    nless_u = MPISum(nless_u);

    if (MPIRank0() && (nless_rho > 0 || nless_u > 0)) {
        cout << "Number of negative primitive rho, u: " << nless_rho << "," << nless_u << endl;
    }

    return TaskStatus::complete;
}

int CountPFlags(MeshData<Real> *md, IndexDomain domain, int verbose)
{
    int n_cells = 0, n_tot = 0, n_neg_in = 0, n_max_iter = 0;
    int n_utsq = 0, n_gamma = 0, n_neg_u = 0, n_neg_rho = 0, n_neg_both = 0;
    auto pmesh = md->GetMeshPointer();

    int block = 0;
    for (auto &pmb : pmesh->block_list) {
        int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
        int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
        int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
        auto& rc = pmb->meshblock_data.Get();
        auto pflag = rc->Get("pflag").data.GetHostMirrorAndCopy();

    // OpenMP causes problems when used separately from Kokkos
    // TODO make this a kokkos reduction to a View
//#pragma omp parallel for simd collapse(3) reduction(+:n_cells,n_tot,n_neg_in,n_max_iter,n_utsq,n_gamma,n_neg_u,n_neg_rho,n_neg_both)
        for(int k=ks; k <= ke; ++k)
            for(int j=js; j <= je; ++j)
                for(int i=is; i <= ie; ++i)
        {
            ++n_cells;
            int flag = (int) pflag(k, j, i);
            if (flag > InversionStatus::success) ++n_tot; // Corner regions use negative flags.  They aren't "failures"
            if (flag == InversionStatus::neg_input) ++n_neg_in;
            if (flag == InversionStatus::max_iter) ++n_max_iter;
            if (flag == InversionStatus::bad_ut) ++n_utsq;
            if (flag == InversionStatus::bad_gamma) ++n_gamma;
            if (flag == InversionStatus::neg_rho) ++n_neg_rho;
            if (flag == InversionStatus::neg_u) ++n_neg_u;
            if (flag == InversionStatus::neg_rhou) ++n_neg_both;

            // TODO MPI Rank
            if (flag > InversionStatus::success && verbose >= 3) {
                printf("Bad inversion (%d) at block %d zone %d %d %d\n", flag, block, i, j, k);
                //compare_P_U(pmb->meshblock_data.Get().get(), k, j, i);
            }
        }

        ++block;
    }

    n_tot = MPISum(n_tot);
    if (verbose > 0 && n_tot > 0) {
        n_cells = MPISum(n_cells);
        n_neg_in = MPISum(n_neg_in);
        n_max_iter = MPISum(n_max_iter);
        n_utsq = MPISum(n_utsq);
        n_gamma = MPISum(n_gamma);
        n_neg_rho = MPISum(n_neg_rho);
        n_neg_u = MPISum(n_neg_u);
        n_neg_both = MPISum(n_neg_both);

        if (MPIRank0()) {
            cout << "PFLAGS: " << n_tot << " (" << ((double) n_tot )/n_cells * 100 << "% of all cells)" << endl;
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
    }
    return n_tot;
}

int CountFFlags(MeshData<Real> *md, IndexDomain domain, int verbose)
{
    int n_cells = 0, n_tot = 0, n_geom_rho = 0, n_geom_u = 0, n_b_rho = 0, n_b_u = 0, n_temp = 0, n_gamma = 0, n_ktot = 0;
    auto pmesh = md->GetMeshPointer();

    for (auto &pmb : pmesh->block_list) {
        int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
        int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
        int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
        auto& rc = pmb->meshblock_data.Get();
        auto fflag = rc->Get("fflag").data.GetHostMirrorAndCopy();

    // See above re: Openmp. TODO Kokkosify
//#pragma omp parallel for simd collapse(3) reduction(+:n_cells,n_tot,n_geom_rho,n_geom_u,n_b_rho,n_b_u,n_temp,n_gamma,n_ktot)
        for(int k=ks; k <= ke; ++k)
            for(int j=js; j <= je; ++j)
                for(int i=is; i <= ie; ++i)
        {
            ++n_cells;
            int flag = (int) fflag(k, j, i);
            if (flag != 0) ++n_tot;
            if (flag & HIT_FLOOR_GEOM_RHO) ++n_geom_rho;
            if (flag & HIT_FLOOR_GEOM_U) ++n_geom_u;
            if (flag & HIT_FLOOR_B_RHO) ++n_b_rho;
            if (flag & HIT_FLOOR_B_U) ++n_b_u;
            if (flag & HIT_FLOOR_TEMP) ++n_temp;
            if (flag & HIT_FLOOR_GAMMA) ++n_gamma;
            if (flag & HIT_FLOOR_KTOT) ++n_ktot;
        }
    }

    n_tot = MPISum(n_tot);
    if (verbose > 0 && n_tot > 0) {
        n_cells = MPISum(n_cells);
        n_geom_rho = MPISum(n_geom_rho);
        n_geom_u = MPISum(n_geom_u);
        n_b_rho = MPISum(n_b_rho);
        n_b_u = MPISum(n_b_u);
        n_temp = MPISum(n_temp);
        n_gamma = MPISum(n_gamma);
        n_ktot = MPISum(n_ktot);

        if (MPIRank0()) {
            cout << "FLOORS: " << n_tot << " (" << (int)(((double) n_tot)/ n_cells * 100) << "% of all cells)" << endl;
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
    }
    return n_tot;
}
