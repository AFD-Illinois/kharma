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
#include "mpi.hpp"
#include "types.hpp"

using namespace Kokkos;

// TODO have nice ways to print vectors, areas, geometry, etc for debugging new modules

/**
 * Counts occurrences of a particular floor bitflag
 */
int CountFFlag(MeshData<Real> *md, const int& flag_val, IndexDomain domain)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Pack variables
    auto& fflag = md->PackVariables(std::vector<std::string>{"fflag"});

    // Get sizes
    IndexRange ib = md->GetBoundsI(domain);
    IndexRange jb = md->GetBoundsJ(domain);
    IndexRange kb = md->GetBoundsK(domain);
    IndexRange block = IndexRange{0, fflag.GetDim(5) - 1};

    int n_flag;
    Kokkos::Sum<int> flag_ct(n_flag);
    pmb0->par_reduce("count_fflag", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE_INT {
            if (((int) fflag(b, 0, k, j, i)) & flag_val) ++local_result;
        }
    , flag_ct);
    return n_flag;
}

/**
 * Counts occurrences of a particular inversion failure mode
 */
int CountPFlag(MeshData<Real> *md, const int& flag_val, IndexDomain domain)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Pack variables
    auto& pflag = md->PackVariables(std::vector<std::string>{"pflag"});

    // Get sizes
    IndexRange ib = md->GetBoundsI(domain);
    IndexRange jb = md->GetBoundsJ(domain);
    IndexRange kb = md->GetBoundsK(domain);
    IndexRange block = IndexRange{0, pflag.GetDim(5) - 1};

    int n_flag;
    Kokkos::Sum<int> flag_ct(n_flag);
    pmb0->par_reduce("count_pflag", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE_INT {
            if (((int) pflag(b, 0, k, j, i)) == flag_val) ++local_result;
        }
    , flag_ct);
    return n_flag;
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

    // Reductions in parallel
    // Only need to reduce to head node, saves time
    static Reduce<int> nzero_tot, nnan_tot;
    nzero_tot.val = nzero;
    nnan_tot.val = nnan;
    nzero_tot.StartReduce(0, MPI_SUM);
    nnan_tot.StartReduce(0, MPI_SUM);
    while (nzero_tot.CheckReduce() == TaskStatus::incomplete);
    while (nnan_tot.CheckReduce() == TaskStatus::incomplete);
    nzero = nzero_tot.val;
    nnan = nnan_tot.val;

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

    // Reductions in parallel
    static Reduce<int> nless_tot, nless_rho_tot, nless_u_tot;
    nless_tot.val = nless;
    nless_rho_tot.val = nless_rho;
    nless_u_tot.val = nless_u;
    nless_tot.StartReduce(0, MPI_SUM);
    nless_rho_tot.StartReduce(0, MPI_SUM);
    nless_u_tot.StartReduce(0, MPI_SUM);
    while (nless_tot.CheckReduce() == TaskStatus::incomplete);
    while (nless_rho_tot.CheckReduce() == TaskStatus::incomplete);
    while (nless_u_tot.CheckReduce() == TaskStatus::incomplete);
    nless = nless_tot.val;
    nless_rho = nless_rho_tot.val;
    nless_u = nless_u_tot.val;

    if (MPIRank0() && nless > 0) {
        cout << "Number of negative conserved rho: " << nless << endl;
    }
    if (MPIRank0() && (nless_rho > 0 || nless_u > 0)) {
        cout << "Number of negative primitive rho, u: " << nless_rho << "," << nless_u << endl;
    }

    return TaskStatus::complete;
}

int CountPFlags(MeshData<Real> *md, IndexDomain domain, int verbose)
{
    Flag("Counting inversion failures");
    int nflags = 0;
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Pack variables
    auto& pflag = md->PackVariables(std::vector<std::string>{"pflag"});

    // Get sizes
    IndexRange ib = md->GetBoundsI(domain);
    IndexRange jb = md->GetBoundsJ(domain);
    IndexRange kb = md->GetBoundsK(domain);
    IndexRange block = IndexRange{0, pflag.GetDim(5) - 1};

    // Count all nonzero values
    Kokkos::Sum<int> sum_reducer(nflags);
    pmb0->par_reduce("count_all_pflags", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE_INT {
            if ((int) pflag(b, 0, k, j, i) > InversionStatus::success) ++local_result;
        }
    , sum_reducer);

    // Need the total on all ranks to evaluate the if statement below
    static AllReduce<int> n_tot;
    n_tot.val = nflags;
    n_tot.StartReduce(MPI_SUM);
    while (n_tot.CheckReduce() == TaskStatus::incomplete);
    nflags = n_tot.val;

    // If necessary, count each flag
    // This is slow, but it can be slow: it's not called for normal operation
    if (verbose > 0 && nflags > 0) {
        std::vector<InversionStatus> all_status_vals = {InversionStatus::neg_input,
                                                        InversionStatus::max_iter,
                                                        InversionStatus::bad_ut,
                                                        InversionStatus::bad_gamma,
                                                        InversionStatus::neg_rho,
                                                        InversionStatus::neg_u,
                                                        InversionStatus::neg_rhou};
        std::vector<std::string> all_status_names = {"Negative input",
                                                     "Hit max iter",
                                                     "Velocity invalid",
                                                     "Gamma invalid",
                                                     "Negative rho",
                                                     "Negative U",
                                                     "Negative rho & U"};

        // Overlap reductions to save time
        static Reduce<int> n_cells_r;
        n_cells_r.val = (block.e - block.s + 1) * (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
        n_cells_r.StartReduce(0, MPI_SUM);
        static std::vector<Reduce<int>> reducers;
        static bool firstc = true;
        if (firstc) {
            for (InversionStatus status : all_status_vals) {
                reducers.push_back(Reduce<int>());
            }
            firstc = false;
        }
        int i = 0;
        for (InversionStatus status : all_status_vals) {
            reducers[i].val = CountPFlag(md, status, domain);
            reducers[i].StartReduce(0, MPI_SUM);
            i++;
        }
        while (n_cells_r.CheckReduce() == TaskStatus::incomplete);
        const int n_cells = n_cells_r.val;
        std::vector<int> n_status_present;
        for (Reduce<int> reducer : reducers) {
            while (reducer.CheckReduce() == TaskStatus::incomplete);
            n_status_present.push_back(reducer.val);
        }

        if (MPIRank0()) {
            std::cout << "PFLAGS: " << nflags << " (" << (int)(((double) nflags )/n_cells * 100) << "% of all cells)" << std::endl;
            if (verbose > 1) {
                for (int i=0; i < all_status_vals.size(); ++i) {
                    if (n_status_present[i] > 0) std::cout << all_status_names[i] << ": " << n_status_present[i] << std::endl;
                }
                std::cout << std::endl;
            }
        }

        // TODO Print zone locations of bad inversions
    }

    Flag("Counted");
    return nflags;
}

int CountFFlags(MeshData<Real> *md, IndexDomain domain, int verbose)
{
    Flag("Couting floor hits");
    int nflags = 0;
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Pack variables
    auto& fflag = md->PackVariables(std::vector<std::string>{"fflag"});

    // Get sizes
    IndexRange ib = md->GetBoundsI(domain);
    IndexRange jb = md->GetBoundsJ(domain);
    IndexRange kb = md->GetBoundsK(domain);
    IndexRange block = IndexRange{0, fflag.GetDim(5) - 1};

    // Count all nonzero values
    Kokkos::Sum<int> sum_reducer(nflags);
    pmb0->par_reduce("count_all_fflags", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE_INT {
            if ((int) fflag(b, 0, k, j, i) != 0) ++local_result;
        }
    , sum_reducer);

    // Need this on all nodes to evaluate the following if statement
    static AllReduce<int> n_tot;
    n_tot.val = nflags;
    n_tot.StartReduce(MPI_SUM);
    while (n_tot.CheckReduce() == TaskStatus::incomplete);
    nflags = n_tot.val;

    if (verbose > 0 && nflags > 0) {
        std::vector<int> all_flag_vals = {HIT_FLOOR_GEOM_RHO,
                                        HIT_FLOOR_GEOM_U,
                                        HIT_FLOOR_B_RHO,
                                        HIT_FLOOR_B_U,
                                        HIT_FLOOR_TEMP,
                                        HIT_FLOOR_GAMMA,
                                        HIT_FLOOR_KTOT};
        std::vector<std::string> all_flag_names = {"GEOM_RHO",
                                                   "GEOM_U",
                                                   "B_RHO",
                                                   "B_U",
                                                   "TEMPERATURE",
                                                   "GAMMA",
                                                   "KTOT"};

        // Overlap reductions to save time
        static Reduce<int> n_cells_r;
        n_cells_r.val = (block.e - block.s + 1) * (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
        n_cells_r.StartReduce(0, MPI_SUM);
        static std::vector<Reduce<int>> reducers;
        static bool firstc = true;
        if (firstc) {
            for (int flag : all_flag_vals) {
                reducers.push_back(Reduce<int>());
            }
            firstc = false;
        }
        int i = 0;
        for (int flag : all_flag_vals) {
            reducers[i].val = CountFFlag(md, flag, domain);
            reducers[i].StartReduce(0, MPI_SUM);
            i++;
        }
        while (n_cells_r.CheckReduce() == TaskStatus::incomplete);
        const int n_cells = n_cells_r.val;
        std::vector<int> n_flag_present;
        for (Reduce<int> reducer : reducers) {
            while (reducer.CheckReduce() == TaskStatus::incomplete);
            n_flag_present.push_back(reducer.val);
        }

        if (MPIRank0()) {
            std::cout << "FLOORS: " << nflags << " (" << (int)(((double) nflags) / n_cells * 100) << "% of all cells)" << std::endl;
            if (verbose > 1) {
                for (int i=0; i < all_flag_vals.size(); ++i) {
                    if (n_flag_present[i] > 0) std::cout << all_flag_names[i] << ": " << n_flag_present[i] << std::endl;
                }
                cout << std::endl;
            }
        }
    }

    Flag("Counted");
    return nflags;
}
