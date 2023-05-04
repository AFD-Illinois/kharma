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

TaskStatus CheckNaN(MeshData<Real> *md, int dir, IndexDomain domain)
{
    Flag("Checking ctop for NaNs");
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // TODO verbose option?

    // Pack variables
    auto& cmax = md->PackVariables(std::vector<std::string>{"Flux.cmax"});
    auto& cmin = md->PackVariables(std::vector<std::string>{"Flux.cmin"});

    // Get sizes
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    IndexRange block = IndexRange{0, cmax.GetDim(5) - 1};

    // TODO these two kernels can be one with some Kokkos magic
    int nzero = 0, nnan = 0;
    Kokkos::Sum<int> zero_reducer(nzero);
    Kokkos::Sum<int> nan_reducer(nnan);
    pmb0->par_reduce("ctop_zeros", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result) {
            if (m::max(cmax(b, dir-1, k, j, i), cmin(b, dir-1, k, j, i)) <= 0.) {
                ++local_result;
            }
        }
    , zero_reducer);
    pmb0->par_reduce("ctop_nans", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result) {
            if (m::isnan(m::max(cmax(b, dir-1, k, j, i), cmin(b, dir-1, k, j, i)))) {
                ++local_result;
                printf("ctop NaN at %d %d %d along dir %d\n", i, j, k, dir); // EDIT
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
        printf("Max signal speed ctop was 0 or NaN, direction %d (%d zero, %d NaN)", dir, nzero, nnan);
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
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result) {
            if (rho_c(b, 0, k, j, i) < 0.) ++local_result;
        }
    , sum_reducer);

    int nless_rho = 0, nless_u = 0;
    Kokkos::Sum<int> sum_reducer_rho(nless_rho);
    Kokkos::Sum<int> sum_reducer_u(nless_u);
    pmb0->par_reduce("count_negative_RHO", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result) {
            if (rho_p(b, 0, k, j, i) < 0.) ++local_result;
        }
    , sum_reducer_rho);
    pmb0->par_reduce("count_negative_UU", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result) {
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
        std::cout << "Number of negative conserved rho: " << nless << std::endl;
    }
    if (MPIRank0() && (nless_rho > 0 || nless_u > 0)) {
        std::cout << "Number of negative primitive rho, u: " << nless_rho << "," << nless_u << std::endl;
    }

    return TaskStatus::complete;
}
