/* 
 *  File: reductions.cpp
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

#include "reductions.hpp"

#include <parthenon/parthenon.hpp>



Real Reductions::EHReduction(MeshData<Real> *md, UserHistoryOperation op, std::function<Real(REDUCE_FUNCTION_ARGS_EH)> fn, int zone)
{
    Flag("Performing accretion reduction");
    auto pmesh = md->GetMeshPointer();

    Real result = 0.;
    for (auto &pmb : pmesh->block_list) {
        // If we're on the inner edge
        if (pmb->boundary_flag[parthenon::BoundaryFace::inner_x1] == BoundaryFlag::user) {
            const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
            const Real gam = pars.Get<Real>("gamma");

            auto& rc = pmb->meshblock_data.Get();
            PackIndexMap prims_map, cons_map;
            const auto& P = rc->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
            const auto& U = rc->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
            const VarMap m_u(cons_map, true), m_p(prims_map, false);

            IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
            IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
            IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
            const auto& G = pmb->coords;

            Real block_result; 
            switch(op) {
            case UserHistoryOperation::sum: {
                Kokkos::Sum<Real> sum_reducer(block_result);
                pmb->par_reduce("accretion_sum", kb.s, kb.e, jb.s, jb.e, ib.s+zone, ib.s+zone,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result) {
                        local_result += fn(G, P, m_p, U, m_u, gam, k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j);
                    }
                , sum_reducer);
                result += block_result;
                break;
            }
            case UserHistoryOperation::max: {
                Kokkos::Max<Real> max_reducer(block_result);
                pmb->par_reduce("accretion_sum", kb.s, kb.e, jb.s, jb.e, ib.s+zone, ib.s+zone,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result) {
                        const Real val = fn(G, P, m_p, U, m_u, gam, k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j);
                        if (val > local_result) local_result = val;
                    }
                , max_reducer);
                if (block_result > result) result = block_result;
                break;
            }
            case UserHistoryOperation::min: {
                Kokkos::Min<Real> min_reducer(block_result);
                pmb->par_reduce("accretion_sum", kb.s, kb.e, jb.s, jb.e, ib.s+zone, ib.s+zone,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result) {
                        const Real val = fn(G, P, m_p, U, m_u, gam, k, j, i) * G.Dxc<3>(k) * G.Dxc<2>(j);
                        if (val < local_result) local_result = val;
                    }
                , min_reducer);
                if (block_result < result) result = block_result;
                break;
            }
            }
        }
    }

    Flag("Reduced");
    return result;
}

Real Reductions::DomainReduction(MeshData<Real> *md, UserHistoryOperation op, std::function<Real(REDUCE_FUNCTION_ARGS_MESH)> fn, Real arg)
{
    Flag("Performing domain reduction");
    auto pmesh = md->GetMeshPointer();

    // TODO TODO MESHDATA THIS
    Real result = 0.;
    const auto& pars = pmesh->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");

    PackIndexMap prims_map, cons_map;
    const auto& P = md->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const auto& U = md->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    IndexRange ib = pmb0->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb0->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb0->cellbounds.GetBoundsK(IndexDomain::interior);
    IndexRange block = IndexRange{0, U.GetDim(5) - 1};
    
    switch(op) {
    case UserHistoryOperation::sum: {
        Kokkos::Sum<Real> sum_reducer(result);
        pmb0->par_reduce("domain_sum", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, double &local_result) {
                const auto& G = U.GetCoords(b);
                local_result += fn(G, P(b), m_p, U(b), m_u, gam, k, j, i, arg) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i);
            }
        , sum_reducer);
        break;
    }
    case UserHistoryOperation::max: {
        Kokkos::Max<Real> max_reducer(result);
        pmb0->par_reduce("domain_max", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, double &local_result) {
                const auto& G = U.GetCoords(b);
                const Real val = fn(G, P(b), m_p, U(b), m_u, gam, k, j, i, arg) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i);
                if (val > local_result) local_result = val;
            }
        , max_reducer);
        break;
    }
    case UserHistoryOperation::min: {
        Kokkos::Min<Real> min_reducer(result);
        pmb0->par_reduce("domain_min", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, double &local_result) {
                const auto& G = U.GetCoords(b);
                const Real val = fn(G, P(b), m_p, U(b), m_u, gam, k, j, i, arg) * G.Dxc<3>(k) * G.Dxc<2>(j) * G.Dxc<1>(i);
                if (val < local_result) local_result = val;
            }
        , min_reducer);
        break;
    }
    }

    Flag("Reduced");
    return result;
}

/**
 * Counts occurrences of a particular flag value
 * 
 */
int Reductions::CountFlag(MeshData<Real> *md, std::string field_name, const int& flag_val, IndexDomain domain, bool is_bitflag)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Pack variables
    std::vector<std::string> flag_vec = {field_name};
    auto& flag = md->PackVariables(flag_vec);

    // Get sizes
    IndexRange ib = md->GetBoundsI(domain);
    IndexRange jb = md->GetBoundsJ(domain);
    IndexRange kb = md->GetBoundsK(domain);
    IndexRange block = IndexRange{0, flag.GetDim(5) - 1};

    int n_flag;
    Kokkos::Sum<int> flag_ct(n_flag);
    pmb0->par_reduce("count_flag", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result) {
            if ((is_bitflag && static_cast<int>(flag(b, 0, k, j, i)) & flag_val) ||
                (!is_bitflag && static_cast<int>(flag(b, 0, k, j, i)) == flag_val))
                ++local_result;
        }
    , flag_ct);
    return n_flag;
}

int Reductions::CountFlags(MeshData<Real> *md, std::string field_name, std::map<int, std::string> flag_values, IndexDomain domain, int verbose, bool is_bitflag)
{
    Flag("Counting inversion failures");
    int nflags = 0;
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Pack variables
    std::vector<std::string> flag_vec = {field_name};
    auto& flag = md->PackVariables(flag_vec);

    // Get sizes
    IndexRange ib = md->GetBoundsI(domain);
    IndexRange jb = md->GetBoundsJ(domain);
    IndexRange kb = md->GetBoundsK(domain);
    IndexRange block = IndexRange{0, flag.GetDim(5) - 1};

    // Count all nonzero (technically, >0) values
    // This works for pflags or fflags, so long as they're separate
    // We don't count negative pflags as they denote zones that shouldn't be fixed
    Kokkos::Sum<int> sum_reducer(nflags);
    pmb0->par_reduce("count_flags", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result) {
            if ((int) flag(b, 0, k, j, i) > 0) ++local_result;
        }
    , sum_reducer);

    // TODO TODO REPLACE ABOVE WITH SOMETHING LIKE:
    // array_sum::array_type<Real, 2> res;
    // parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, "RadiationResidual1",
    //                         DevExecSpace(), 0, mout->NumBlocks()-1,
    //                         0, nang1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    // KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i,
    //                 array_sum::array_type<Real, 2>& dsum) {
    //     dsum.my_array[0] += fabs(iiter(b,n,k,j,i) - iout(b,n,k,j,i));
    //     dsum.my_array[1] += iout(b,n,k,j,i);
    // }, array_sum::GlobalSum<Real, Kokkos::HostSpace, 2>(res));

    // Need the total on all ranks to evaluate the if statement below
    static AllReduce<int> n_tot;
    n_tot.val = nflags;
    n_tot.StartReduce(MPI_SUM);
    while (n_tot.CheckReduce() == TaskStatus::incomplete);
    nflags = n_tot.val;

    // If necessary, count each flag
    // This is slow, but it can be slow: it's not called for normal operation
    if (verbose > 0 && nflags > 0) {
        // Overlap reductions to save time
        // ...at the cost of considerable complexity...

        // TODO TODO eliminate static reducers, they crash the program after it finishes
        static Reduce<int> n_cells_r;
        n_cells_r.val = (block.e - block.s + 1) * (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
        n_cells_r.StartReduce(0, MPI_SUM);

        static std::vector<std::shared_ptr<Reduce<int>>> reducers;
        // Initialize reducers if they haven't been
        if (reducers.size() == 0) {
            for (auto& status : flag_values) {
                std::shared_ptr<Reduce<int>> reducer = std::make_shared<Reduce<int>>();
                reducers.push_back(reducer);
            }
        }
        // Count occurrences of each flag value, assign to a reducer in order
        int i = 0;
        for (auto& status : flag_values) {
            reducers[i]->val = CountFlag(md, field_name, (int) status.first, domain, is_bitflag);
            reducers[i]->StartReduce(0, MPI_SUM);
            ++i;
        }
        while (n_cells_r.CheckReduce() == TaskStatus::incomplete);
        const int n_cells = n_cells_r.val;
        // Check each reducer in order, add to a vector
        std::vector<int> n_status_present;
        for (std::shared_ptr<Reduce<int>> reducer : reducers) {
            while (reducer->CheckReduce() == TaskStatus::incomplete);
            n_status_present.push_back(reducer->val);
        }

        if (MPIRank0()) {
            std::cout << field_name << ": " << nflags << " (" << (int)(((double) nflags )/n_cells * 100) << "% of all cells)" << std::endl;
            if (verbose > 1) {
                // Print nonzero vector contents against flag names in order
                int i = 0;
                for (auto& status : flag_values) {
                    if (n_status_present[i] > 0) std::cout << status.second << ": " << n_status_present[i] << std::endl;
                    ++i;
                }
                std::cout << std::endl;
            }
        }

        // TODO Print zone locations of bad inversions
    }

    Flag("Counted");
    return nflags;
}
