/**
 * In a perfect world, I would hand my loop specs to Kokkos, and it would make them fast.
 * Instead, we have this.
 */
#pragma once

#include "decs.hpp"

#include <string>

#define MDRANGE 1
#define RANGE   0
#define TPX     0
#define GRABBAG 0
#define SIMDFOR 0

// TODO remember to test right&left storage, AND right&left Ranges

template<typename... Properties, typename LambdaType>
void abstract_for(std::string name, Kokkos::MDRangePolicy<Properties...> range, LambdaType fn)
{
#if MDRANGE
    // MDRange loops
    Kokkos::parallel_for(name, range, fn);
#else
    using traits = Kokkos::Impl::PolicyTraits<Properties...>;
    using execution_space = typename traits::execution_space;
    using iteration_pattern = typename traits::iteration_pattern;
    int rank = static_cast<int>(iteration_pattern::rank);
#if RANGE
    // Kokkos 1D Range
    if (rank == 3) {
        int il = range.m_lower[0]; int ih = range.m_upper[0];
        int jl = range.m_lower[1]; int jh = range.m_upper[1];
        int kl = range.m_lower[2]; int kh = range.m_upper[2];
        int ni = ih - il;
        int nj = jh - jl;
        int nk = kh - kl;
        // TODO left and right here...
        int total_size = ni * nj * nk;
        int stride_i = nj * nk;
        int stride_j = nk;
        int stride_k = 1;

        Kokkos::parallel_for(name, Kokkos::RangePolicy<execution_space>(0,total_size),
            KOKKOS_LAMBDA (const int& idx)
            {
                int i = il + idx / stride_i;
                int j = jl + (idx - i*stride_i) / stride_j;
                int k = kl + (idx - i*stride_i - j*stride_j) / stride_k;
                fn(i,j,k);
            }
        );
    } else if (rank == 4) {
        // TODO how to convince compiler rank 4 == 4-arg func?
        // int il = range.m_lower[0]; int ih = range.m_upper[0];
        // int jl = range.m_lower[1]; int jh = range.m_upper[1];
        // int kl = range.m_lower[2]; int kh = range.m_upper[2];
        // int pl = range.m_lower[3]; int ph = range.m_upper[3];
        // int ni = ih - il;
        // int nj = jh - jl;
        // int nk = kh - kl;
        // int np = ph - pl;
        // // TODO left and right here...
        // int total_size = ni*nj*nk*np;
        // int stride_i = nj * nk * np;
        // int stride_j = nk * np;
        // int stride_k = np;
        // int stride_p = 1;

        // Kokkos::parallel_for(name, Kokkos::RangePolicy<execution_space>(0,total_size),
        //     KOKKOS_LAMBDA (const int& idx)
        //     {
        //         int i = il + idx / stride_i;
        //         int j = jl + (idx - i*stride_i) / stride_j;
        //         int k = kl + (idx - i*stride_i - j*stride_j) / stride_k;
        //         int p = pl + idx - i*stride_i - j*stride_j - k*stride_k;
        //         fn(i,j,k,p);
        //     }
        // );
    }
#elif TPX
  // TeamPolicy loops
    const int NN = NU - NL + 1;
    const int NK = KU - KL + 1;
    const int NJ = JU - JL + 1;
    const int NKNJ = NK * NJ;
    const int NNNKNJ = NN * NK * NJ;
    Kokkos::parallel_for(NAME,
      team_policy (NNNKNJ, Kokkos::AUTO,KOKKOS_VECTOR_LENGTH),
      KOKKOS_LAMBDA (member_type team_member) {
        int n = team_member.league_rank() / NKNJ;
        int k = (team_member.league_rank() - n*NKNJ) / NJ;
        int j = team_member.league_rank() - n*NKNJ - k*NJ + JL;
        n += NL;
        k += KL;
        Kokkos::parallel_for(
          TPINNERLOOP<>(team_member,IL,IU+1),
          [&] (const int i) {
            function(n,k,j,i);
          });
      });
#elif GRABBAG
    // TeamPolicy with nested TeamThreadRange and ThreadVectorRange
    const int NN = NU - NL + 1;
    const int NK = KU - KL + 1;
    const int NNNK = NN * NK;
    Kokkos::parallel_for(NAME,
      team_policy (NNNK, Kokkos::AUTO,KOKKOS_VECTOR_LENGTH),
      KOKKOS_LAMBDA (member_type team_member) {
        int n = team_member.league_rank() / NK + NL;
        int k = team_member.league_rank() % NK + KL;
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange<>(team_member,JL,JU+1),
          [&] (const int j) {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange<>(team_member,IL,IU+1),
              [&] (const int i) {
                function(n,k,j,i);
              });
          });
      });
#elif SIMDFOR
    // SIMD FOR loops
    Kokkos::Profiling::pushRegion(NAME);
    for (auto n = NL; n <= NU; n++)
      for (auto k = KL; k <= KU; k++)
        for (auto j = JL; j <= JU; j++)
          #pragma omp simd
          for (auto i = IL; i <= IU; i++)
            function(n,k,j,i);
    Kokkos::Profiling::popRegion();
#endif
#endif
}