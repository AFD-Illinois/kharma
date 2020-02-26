#ifndef ATHENA_HPP_
#define ATHENA_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

// C headers
#include <math.h>
#include <stdint.h>  // int64_t

// Kokkos headers
#include <Kokkos_Core.hpp>

// C++ headers
#include <string>

#ifdef KOKKOS_ENABLE_CUDA_UVM
typedef Kokkos::CudaUVMSpace     DevSpace;
typedef Kokkos::CudaUVMSpace     HostSpace;
#else
typedef Kokkos::DefaultExecutionSpace     DevSpace;
typedef Kokkos::HostSpace                 HostSpace;

#endif

#ifdef INNER_TTR_LOOP
#define TPINNERLOOP Kokkos::TeamThreadRange
#elif defined INNER_TVR_LOOP
#define TPINNERLOOP Kokkos::ThreadVectorRange
#else
#define TPINNERLOOP Kokkos::TeamThreadRange
#endif

typedef Kokkos::TeamPolicy<>               team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;

enum class LoopPattern { SIMDFOR, RANGE, MDRANGE, TPX, TPTTRTVR, UNDEFINED };
#ifdef MANUAL1D_LOOP
#define DEFAULT_LOOP_PATTERN LoopPattern::RANGE
#elif defined FOR_LOOP
#define DEFAULT_LOOP_PATTERN LoopPattern::SIMDFOR
#elif defined MDRANGE_LOOP
#define DEFAULT_LOOP_PATTERN LoopPattern::MDRANGE
#elif defined TP_INNERX_LOOP
#define DEFAULT_LOOP_PATTERN LoopPattern::TPX
#elif defined TPTTRTVR_LOOP
#define DEFAULT_LOOP_PATTERN LoopPattern::TPTTRTVR
#else
#define DEFAULT_LOOP_PATTERN LoopPattern::UNDEFINED
#endif

// 3D default loop pattern
template <typename Function>
inline void athena_for(const std::string & NAME,
                const int & KL, const int & KU,
                const int & JL, const int & JU,
                const int & IL, const int & IU,
                Function function) {
  athena_for(DEFAULT_LOOP_PATTERN,NAME,KL,KU,JL,JU,IL,IU,function);
}

// 4D default loop pattern
template <typename Function>
inline void athena_for(const std::string & NAME,
                const int & NL, const int & NU,
                const int & KL, const int & KU,
                const int & JL, const int & JU,
                const int & IL, const int & IU,
                Function function) {
  athena_for(DEFAULT_LOOP_PATTERN,NAME,NL,NU,KL,KU,JL,JU,IL,IU,function);
}

// 3D loop
template <typename Function>
inline void athena_for(LoopPattern lp, const std::string & NAME,
                const int & KL, const int & KU,
                const int & JL, const int & JU,
                const int & IL, const int & IU,
                Function function) {
  // Kokkos 1D Range
  if (lp == LoopPattern::RANGE) {
    const int NK = KU - KL + 1;
    const int NJ = JU - JL + 1;
    const int NI = IU - IL + 1;
    const int NKNJNI = NK*NJ*NI;
    const int NJNI = NJ * NI;
    Kokkos::parallel_for(NAME,
      NKNJNI,
      KOKKOS_LAMBDA (const int& IDX) {
      int k = IDX / NJNI;
      int j = (IDX - k*NJNI) / NI;
      int i = IDX - k*NJNI - j*NI;
      k += KL;
      j += JL;
      i += IL;
      function(k,j,i);
      });
  
  // MDRange loops
  } else if (lp == LoopPattern::MDRANGE) {
    Kokkos::parallel_for(NAME,
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {KL,JL,IL},{KU+1,JU+1,IU+1}),
      function);
  
  // TeamPolicy with single inner loops
  } else if (lp == LoopPattern::TPX) {
    const int NK = KU - KL + 1;
    const int NJ = JU - JL + 1;
    const int NKNJ = NK * NJ;
    Kokkos::parallel_for(NAME,
      team_policy (NKNJ, Kokkos::AUTO,KOKKOS_VECTOR_LENGTH),
      KOKKOS_LAMBDA (member_type team_member) {
        const int k = team_member.league_rank() / NJ + KL;
        const int j = team_member.league_rank() % NJ + JL;
        Kokkos::parallel_for(
          TPINNERLOOP<>(team_member,IL,IU+1), 
          [&] (const int i) {
            function(k,j,i);
          });
      });
  
  // TeamPolicy with nested TeamThreadRange and ThreadVectorRange
  } else if (lp == LoopPattern::TPTTRTVR) {
    const int NK = KU - KL + 1;
    Kokkos::parallel_for(NAME,
      team_policy (NK, Kokkos::AUTO,KOKKOS_VECTOR_LENGTH),
      KOKKOS_LAMBDA (member_type team_member) {
        const int k = team_member.league_rank() + KL;
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange<>(team_member,JL,JU+1),
          [&] (const int j) {
            Kokkos::parallel_for(
              Kokkos::ThreadVectorRange<>(team_member,IL,IU+1),
              [&] (const int i) {
                function(k,j,i);
              });
          });
      });
  
  // SIMD FOR loops
  } else if (lp == LoopPattern::SIMDFOR) {
    Kokkos::Profiling::pushRegion(NAME);
    for (auto k = KL; k <= KU; k++)
      for (auto j = JL; j <= JU; j++)
        #pragma omp simd
        for (auto i = IL; i <= IU; i++)
          function(k,j,i);
    Kokkos::Profiling::popRegion();
  } else {
    throw std::runtime_error("Unknown/undefined LoopPattern used.");
  }
}

// 4D loop
template <typename Function>
inline void athena_for(LoopPattern lp, const std::string & NAME,
                const int NL, const int NU,
                const int KL, const int KU,
                const int JL, const int JU,
                const int IL, const int IU,
                Function function) {
  
  // Kokkos 1D Range
  if (lp == LoopPattern::RANGE) {
    const int NN = (NU) - (NL) + 1;
    const int NK = (KU) - (KL) + 1;
    const int NJ = (JU) - (JL) + 1;
    const int NI = (IU) - (IL) + 1;
    const int NNNKNJNI = NN*NK*NJ*NI;
    const int NKNJNI = NK*NJ*NI;
    const int NJNI = NJ * NI;
    Kokkos::parallel_for(NAME,
      NNNKNJNI,
      KOKKOS_LAMBDA (const int& IDX) {
      int n = IDX / NKNJNI;
      int k = (IDX - n*NKNJNI) / NJNI;
      int j = (IDX - n*NKNJNI - k*NJNI) / NI;
      int i = IDX - n*NKNJNI - k*NJNI - j*NI;
      n += (NL);
      k += (KL);
      j += (JL);
      i += (IL);
      function(n,k,j,i);
      });
  
  // MDRange loops
  } else if (lp == LoopPattern::MDRANGE) {
    Kokkos::parallel_for(NAME,
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
        {NL,KL,JL,IL},{NU+1,KU+1,JU+1,IU+1}),
      function);

  // TeamPolicy loops
  } else if (lp == LoopPattern::TPX) {
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
  
  // TeamPolicy with nested TeamThreadRange and ThreadVectorRange
  } else if (lp == LoopPattern::TPTTRTVR) {
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

  // SIMD FOR loops
  } else if (lp == LoopPattern::SIMDFOR) {
    Kokkos::Profiling::pushRegion(NAME);
    for (auto n = NL; n <= NU; n++)
      for (auto k = KL; k <= KU; k++)
        for (auto j = JL; j <= JU; j++)
          #pragma omp simd
          for (auto i = IL; i <= IU; i++)
            function(n,k,j,i);
    Kokkos::Profiling::popRegion();
  } else {
    throw std::runtime_error("Unknown/undefined LoopPattern used.");
  }
}

#endif // ATHENA_HPP_
