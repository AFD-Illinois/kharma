#ifndef __KOKKOSBATCHED_APPLY_PIVOT_DECL_HPP__
#define __KOKKOSBATCHED_APPLY_PIVOT_DECL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

///
/// TeamVector
/// ==========
template <typename MemberType, typename ArgSide, typename ArgDirect>
struct TeamVectorApplyPivot {
  template <typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const int piv, const AViewType &A);

  template <typename PivViewType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const PivViewType piv,
                                           const AViewType &A);
};

template <typename ArgSide, typename ArgDirect>
struct SerialApplyPivot {
  template <typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int piv, const AViewType &A);

  template <typename PivViewType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const PivViewType piv,
                                           const AViewType &A);
};

}  // namespace KokkosBatched

#include "KokkosBatched_ApplyPivot_Impl.hpp"

#endif
