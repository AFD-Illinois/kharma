#ifndef __KOKKOSBATCHED_APPLY_PIVOT_INTERNAL_HPP__
#define __KOKKOSBATCHED_APPLY_PIVOT_INTERNAL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

///
/// TeamVector Internal Impl
/// ========================

///
/// Forward
///
struct TeamVectorApplyPivotVectorForwardInternal {
  template <typename MemberType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const int piv,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0) {
    if (piv != 0) {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        const int idx_p     = piv * as0;
        const ValueType tmp = A[0];
        A[0]                = A[idx_p];
        A[idx_p]            = tmp;
      });
    }
    return 0;
  }

  template <typename MemberType, typename IntType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const int plen,
                                           const IntType *KOKKOS_RESTRICT p,
                                           const int ps0,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0) {
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      for (int i = 0; i < plen; ++i) {
        const int piv = p[i * ps0];
        if (piv != 0) {
          const int idx_i = i * as0, idx_p = (i + piv) * as0;
          const ValueType tmp = A[idx_i];
          A[idx_i]            = A[idx_p];
          A[idx_p]            = tmp;
        }
      }
    });
    return 0;
  }
};

/// Pivot a row
struct TeamVectorApplyPivotMatrixForwardInternal {
  template <typename MemberType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const int n, const int piv,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0, const int as1) {
    if (piv != 0) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n),
                           [&](const int &j) {
                             ValueType *KOKKOS_RESTRICT A_at_j = A + j * as1;
                             const int idx_p                   = piv * as0;
                             const ValueType tmp               = A_at_j[0];
                             A_at_j[0]                         = A_at_j[idx_p];
                             A_at_j[idx_p]                     = tmp;
                           });
    }
    return 0;
  }

  template <typename MemberType, typename IntType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const int n, const int plen,
                                           const IntType *KOKKOS_RESTRICT p,
                                           const int ps0,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0, const int as1) {
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), [&](const int &j) {
      ValueType *KOKKOS_RESTRICT A_at_j = A + j * as1;
      for (int i = 0; i < plen; ++i) {
        const int piv = p[i * ps0];
        if (piv != 0) {
          const int idx_i = i * as0, idx_p = (i + piv) * as0;
          const ValueType tmp = A_at_j[idx_i];
          A_at_j[idx_i]       = A_at_j[idx_p];
          A_at_j[idx_p]       = tmp;
        }
      }
    });
    return 0;
  }
};

///
/// Backward
///
struct TeamVectorApplyPivotVectorBackwardInternal {
  template <typename MemberType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const int piv,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0) {
    if (piv != 0) {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        const int idx_p     = piv * as0;
        const ValueType tmp = A[0];
        A[0]                = A[idx_p];
        A[idx_p]            = tmp;
      });
    }
    return 0;
  }

  template <typename MemberType, typename IntType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const int plen,
                                           const IntType *KOKKOS_RESTRICT p,
                                           const int ps0,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0) {
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
      for (int i = (plen - 1); i >= 0; --i) {
        const int piv = p[i * ps0];
        if (piv != 0) {
          const int idx_i = i * as0, idx_p = (i + piv) * as0;
          const ValueType tmp = A[idx_i];
          A[idx_i]            = A[idx_p];
          A[idx_p]            = tmp;
        }
      }
    });
    return 0;
  }
};

/// Pivot a row
struct TeamVectorApplyPivotMatrixBackwardInternal {
  template <typename MemberType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const int n, const int piv,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0, const int as1) {
    if (piv != 0) {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n),
                           [&](const int &j) {
                             ValueType *KOKKOS_RESTRICT A_at_j = A + j * as1;
                             const int idx_p                   = piv * as0;
                             const ValueType tmp               = A_at_j[0];
                             A_at_j[0]                         = A_at_j[idx_p];
                             A_at_j[idx_p]                     = tmp;
                           });
    }
    return 0;
  }

  template <typename MemberType, typename IntType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const int n, const int plen,
                                           const IntType *KOKKOS_RESTRICT p,
                                           const int ps0,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0, const int as1) {
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), [&](const int &j) {
      ValueType *KOKKOS_RESTRICT A_at_j = A + j * as1;
      for (int i = (plen - 1); i >= 0; --i) {
        const int piv = p[i * ps0];
        if (piv != 0) {
          const int idx_i = i * as0, idx_p = (i + piv) * as0;
          const ValueType tmp = A_at_j[idx_i];
          A_at_j[idx_i]       = A_at_j[idx_p];
          A_at_j[idx_p]       = tmp;
        }
      }
    });
    return 0;
  }
};

///
/// Serial Internal Impl
/// ========================

///
/// Forward
///

struct SerialApplyPivotVectorForwardInternal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int piv,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0) {
    if (piv != 0) {
      const int idx_p     = piv * as0;
      const ValueType tmp = A[0];
      A[0]                = A[idx_p];
      A[idx_p]            = tmp;
    }
    return 0;
  }

  // template <typename MemberType, typename IntType, typename ValueType>
  // KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
  //                                          const int plen,
  //                                          const IntType *KOKKOS_RESTRICT p,
  //                                          const int ps0,
  //                                          /* */ ValueType *KOKKOS_RESTRICT A,
  //                                          const int as0) {
  //   Kokkos::single(Kokkos::PerTeam(member), [&]() {
  //     for (int i = 0; i < plen; ++i) {
  //       const int piv = p[i * ps0];
  //       if (piv != 0) {
  //         const int idx_i = i * as0, idx_p = (i + piv) * as0;
  //         const ValueType tmp = A[idx_i];
  //         A[idx_i]            = A[idx_p];
  //         A[idx_p]            = tmp;
  //       }
  //     }
  //   });
  //   return 0;
  // }
};

/// Pivot a row
struct SerialApplyPivotMatrixForwardInternal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int n, const int piv,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0, const int as1) {
    if (piv != 0) {
      for (int j=0; j < n; j++) {
        ValueType *KOKKOS_RESTRICT A_at_j = A + j * as1;
        const int idx_p                   = piv * as0;
        const ValueType tmp               = A_at_j[0];
        A_at_j[0]                         = A_at_j[idx_p];
        A_at_j[idx_p]                     = tmp;
      }
    }
    return 0;
  }

  // template <typename MemberType, typename IntType, typename ValueType>
  // KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
  //                                          const int n, const int plen,
  //                                          const IntType *KOKKOS_RESTRICT p,
  //                                          const int ps0,
  //                                          /* */ ValueType *KOKKOS_RESTRICT A,
  //                                          const int as0, const int as1) {
  //   Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), [&](const int &j) {
  //     ValueType *KOKKOS_RESTRICT A_at_j = A + j * as1;
  //     for (int i = 0; i < plen; ++i) {
  //       const int piv = p[i * ps0];
  //       if (piv != 0) {
  //         const int idx_i = i * as0, idx_p = (i + piv) * as0;
  //         const ValueType tmp = A_at_j[idx_i];
  //         A_at_j[idx_i]       = A_at_j[idx_p];
  //         A_at_j[idx_p]       = tmp;
  //       }
  //     }
  //   });
  //   return 0;
  // }
};

///
/// Backward
///
struct SerialApplyPivotVectorBackwardInternal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int piv,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0) {
    if (piv != 0) {
      const int idx_p     = piv * as0;
      const ValueType tmp = A[0];
      A[0]                = A[idx_p];
      A[idx_p]            = tmp;
    }
    return 0;
  }

  template <typename IntType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int plen,
                                           const IntType *KOKKOS_RESTRICT p,
                                           const int ps0,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0) {
    for (int i = (plen - 1); i >= 0; --i) {
      const int piv = p[i * ps0];
      if (piv != 0) {
        const int idx_i = i * as0, idx_p = (i + piv) * as0;
        const ValueType tmp = A[idx_i];
        A[idx_i]            = A[idx_p];
        A[idx_p]            = tmp;
      }
    }
    return 0;
  }
};

/// Pivot a row
struct SerialApplyPivotMatrixBackwardInternal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int n, const int piv,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0, const int as1) {
    if (piv != 0) {
      for (int j=0; j < n; ++j) {
        ValueType *KOKKOS_RESTRICT A_at_j = A + j * as1;
        const int idx_p                   = piv * as0;
        const ValueType tmp               = A_at_j[0];
        A_at_j[0]                         = A_at_j[idx_p];
        A_at_j[idx_p]                     = tmp;
      }
    }
    return 0;
  }

  template <typename IntType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int n, const int plen,
                                           const IntType *KOKKOS_RESTRICT p,
                                           const int ps0,
                                           /* */ ValueType *KOKKOS_RESTRICT A,
                                           const int as0, const int as1) {
    for (int j=0; j < n; ++j) {
      ValueType *KOKKOS_RESTRICT A_at_j = A + j * as1;
      for (int i = (plen - 1); i >= 0; --i) {
        const int piv = p[i * ps0];
        if (piv != 0) {
          const int idx_i = i * as0, idx_p = (i + piv) * as0;
          const ValueType tmp = A_at_j[idx_i];
          A_at_j[idx_i]       = A_at_j[idx_p];
          A_at_j[idx_p]       = tmp;
        }
      }
    }
    return 0;
  }
};


}  // namespace KokkosBatched

#endif
