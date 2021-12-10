/*
 * Everything that doesn't fit somewhere else.  General C/C++ convenience functions. 
 */
#pragma once

#include "decs.hpp"

#include <memory>
#include <string>
#include <stdexcept>

using namespace std; // This allows CUDA to override max & min

/**
 * This takes a number n and clips it to lie on the real line between 'lower' and 'upper'
 * If n is NaN, it returns the *lower* bound, unless this is also NaN, in which case it returns the upper.
 * Note that you can disable a bound by passing NaN.
 *
 * Lightly edited from https://stackoverflow.com/questions/9323903/most-efficient-elegant-way-to-clip-a-number
 */
template <typename T>
KOKKOS_INLINE_FUNCTION T clip(const T& n, const T& lower, const T& upper)
{
#if DEBUG
  //if (isnan(n)) printf("Clipping a NaN value!\n");
  //if (n > upper) printf("Clip %g to %g\n", n, upper);
  //if (n < lower) printf("Clip %g to %g\n", n, lower);
#endif
  return min(max(lower, n), upper);
}
