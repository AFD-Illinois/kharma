/*
 * Everything that doesn't fit somewhere else.  General C/C++ convenience functions. 
 */
#pragma once

#include "decs.hpp"

#include <memory>
#include <string>
#include <stdexcept>

using namespace std; // This allows CUDA to override max & min
// Thanks https://stackoverflow.com/questions/9323903/most-efficient-elegant-way-to-clip-a-number
template <typename T>
KOKKOS_INLINE_FUNCTION T clip(const T& n, const T& lower, const T& upper) {
  return max(lower, min(n, upper));
}
