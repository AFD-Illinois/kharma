/* 
 *  File: kharma_utils.hpp
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
#pragma once

#include "decs.hpp"

#include <memory>
#include <string>
#include <stdexcept>

/*
 * A file for everything that doesn't fit somewhere else.  General C/C++ convenience functions.
 */

/**
 * This takes a number n and clips it to lie on the real line between 'lower' and 'upper'
 * If n is NaN, it returns the *lower* bound, unless this is also NaN, in which case it returns the upper.
 * Note that you can disable a bound by passing NaN.
 * If lower > upper, clip() will return the bound marked 'upper', that is, the lower number.
 *
 * Lightly edited from https://stackoverflow.com/questions/9323903/most-efficient-elegant-way-to-clip-a-number
 * Note that in C++17+ this can likely be a straight std::clamp call
 */
template <typename T>
KOKKOS_INLINE_FUNCTION T clip(const T& n, const T& lower, const T& upper)
{
#if TRACE
  // This isn't so useful without context
  //if (m::isnan(n)) printf("Clipping a NaN value!\n");
  //if (n > upper) printf("Clip %g to %g\n", n, upper);
  //if (n < lower) printf("Clip %g to %g\n", n, lower);
#endif
    return m::min(m::max(lower, n), upper);
}
// Version which "bounces" any excess over the bounds, useful for the polar coordinate
template <typename T>
KOKKOS_INLINE_FUNCTION T bounce(const T& n, const T& lower, const T& upper)
{
    return (n < lower) ? 2*lower - n : ( (n > upper) ? 2*upper - n : n );
}
// Version which "excises" anything within a range
template <typename T>
KOKKOS_INLINE_FUNCTION T excise(const T& n, const T& center, const T& range)
{
    return (m::abs(n - center) > range) ? n : ( (n > center) ? center + range : center - range );
}

template <typename T>
KOKKOS_INLINE_FUNCTION T close_to(const T& x, const T& y, const Real& rel_tol=1e-8, const Real& abs_tol=1e-8)
{
    return ((abs(x - y) / y) < rel_tol) || (abs(x) < abs_tol && abs(y) < abs_tol);
}

// Quickly zero n elements of an array
// Types can fail to resolve if gzeroN() calls zeroN(),
// so we duplicate code a bit
template <typename T>
KOKKOS_INLINE_FUNCTION void zero(T* a, const int& n)
{
    memset(a, 0, n*sizeof(T));
}
template <typename T>
KOKKOS_INLINE_FUNCTION void gzero(T a[GR_DIM])
{
    memset(a, 0, GR_DIM*sizeof(T));
}
template <typename T>
KOKKOS_INLINE_FUNCTION void zero2(T* a[], const int& n)
{
    memset(&(a[0][0]), 0, n*sizeof(T));
}
template <typename T>
KOKKOS_INLINE_FUNCTION void gzero2(T a[GR_DIM][GR_DIM])
{
    memset(&(a[0][0]), 0, GR_DIM*GR_DIM*sizeof(T));
}
