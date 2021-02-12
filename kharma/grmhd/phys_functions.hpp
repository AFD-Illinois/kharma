/* 
 *  File: phys_functions.hpp
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

/**
 * Device-side hydrodynamics functions
 * They are specifically the subset which require the fluid primitives P & B field both
 *
 * These functions mostly have several overloads, related to local vs global variables
 * One version usually takes a local cache e.g. P[NPRIM] of state indexed P[p]
 * The other version(s) take e.g. P, the pointer to the full array indexed by P(p,i,j,k)
 *
 * This allows easy fusing/splitting of loops & use in different contexts
 */

/**
 * Find gamma-factor of the fluid w.r.t. normal observer
 *
 * TODO Check qsq inline and/or fabs() it for output
 */
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates &G, const GridVars P,
                                             const int& k, const int& j, const int& i,
                                             const Loci loc)
{

    Real qsq = G.gcov(loc, j, i, 1, 1) * P(prims::u1, k, j, i) * P(prims::u1, k, j, i) +
               G.gcov(loc, j, i, 2, 2) * P(prims::u2, k, j, i) * P(prims::u2, k, j, i) +
               G.gcov(loc, j, i, 3, 3) * P(prims::u3, k, j, i) * P(prims::u3, k, j, i) +
            2. * (G.gcov(loc, j, i, 1, 2) * P(prims::u1, k, j, i) * P(prims::u2, k, j, i) +
                  G.gcov(loc, j, i, 1, 3) * P(prims::u1, k, j, i) * P(prims::u3, k, j, i) +
                  G.gcov(loc, j, i, 2, 3) * P(prims::u2, k, j, i) * P(prims::u3, k, j, i));

    return sqrt(1. + qsq);
}
KOKKOS_INLINE_FUNCTION Real lorentz_calc(const GRCoordinates &G, const Real uv[NVEC],
                                             const int& j, const int& i, const Loci loc)
{
    Real qsq = G.gcov(loc, j, i, 1, 1) * uv[0] * uv[0] +
               G.gcov(loc, j, i, 2, 2) * uv[1] * uv[1] +
               G.gcov(loc, j, i, 3, 3) * uv[2] * uv[2] +
            2. * (G.gcov(loc, j, i, 1, 2) * uv[0] * uv[1] +
                  G.gcov(loc, j, i, 1, 3) * uv[0] * uv[2] +
                  G.gcov(loc, j, i, 2, 3) * uv[1] * uv[2]);

    return sqrt(1. + qsq);
}