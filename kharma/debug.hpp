/* 
 *  File: debug.hpp
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

#include "mesh/mesh.hpp"
#include "mpi.hpp"

using namespace std;

/**
 * Check the max signal speed (ctop) for 0-values or NaNs.
 * This is a final warning that something is very wrong and we should crash.
 */
TaskStatus CheckNaN(MeshData<Real> *md, int dir, IndexDomain domain=IndexDomain::interior);

/**
 * Check the primitive and conserved variables for negative values that definitely shouldn't be negative
 * That is: primitive rho, u, conserved rho*u^t
 */
TaskStatus CheckNegative(MeshData<Real> *md, IndexDomain domain);

// The compiler is not so good with aliases.  We guide it.
// using ParArrayNDHost = ParArrayNDGeneric<Kokkos::View<Real ******, parthenon::LayoutWrapper, Kokkos::HostSpace::memory_space>>;
// using ParArrayNDIntHost = ParArrayNDGeneric<Kokkos::View<int ******, parthenon::LayoutWrapper, Kokkos::HostSpace::memory_space>>;

/**
 * Function for counting & printing pflags.  Note this needs a host-side array! Call pflags.getHostMirrorAndCopy() first!
 */
int CountPFlags(MeshData<Real> *md, IndexDomain domain=IndexDomain::entire, int verbose=0);

/**
 * Function for counting & printing pflags.  Note this needs a host-side array! Call fflags.getHostMirrorAndCopy() first!
 */
int CountFFlags(MeshData<Real> *md, IndexDomain domain=IndexDomain::interior, int verbose=0);

// Miscellaneous print functions.
KOKKOS_FORCEINLINE_FUNCTION void print_matrix(const std::string name, const double g[GR_DIM][GR_DIM], bool kill_on_nan=false)
{
    // Print a name and a matrix
    printf("%s:\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n%g\t%g\t%g\t%g\n", name.c_str(),
            g[0][0], g[0][1], g[0][2], g[0][3], g[1][0], g[1][1], g[1][2],
            g[1][3], g[2][0], g[2][1], g[2][2], g[2][3], g[3][0], g[3][1],
            g[3][2], g[3][3]);

    if (kill_on_nan) {
        // Additionally kill things if/when we hit NaNs
        DLOOP2 if (isnan(g[mu][nu])) exit(-1);
    }
}
KOKKOS_FORCEINLINE_FUNCTION void print_vector(const std::string name, const double v[GR_DIM], bool kill_on_nan=false)
{
    printf("%s: %g\t%g\t%g\t%g\n", name.c_str(), v[0], v[1], v[2], v[3]);

    if (kill_on_nan) {
        DLOOP2 if (isnan(v[nu])) exit(-1);
    }
}
