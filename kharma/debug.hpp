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
 * Calculate maximum divergence of magnetic field, to check it is being preserved ==0
 */
double MaxDivB(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain=IndexDomain::interior);

/**
 * Version of MaxDivB operating on Primitives
 */
double MaxDivB_P(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain=IndexDomain::interior);

/**
 * Print any diagnostic values which don't require pflag/fflag to calculate/print
 */
TaskStatus Diagnostic(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain=IndexDomain::interior);

/**
 * Check the max signal speed (ctop) for 0-values or NaNs.
 * This is a final warning that something is very wrong and we should crash.
 */
TaskStatus CheckNaN(std::shared_ptr<MeshBlockData<Real>>& rc, int dir, IndexDomain domain=IndexDomain::interior);

/**
 * The compiler is not so good with aliases.  Guide it.
 */
using ParArrayNDIntHost = ParArrayNDGeneric<Kokkos::View<int ******, parthenon::LayoutWrapper, Kokkos::HostSpace::memory_space>>;

/**
 * Function for counting & printing pflags.  Note this is defined host-side! Call pflags.getHostMirrorAndCopy() first!
 */
int CountPFlags(std::shared_ptr<MeshBlock> pmb, ParArrayNDIntHost pflag, IndexDomain domain=IndexDomain::entire, int verbose=0);

/**
 * Function for counting & printing pflags.  Note this is defined host-side! Call fflags.getHostMirrorAndCopy() first!
 */
int CountFFlags(std::shared_ptr<MeshBlock> pmb, ParArrayNDIntHost fflag, IndexDomain domain=IndexDomain::interior, int verbose=0);

// Misc print functions
void print_a_geom_tensor(const GeomTensor2 g, const int& i, const int& j);
void print_a_geom_tensor3(const GeomTensor3 g, const int& i, const int& j);
void compare_P_U(std::shared_ptr<MeshBlockData<Real>>& rc, const int& k, const int& j, const int& i);