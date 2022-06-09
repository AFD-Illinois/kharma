/*
 *  File: b_cleanup.hpp
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

#include <memory>

#include <parthenon/parthenon.hpp>

#include "grmhd_functions.hpp"
#include "types.hpp"

using namespace parthenon;

/**
 * This physics package implements an elliptic solver which minimizes the divergence of
 * the magnetic field B, most useful for mesh resizing.
 * Written to leave open the possibility of using this at every 
 * 
 * Mostly now, it is used when resizing input arrays
 */
namespace B_Cleanup {
/**
 * Declare fields, initialize (few) parameters
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);

/**
 * Calculate the field divergence, and sum the absolute value as a reduction
 * (for convergence comparisons).
 */
TaskStatus CalcSumDivB(MeshData<Real> *du, Real& reduce_sum);

/**
 * Set P = divB as initial guess
 */
TaskStatus InitP(MeshData<Real> *md);

/**
 * Take a Gauss-Seidel/SOR step.
 */
TaskStatus UpdateP(MeshData<Real> *md);

/**
 * Functions to calculate the remaining error, that is, the difference del^2 p - divB
 */
TaskStatus SumError(MeshData<Real> *du, Real& reduce_sum);
TaskStatus SumP(MeshData<Real> *md, Real& reduce_sum);
TaskStatus MaxError(MeshData<Real> *md, Real& reduce_max);

/**
 * Apply B -= grad(P) to subtract divergence from the magnetic field
 */
TaskStatus ApplyP(MeshData<Real> *md);

/**
 * Single-call divergence cleanup.  Lots of MPI syncs, probably slow to use in task lists.
 */
void CleanupDivergence(std::shared_ptr<MeshData<Real>>& md, Driver* driver, ParameterInput *pin, bool read_p=false);

/**
 * Add the iterative tasks required for B field cleanup to the tasklist
 * Likely faster than above if we want to clean periodically
 */
//void AddBCleanupTasks(TaskList tl, TaskID t_dep);

} // namespace B_Cleanup
