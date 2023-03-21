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
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Single-call divergence cleanup.  Lots of MPI syncs, probably slow to use in task lists.
 */
void CleanupDivergence(std::shared_ptr<MeshData<Real>>& md);

/**
 * Add the iterative tasks required for B field cleanup to the tasklist
 * Likely faster than above if we want to clean periodically
 */
//void AddBCleanupTasks(TaskList tl, TaskID t_dep);

/**
 * Remove the extra solver fields which B_Cleanup added during initialization.
 * Must be run before every step as the meshblocks are reconstructed per-step from
 * package variable lists.
 */
TaskStatus RemoveExtraFields(BlockList_t &blocks);

/**
 * Calculate the laplacian using divergence at corners
 */
TaskStatus CornerLaplacian(MeshData<Real>* md, const std::string& p_var, MeshData<Real>* md_again, const std::string& lap_var);

/**
 * Apply B -= grad(P) to subtract divergence from the magnetic field
 */
TaskStatus ApplyP(MeshData<Real> *msolve, MeshData<Real> *md);

} // namespace B_Cleanup
