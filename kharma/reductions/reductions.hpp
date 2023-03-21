/* 
 *  File: reductions.hpp
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

#include "debug.hpp"

#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "types.hpp"

// This is for flux/accretion rate 
#define REDUCE_FUNCTION_ARGS_EH const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p, \
                        const VariableFluxPack<Real>& U, const VarMap& m_u, const Real& gam, \
                        const int& k, const int& j, const int& i

// Notice this list also includes a generic Real-type argument: this is for denoting a radius or placement.
// Provided as argument in case reductions at/within/etc multiple places are desired
// (e.g., disk and jet, inner & outer, multiple radii)
// TODO take off 'b' from arg list and pass block contents?
#define REDUCE_FUNCTION_ARGS_MESH const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p, \
                        const VariableFluxPack<Real>& U, const VarMap& m_u, const Real& gam, \
                        const int& k, const int& j, const int& i, const Real& arg

namespace Reductions {

/**
 * Perform a reduction using operation 'op' over a spherical shell at the given zone, measured from left side of
 * innermost block in radius.
 * As this only runs on innermost blocks, this is intended for accretion/event horizon
 * measurements in black hole simulations.
 */
Real EHReduction(MeshData<Real> *md, UserHistoryOperation op, std::function<Real(REDUCE_FUNCTION_ARGS_EH)> fn, int zone);

/**
 * Perform a reduction using operation 'op' over all zones.
 * The extra 'arg' is passed as the last argument to the device-side function.
 * It is generally used to denote a radius inside, outside, or at which the reduction should be taken.
 * This should be used for 2D shell sums not at the EH: just divide the function result by the zone spacing dx1.
 */
Real DomainReduction(MeshData<Real> *md, UserHistoryOperation op, std::function<Real(REDUCE_FUNCTION_ARGS_MESH)> fn, Real arg);

/**
 * Count instances of a particular flag value in the named field.
 * is_bitflag specifies whether multiple flags may be present and will be orthogonal (e.g. FFlag),
 * or whether flags receive consecutive integer values.
 */
int CountFlag(MeshData<Real> *md, std::string field_name, const int& flag_val, IndexDomain domain, bool is_bitflag);

/**
 * Count instances of a particular flag value in the named field.
 * is_bitflag specifies whether multiple flags may be present and will be orthogonal (e.g. FFlag),
 * or whether flags receive consecutive integer values.
 * TODO could return numbers for all flags instead of just printing
 */
int CountFlags(MeshData<Real> *md, std::string field_name, std::map<int, std::string> flag_values, IndexDomain domain, int verbose, bool is_bitflag);

} // namespace Reductions
