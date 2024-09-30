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

#include "reductions_variables.hpp"

#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "types.hpp"

namespace Reductions {

// Think about how to do channels as not ints
//constexpr enum class Channel{fflag, pflag, iflag, };

/**
 * These, too, are a package.
 * Mostly it exists to keep track of Reducers, so we can clean them up to keep MPI happy.
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Perform a reduction using operation 'op' over a given domain.
 * Note startx/stopx are in *embedding* coordinates e.g. Kerr-Schild, not MKS/FMKS
 * 
 * This should be used for all 2D shell sums not around the EH:
 * Just set equal min/max, 2D slices are detected
 */
template<Var var, typename T>
T DomainReduction(MeshData<Real> *md, UserHistoryOperation op, const GReal startx[3], const GReal stopx[3], int channel=-1);
template<Var var, typename T>
T DomainReduction(MeshData<Real> *md, UserHistoryOperation op, int channel=-1) {
    const GReal startx[3] = {std::numeric_limits<GReal>::min(), std::numeric_limits<GReal>::min(), std::numeric_limits<GReal>::min()};
    const GReal stopx[3] = {std::numeric_limits<GReal>::max(), std::numeric_limits<GReal>::max(), std::numeric_limits<GReal>::max()};
    return DomainReduction<var, T>(md, op, startx, stopx, channel);
}
template<Var var, typename T>
T ShellReduction(MeshData<Real> *md, UserHistoryOperation op, GReal r, int channel=-1) {
    const GReal startx[3] = {r, std::numeric_limits<GReal>::min(), std::numeric_limits<GReal>::min()};
    const GReal stopx[3] = {r, std::numeric_limits<GReal>::max(), std::numeric_limits<GReal>::max()};
    return DomainReduction<var, T>(md, op, startx, stopx, channel);
}

// Parthenon doesn't allow taking options, so we define some common reductions
template<Var var>
Real SumAt0(MeshData<Real> *md)
{
    const GReal r_in = md->GetMeshPointer()->packages.Get("Reductions")->Param<GReal>("domain_r_in");
    return Reductions::ShellReduction<var, Real>(md, UserHistoryOperation::sum, r_in);
}
template<Var var>
Real SumAtEH(MeshData<Real> *md)
{
    const GReal r_eh = md->GetMeshPointer()->block_list[0]->coords.coords.get_horizon();
    return Reductions::ShellReduction<var, Real>(md, UserHistoryOperation::sum, r_eh);
}
template<Var var>
Real SumAt5M(MeshData<Real> *md)
{
    return Reductions::ShellReduction<var, Real>(md, UserHistoryOperation::sum, 5.);
}
template<Var var>
Real Total(MeshData<Real> *md)
{
    return Reductions::DomainReduction<var, Real>(md, UserHistoryOperation::sum);
}

/**
 * Start reductions with a value you have on hand
 */
template<typename T>
void Start(MeshData<Real> *md, int channel, T val, MPI_Op op);
template<typename T>
void StartToAll(MeshData<Real> *md, int channel, T val, MPI_Op op);

/**
 * Check the results of reductions that have been started.
 * Remember channels are COUNTED SEPARATELY between the 4 lists:
 * Real/default, int, vector<Real> and vector<int> (i.e. Flags)
 */
template<typename T>
T Check(MeshData<Real> *md, int channel);
template<typename T>
T CheckOnAll(MeshData<Real> *md, int channel);

/**
 * Check the results of reductions that have been started.
 * Remember channels are COUNTED SEPARATELY between the 4 lists:
 * Real/default, int, vector<Real> and vector<int> (i.e. Flags)
 */
template<typename T>
T Check(MeshData<Real> *md, int channel);

/**
 * Count instances of a particular flag value in the named field.
 * is_bitflag specifies whether multiple flags may be present and will be orthogonal (e.g. FFlag),
 * or whether flags receive consecutive integer values.
 */
int CountFlag(MeshData<Real> *md, std::string field_name, const int& flag_val, IndexDomain domain, bool is_bitflag);

/**
 * Count instances of all flags in the named field.
 * is_bitflag specifies whether multiple flags may be present and will be orthogonal (e.g. FFlag),
 * or whether flags receive consecutive integer values.
 */
std::vector<int> CountFlags(MeshData<Real> *md, std::string field_name, const std::map<int, std::string> &flag_values, IndexDomain domain, bool is_bitflag);

/**
 * Determine number of local flags hit with CountFlags, and send the value over MPI reducer 'channel'
 */
void StartFlagReduce(MeshData<Real> *md, std::string field_name, const std::map<int, std::string> &flag_values, IndexDomain domain, bool is_bitflag, int channel);

/**
 * Check a flag's MPI reduction and print any flags hit
 */
std::vector<int> CheckFlagReduceAndPrintHits(MeshData<Real> *md, std::string field_name, const std::map<int, std::string> &flag_values,
                                             IndexDomain domain, bool is_bitflag, int channel);

} // namespace Reductions

// See the file for why we do this
#include "reductions_impl.hpp"
