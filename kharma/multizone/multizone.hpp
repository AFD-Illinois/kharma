/* 
 *  File: multizone.hpp
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
#include "types.hpp"

#include <parthenon/parthenon.hpp>


/**
 * This package is for running only pieces of a grid, without stepping forward every zone.
 * It allows for multi-scale simulations, by 
 */
namespace Multizone {
/**
 * Initialization of multi-zone.
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * diagnostics for multi-zone runs
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md);

/**
 * calculate Bondi radius for a given sonic radius rs
 */
KOKKOS_INLINE_FUNCTION Real CalcRB(const Real &gam, const Real &rs) {
    Real n = 1. / (gam - 1);
    if (m::abs(gam - 5. / 3.) < 1e-2) {
        return (80. * m::pow(rs, 2.) / (27. * gam));
    }
    else return 4. * (1. + n) * rs / ((2. * (n + 3.) - 9.) * gam);
}

/**
 * calculate runtime per zone similarly to the original multizone runs
 */
KOKKOS_INLINE_FUNCTION Real CalcRuntime(const Real &r_in, const int &base, const Real &gam, const Real &rs, const bool &loc_tchar) {
    // copying the old calculations even if there is some room for improvement
    // for ex) it can be min(tB, tchar_at_prev_rin)
    Real effectve_rout = r_in * m::pow(base, 2.); // as if all zones are evenly sized annuli
    Real r_b = CalcRB(gam, rs);
    if (loc_tchar) return effectve_rout / m::sqrt(1. / effectve_rout + 1. / r_b);
    else return m::pow(m::min(effectve_rout, r_b), 3. / 2.);
}

/**
 * Depending on the location in the V cycles, decide which blocks are active and where to apply boundary conditions for the current step
 * 
 */
void DecideActiveBlocksAndBoundaryConditions(Mesh *pmesh, const SimTime &tm, bool *is_active, bool apply_boundary_condition[][BOUNDARY_NFACES], const bool verbose);

/**
 * Decide whether or not to progress in the V cycle
 * 
 */
TaskStatus DecideToSwitch(MeshData<Real> *md, const SimTime &tm, bool &switch_zone);

/**
 * Decide which blocks are active for the next step and progress in the V cycle, in order to determine dt
 * 
 */
TaskStatus DecideNextActiveBlocks(MeshData<Real> *md, const SimTime &tm, const int iblock, bool &is_active, const bool switch_zone);

/**
 * Average EMFs on seams (internal boundaries) between this block and specified other blocks.
 * This is a light wrapper around B_CT::AverageEMF which just applies it to certain boundaries.
 */
TaskStatus AverageEMFSeams(MeshData<Real> *md_emf_only, bool *apply_boundary_condition);

} // Multizone
