/* 
 *  File: boundaries.hpp
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

#include "flux.hpp"
#include "grmhd_functions.hpp"

/**
 * This package has any functions related to KHARMA's treatment of "domain" boundary conditions:
 * the exterior simulation edges, as opposed to internal meshblock boundaries.
 * 
 * This package implements Parthenon's "user" boundary conditions in order to add some
 * features related to GRMHD.  
 */
namespace KBoundaries {

/**
 * Choose which boundary conditions will be used based on inputs,
 * declare any fields needed to store e.g. constant boundary conditions
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Generic KHARMA override function for Parthenon domain boundary conditions.
 * This is registered as the "user" boundary condition with Parthenon, and
 * replaces Parthenon's reflecting or outflow boundary conditions wherever those
 * would be applied.
 * 
 * Mostly calls "DefaultBoundary," unless overridden by a problem.
 * 
 * LOCKSTEP: respects P and return consistent P<->U
 */
void ApplyBoundary(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse);
// Template version to conform to Parthenon's calling convention. See above.
template <IndexDomain domain>
inline void ApplyBoundaryTemplate(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse)
{ ApplyBoundary(rc, domain, coarse); }

/**
 * Boundary conditions when not overridden by a problem (or handled by Parthenon).
 * Outflow boundaries in X1 with an optional check for inflow, Reflecting boundaries in X2
 */
void DefaultBoundary(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse);

/**
 * Dirichlet boundaries implementation.
 * Problems can assign these to the KHARMA*Boundary callbacks, then fill the "bound.*"
 * fields populated as a part of the "Boundaries" package.
 */
void Dirichlet(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse);

/**
 * Fix fluxes on physical boundaries.
 * 1. Ensure no inflow of density onto the domain
 * 2. Ensure flux through the size-zero faces on poles is zero
 * The latter may be unnecessary
 */
TaskStatus FixFlux(MeshData<Real> *rc);

// INTERNAL FUNCTIONS

/**
 * Check for inflowing material on an outflow boundary, and
 * reset the velocity of such material so it is no longer inflowing.
 */
void CheckInflow(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse);

/**
 * KHARMA is very particular about corner boundaries.
 * In particular, we apply the outflow boundary over ALL X2 & X3.
 * Then we apply the polar bound only where outflow is not applied,
 * and periodic bounds only where neither other bound applies.
 * The latter is accomplished regardless of Parthenon's definitions,
 * since these functions are run after Parthenon's MPI boundary syncs &
 * replace whatever they've done.
 * However, the former must be added after the X2 boundary call,
 * replacing the reflecting conditions in the X1/X2 corner (or in 3D, edge)
 * with outflow conditions based on the updated ghost cells.
 */
void FixCorner(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse);

/**
 * We apply Parthenon's boundary condition implementations, which are not GR-aware.
 * When applied to the magnetic field values, the result must be scaled by the relative change
 * in metric determinant.  This function applies that change.
 */
void CorrectBField(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse);

/**
 * Check for velocity toward the simulation domain in a zone, and eliminate it.
 */
KOKKOS_INLINE_FUNCTION void check_inflow(const GRCoordinates &G, const VariablePack<Real>& P, const IndexDomain domain,
                                         const int& index_u1, const int& k, const int& j, const int& i)
{
    // TODO fewer temporaries?
    Real uvec[NVEC], ucon[GR_DIM];
    VLOOP uvec[v] = P(index_u1 + v, k, j, i);
    GRMHD::calc_ucon(G, uvec, k, j, i, Loci::center, ucon);

    if (((ucon[1] > 0.) && (domain == IndexDomain::inner_x1)) ||
        ((ucon[1] < 0.) && (domain == IndexDomain::outer_x1)))
    {
        // Find gamma and remove it from primitive velocity
        double gamma = GRMHD::lorentz_calc(G, uvec, k, j, i, Loci::center);
        VLOOP uvec[v] /= gamma;

        // Reset radial velocity so radial 4-velocity is zero
        Real alpha = 1. / m::sqrt(-G.gcon(Loci::center, j, i, 0, 0));
        Real beta1 = G.gcon(Loci::center, j, i, 0, 1) * alpha * alpha;
        uvec[V1] = beta1 / alpha;

        // Now find new gamma and put it back in
        Real vsq = G.gcov(Loci::center, j, i, 1, 1) * uvec[V1] * uvec[V1] +
                   G.gcov(Loci::center, j, i, 2, 2) * uvec[V2] * uvec[V2] +
                   G.gcov(Loci::center, j, i, 3, 3) * uvec[V3] * uvec[V3] +
        2. * (G.gcov(Loci::center, j, i, 1, 2) * uvec[V1] * uvec[V2] +
              G.gcov(Loci::center, j, i, 1, 3) * uvec[V1] * uvec[V3] +
              G.gcov(Loci::center, j, i, 2, 3) * uvec[V2] * uvec[V3]);

        clip(vsq, 1.e-13, 1. - 1./(50.*50.));

        gamma = 1./m::sqrt(1. - vsq);

        VLOOP uvec[v] *= gamma;
        VLOOP P(index_u1 + v, k, j, i) = uvec[v];
    }
}

}
