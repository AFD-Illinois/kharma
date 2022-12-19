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

#include "bondi.hpp"
#include "grmhd_functions.hpp"

/**
 * Any functions related to KHARMA's treatment of boundary conditions.
 * These largely build on/fill in Parthenon's boundary functions,
 * which KHARMA uses to handle all MPI & periodic boundaries.
 * 
 * Thus this Namespace is for outflow, reflecting, and problem-specific
 * bounds, which KHARMA has to handle separately from Parthenon.
 */
namespace KBoundaries {

/**
 * Any KHARMA-defined boundaries.
 * These usually behave like Parthenon's Outflow in X1 and Reflect in X2, except
 * that they operate on the fluid primitive variables p,u,u1,u2,u3.
 * All other variables are unchanged.
 * 
 * These functions also handle calling through to problem-defined boundaries e.g. Bondi outer X1
 * 
 * LOCKSTEP: these functions respect P and return consistent P<->U
 */
void InnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void InnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void OuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);

/**
 * Fix fluxes on physical boundaries. Ensure no inflow flux, correct B fields on reflecting conditions.
 */
TaskStatus FixFlux(MeshData<Real> *rc);

/**
 * Add a synchronization step to a task list tl, dependent upon taskID t_start, syncing mesh mc1
 * 
 * This sequence is used identically in several places, so it makes sense
 * to define once and use elsewhere.
 * TODO could make member of a HARMDriver/ImExDriver superclass?
 */
TaskID AddBoundarySync(TaskID t_start, TaskList &tl, std::shared_ptr<MeshData<Real>> mc1);

/**
 * Single call to sync all boundary conditions.
 * Used anytime boundary sync is needed outside the usual loop of steps.
 */
void SyncAllBounds(std::shared_ptr<MeshData<Real>> md, bool apply_domain_bounds=true);

/**
 * Check for flow into simulation and reset velocity to eliminate it
 * TODO does Parthenon do something like this for outflow bounds already?
 *
 * @param type: 0 to check outflow from EH, 1 to check inflow from outer edge
 */
KOKKOS_INLINE_FUNCTION void check_inflow(const GRCoordinates &G, const VariablePack<Real>& P, const IndexDomain domain,
                                         const int& index_u1, const int& k, const int& j, const int& i)
{
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
