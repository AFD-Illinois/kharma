//
#pragma once

#include "decs.hpp"

#include "bondi.hpp"
#include "mhd_functions.hpp"

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
 * Check for flow into simulation and reset velocity to eliminate it
 * TODO does Parthenon do something like this for outflow bounds already?
 *
 * @param type: 0 to check outflow from EH, 1 to check inflow from outer edge
 */
KOKKOS_INLINE_FUNCTION void check_inflow(const GRCoordinates &G, const VariablePack<Real>& P, const int& u_start, const int& k, const int& j, const int& i, int type)
{
    Real uvec[NVEC], ucon[GR_DIM];
    VLOOP uvec[v] = P(u_start + v, k, j, i);
    GRMHD::calc_ucon(G, uvec, k, j, i, Loci::center, ucon);

    if (((ucon[1] > 0.) && (type == 0)) ||
        ((ucon[1] < 0.) && (type == 1)))
    {
        // Find gamma and remove it from primitive velocity
        double gamma = GRMHD::lorentz_calc(G, uvec, k, j, i, Loci::center);
        VLOOP uvec[v] /= gamma;

        // Reset radial velocity so radial 4-velocity is zero
        Real alpha = 1. / sqrt(-G.gcon(Loci::center, j, i, 0, 0));
        Real beta1 = G.gcon(Loci::center, j, i, 0, 1) * alpha * alpha;
        uvec[0] = beta1 / alpha;

        // Now find new gamma and put it back in
        Real vsq = G.gcov(Loci::center, j, i, 1, 1) * uvec[0] * uvec[0] +
                   G.gcov(Loci::center, j, i, 2, 2) * uvec[1] * uvec[1] +
                   G.gcov(Loci::center, j, i, 3, 3) * uvec[2] * uvec[2] +
        2. * (G.gcov(Loci::center, j, i, 1, 2) * uvec[0] * uvec[1] +
              G.gcov(Loci::center, j, i, 1, 3) * uvec[0] * uvec[2] +
              G.gcov(Loci::center, j, i, 2, 3) * uvec[1] * uvec[2]);

        clip(vsq, 1.e-13, 1. - 1./(50.*50.));

        gamma = 1./sqrt(1. - vsq);

        VLOOP uvec[v] *= gamma;
        VLOOP P(u_start + v, k, j, i) = uvec[v];
    }
}

}
