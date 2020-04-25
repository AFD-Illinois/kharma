// Functions for boundaries not provided by Parthenon

#include "decs.hpp"

#include "bondi.hpp"
#include "phys.hpp"

TaskStatus ApplyCustomBoundaries(Container<Real>& rc)
{
    MeshBlock *pmb = rc.pmy_block;
    GridVars U = rc.Get("c.c.bulk.cons").data;

    // TODO TODO only if mesh is the last in X1...
    // TODO inflow check?
    if (pmb->packages["GRMHD"]->Param<std::string>("problem") == "bondi") {
        FLAG("Applying Bondi problem boundary");
        ApplyBondiBoundary(rc);
    }

    return TaskStatus::complete;
}

/**
 * Check for flow into simulation and reset velocity to eliminate it
 * TODO does Parthenon do something like this for outflow bounds already?
 *
 * @param type: 0 to check outflow from EH, 1 to check inflow from outer edge
 */
KOKKOS_INLINE_FUNCTION void check_inflow(Grid &G, GridVars P, const int& k, const int& j, const int& i, int type)
{
    Real ucon[NDIM];
    ucon_calc(G, P, k, j, i, Loci::center, ucon);

    if (((ucon[1] > 0.) && (type == 0)) ||
        ((ucon[1] < 0.) && (type == 1)))
    {
        // Find gamma and remove it from primitive velocity
        // TODO check failures?
        double gamma = mhd_gamma_calc(G, P, k, j, i, Loci::center);
        P(prims::u1, k, j, i) /= gamma;
        P(prims::u2, k, j, i) /= gamma;
        P(prims::u3, k, j, i) /= gamma;

        // Reset radial velocity so radial 4-velocity is zero
        Real alpha = 1 / G.gcon(Loci::center, j, i, 0, 0);
        Real beta1 = G.gcon(Loci::center, j, i, 0, 1) * alpha * alpha;
        P(prims::u1, k, j, i) = beta1 / alpha;

        // Now find new gamma and put it back in
        gamma = mhd_gamma_calc(G, P, k, j, i, Loci::center);

        P(prims::u1, k, j, i) *= gamma;
        P(prims::u2, k, j, i) *= gamma;
        P(prims::u3, k, j, i) *= gamma;
    }
}

/**
 * Fix fluxes on domain boundaries. No inflow, correct B fields on reflecting conditions.
 * TODO I bet strongly that Parthenon does this, if given to understand B is a vector
 */
void FixFlux(Container<Real>& rc)
{
    MeshBlock *pmb = rc.pmy_block;
    GridVars F1 = rc.Get("c.c.bulk.cons").flux[0];
    GridVars F2 = rc.Get("c.c.bulk.cons").flux[1];
    GridVars F3 = rc.Get("c.c.bulk.cons").flux[2];

    int NG = pmb->cnghost, N1 = pmb->ncells1, N2 = pmb->ncells2;

    // TODO check mesh block location
    //if (first in X1 and now inflow X1L)
    {
        pmb->par_for("fix_flux_in_l", pmb->ks, pmb->ke, pmb->js, pmb->je,
            KOKKOS_LAMBDA (const int& k, const int& j) {
                F1(prims::rho, k, j, 0 + NG) = min(F1(prims::rho, k, j, 0 + NG), 0.);
            }
        );
    }

    //if (last in X1 and no inflow X1R)
    {
        pmb->par_for("fix_flux_in_r", pmb->ks, pmb->ke, pmb->js, pmb->je,
            KOKKOS_LAMBDA (const int& k, const int& j) {
                F1(prims::rho, k, j, N1 + NG) = min(F1(prims::rho, k, j, N1 + NG), 0.);
            }
        );
    }

    //if (first in X2)
    {
        pmb->par_for("fix_flux_b_l", pmb->ks, pmb->ke, pmb->is, pmb->ie,
            KOKKOS_LAMBDA (const int& k, const int& i) {
                F1(prims::B2, k, -1 + NG, i) = -F1(prims::B2, k, 0 + NG, i);
                F3(prims::B2, k, -1 + NG, i) = -F3(prims::B2, k, 0 + NG, i);
                PLOOP F2(p, i, 0 + NG, k) = 0.;
            }
        );
    }

    //if (last in X2)
    {
        pmb->par_for("fix_flux_b_r", pmb->ks, pmb->ke, pmb->is, pmb->ie,
            KOKKOS_LAMBDA (const int& k, const int& i) {
                F1(prims::B2, k, N2 + NG, i) = -F1(prims::B2, k, N2 - 1 + NG, i);
                F3(prims::B2, k, N2 + NG, i) = -F3(prims::B2, k, N2 - 1 + NG, i);
                PLOOP F2(p, k, N2 + NG, i) = 0.;
            }
        );
    }
}