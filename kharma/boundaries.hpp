//

#include "decs.hpp"

#include "phys.hpp"

/**
 * Check for flow into simulation and reset velocity to eliminate it
 * TODO does Parthenon do something like this for outflow bounds already?
 *
 * @param type: 0 to check outflow from EH, 1 to check inflow from outer edge
 */
void CheckInflow(Grid &G, GridVars P, int i, int j, int k, int type)
{
    Real ucon[NDIM];
    ucon_calc(G, P, i, j, k, Loci::center, ucon);

    if (((ucon[1] > 0.) && (type == 0)) ||
        ((ucon[1] < 0.) && (type == 1)))
    {
        // Find gamma and remove it from primitive velocity
        // TODO check failures?
        double gamma = mhd_gamma_calc(G, P, i, j, k, Loci::center);
        P(prims::u1, i, j, k) /= gamma;
        P(prims::u2, i, j, k) /= gamma;
        P(prims::u3, i, j, k) /= gamma;

        // Reset radial velocity so radial 4-velocity is zero
        Real alpha = 1 / G.gcon(Loci::center, i, j, 0, 0);
        Real beta1 = G.gcon(Loci::center, i, j, 0, 1) * alpha * alpha;
        P(prims::u1, i, j, k) = beta1 / alpha;

        // Now find new gamma and put it back in
        gamma = mhd_gamma_calc(G, P, i, j, k, Loci::center);

        P(prims::u1, i, j, k) *= gamma;
        P(prims::u2, i, j, k) *= gamma;
        P(prims::u3, i, j, k) *= gamma;
    }
}

/**
 * Fix fluxes on domain boundaries. No inflow, correct B fields on reflecting conditions.
 * I bet strongly that Parthenon does this.
 */
void FixFlux(Container<Real>& rc, GridVars F1, GridVars F2, GridVars F3)
{
    MeshBlock *pmb = rc.pmy_block;
    int NG = pmb->cnghost, N1 = pmb->ncells1, N2 = pmb->ncells2;

    // TODO check mesh block location
    //if (first in X1 and now inflow X1L)
    {
        pmb->par_for("fix_flux_in_l", pmb->js, pmb->je, pmb->ks, pmb->ke,
            KOKKOS_LAMBDA (const int& j, const int& k) {
                F1(prims::rho, 0 + NG, j, k) = std::min(F1(prims::rho, 0 + NG, j, k), 0.);
            }
        );
    }

    //if (last in X1 and no inflow X1R)
    {
        pmb->par_for("fix_flux_in_r", pmb->js, pmb->je, pmb->ks, pmb->ke,
            KOKKOS_LAMBDA (const int& j, const int& k) {
                F1(prims::rho, N1 + NG, j, k) = std::min(F1(prims::rho, N1 + NG, j, k), 0.);
            }
        );
    }

    //if (first in X2)
    {
        pmb->par_for("fix_flux_b_l", pmb->is, pmb->ie, pmb->ks, pmb->ke,
            KOKKOS_LAMBDA (const int& i, const int& k) {
                F1(prims::B2, i, -1 + NG, k) = -F1(prims::B2, i, 0 + NG, k);
                F3(prims::B2, i, -1 + NG, k) = -F3(prims::B2, i, 0 + NG, k);
                PLOOP F2(p, i, 0 + NG, k) = 0.;
            }
        );
    }

    //if (last in X2)
    {
        pmb->par_for("fix_flux_b_r", pmb->is, pmb->ie, pmb->ks, pmb->ke,
            KOKKOS_LAMBDA (const int& i, const int& k) {
                F1(prims::B2, i, N2 + NG, k) = -F1(prims::B2, i, N2 - 1 + NG, k);
                F3(prims::B2, i, N2 + NG, k) = -F3(prims::B2, i, N2 - 1 + NG, k);
                PLOOP F2(p, i, N2 + NG, k) = 0.;
            }
        );
    }
}