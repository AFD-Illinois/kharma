// Functions for boundaries not provided by Parthenon

#include "decs.hpp"

#include "boundaries.hpp"

#include "bondi.hpp"
#include "phys.hpp"

#include "basic_types.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

TaskStatus ApplyCustomBoundaries(std::shared_ptr<Container<Real>>& rc)
{
    MeshBlock *pmb = rc->pmy_block;
    GridVars U = rc->Get("c.c.bulk.cons").data;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // Implement our own reflecting boundary for our primitives
    if(pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect) {
        pmb->par_for("inner_x2_reflect", 0, NPRIM-1, kb.s, kb.e, 0, jb.s-1, 0, n1-1,
            KOKKOS_LAMBDA_VARS {
                Real reflect = ((p == prims::u2 || p == prims::B2) ? -1.0 : 1.0);
                U(p, k, j, i) = reflect * U(p, k, 2 * jb.s - j - 1, i);
            }
        );
    }
    if(pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect) {
        pmb->par_for("outer_x2_reflect", 0, NPRIM-1, kb.s, kb.e, jb.e+1, n2-1, 0, n1-1,
            KOKKOS_LAMBDA_VARS {
                Real reflect = ((p == prims::u2 || p == prims::B2) ? -1.0 : 1.0);
                U(p, k, j, i) = reflect * U(p, k, 2 * jb.e - j + 1, i);
            }
        );
    }

    // TODO check for inflow here too
    if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow &&
        pmb->packages["GRMHD"]->Param<std::string>("problem") == "bondi") {
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
KOKKOS_INLINE_FUNCTION void check_inflow(GRCoordinates &G, GridVars P, const int& k, const int& j, const int& i, int type)
{
    Real ucon[GR_DIM];
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
 * TODO Parthenon does this, if given to understand B is a vector
 */
TaskStatus FixFlux(std::shared_ptr<Container<Real>>& rc)
{
    MeshBlock *pmb = rc->pmy_block;
    GridVars F1 = rc->Get("c.c.bulk.cons").flux[X1DIR];
    GridVars F2 = rc->Get("c.c.bulk.cons").flux[X2DIR];
    GridVars F3 = rc->Get("c.c.bulk.cons").flux[X3DIR];

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    int NG = NGHOST;

    // TODO option to allow inflow?
    if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::outflow)
    {
        pmb->par_for("fix_flux_in_l", ks, ke, js, je,
            KOKKOS_LAMBDA (const int& k, const int& j) {
                F1(prims::rho, k, j, 0 + NG) = min(F1(prims::rho, k, j, 0 + NG), 0.);
            }
        );
    }

    if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::outflow &&
        !(pmb->packages["GRMHD"]->Param<std::string>("problem") == "bondi"))
    {
        pmb->par_for("fix_flux_in_r", ks, ke, js, je,
            KOKKOS_LAMBDA (const int& k, const int& j) {
                F1(prims::rho, k, j, ie + NG) = min(F1(prims::rho, k, j, ie + NG), 0.);
            }
        );
    }

    if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect)
    {
        pmb->par_for("fix_flux_b_l", ks, ke, is, ie,
            KOKKOS_LAMBDA (const int& k, const int& i) {
                F1(prims::B2, k, -1 + NG, i) = -F1(prims::B2, k, 0 + NG, i);
                F3(prims::B2, k, -1 + NG, i) = -F3(prims::B2, k, 0 + NG, i);
                PLOOP F2(p, k, 0 + NG, i) = 0.;
            }
        );
    }

    if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect)
    {
        pmb->par_for("fix_flux_b_r", ks, ke, is, ie,
            KOKKOS_LAMBDA (const int& k, const int& i) {
                F1(prims::B2, k, je + NG, i) = -F1(prims::B2, k, je - 1 + NG, i);
                F3(prims::B2, k, je + NG, i) = -F3(prims::B2, k, je - 1 + NG, i);
                PLOOP F2(p, k, je + NG, i) = 0.;
            }
        );
    }
    return TaskStatus::complete;
}