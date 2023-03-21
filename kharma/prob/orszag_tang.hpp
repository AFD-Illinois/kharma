#pragma once

#include "decs.hpp"

using namespace parthenon;

/**
 * relativistic version of the Orszag-Tang vortex 
 * Orszag & Tang 1979, JFM 90, 129-143. 
 * original OT problem was incompressible 
 * this is based on compressible version given
 * in Toth 2000, JCP 161, 605.
 * 
 * in the limit tscale -> 0 the problem is identical
 * to the nonrelativistic problem; as tscale increases
 * the problem becomes increasingly relativistic
 * 
 * Stolen directly from iharm2d_v3
 */
TaskStatus InitializeOrszagTang(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    Flag(rc, "Initializing Orszag-Tang problem");
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real tscale = pin->GetOrAddReal("orszag_tang", "tscale", 0.05);
    // Default phase puts the current sheet in the middle of the domain
    const Real phase = pin->GetOrAddReal("orszag_tang", "phase", M_PI);

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("ot_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Real X[GR_DIM];
            G.coord(k, j, i, Loci::center, X);
            rho(k, j, i) = 25./9.;
            u(k, j, i) = 5./(3.*(gam - 1.));
            uvec(0, k, j, i) = -sin(X[2] + phase);
            uvec(1, k, j, i) = sin(X[1] + phase);
            uvec(2, k, j, i) = 0.;
            B_P(0, k, j, i) = -sin(X[2] + phase);
            B_P(1, k, j, i) = sin(2.*(X[1] + phase));
            B_P(2, k, j, i) = 0.;
        }
    );
    // Rescale primitive velocities & B field by tscale, and internal energy by the square.
    pmb->par_for("ot_renorm", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            u(k, j, i) *= tscale * tscale;
            VLOOP uvec(v, k, j, i) *= tscale;
            VLOOP B_P(v, k, j, i) *= tscale;
        }
    );

    return TaskStatus::complete;
}
