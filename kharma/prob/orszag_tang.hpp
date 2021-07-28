#pragma once

#include "decs.hpp"


using namespace std;
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
void InitializeOrszagTang(MeshBlock *pmb, const GRCoordinates& G, const GridVars& P, const GridVector& B_P, Real tscale=0.05)
{
    // Puts the current sheet in the middle of the domain
    Real phase = M_PI;

    Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    IndexDomain domain = IndexDomain::entire;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    pmb->par_for("ot_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord(k, j, i, Loci::center, X);
            P(prims::rho, k, j, i) = 25./9.;
            P(prims::u, k, j, i) = 5./(3.*(gam - 1.));
            P(prims::u1, k, j, i) = -sin(X[2] + phase);
            P(prims::u2, k, j, i) = sin(X[1] + phase);
            P(prims::u3, k, j, i) = 0.;
            B_P(0, k, j, i) = -sin(X[2] + phase);
            B_P(1, k, j, i) = sin(2.*(X[1] + phase));
            B_P(2, k, j, i) = 0.;
        }
    );
    // Rescale primitive velocities & B field by tscale, and internal energy by the square.
    pmb->par_for("ot_renorm", 0, NVEC-1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_VARS {
            P(prims::u1 + p, k, j, i) *= tscale;
            B_P(p, k, j, i) *= tscale;
        }
    ); 
    pmb->par_for("ot_renorm_u", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            P(prims::u, k, j, i) *= tscale * tscale;
        }
    );
}
