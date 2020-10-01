#pragma once

#include "decs.hpp"
#include "eos.hpp"

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
void InitializeOrszagTang(std::shared_ptr<MeshBlock> pmb, const GRCoordinates& G, const GridVars& P, Real tscale=0.05)
{
    // Puts the current sheet in the middle of the domain
    Real phase = M_PI;

    Real gam = pmb->packages["GRMHD"]->Param<Real>("gamma");

    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("ot_init", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord(k, j, i, Loci::center, X);
            P(prims::rho, k, j, i) = 25./9.;
            P(prims::u, k, j, i) = 5./(3.*(gam - 1.));
            P(prims::u1, k, j, i) = -sin(X[2] + phase);
            P(prims::u2, k, j, i) = sin(X[1] + phase);
            P(prims::u3, k, j, i) = 0.;
            P(prims::B1, k, j, i) = -sin(X[2] + phase);
            P(prims::B2, k, j, i) = sin(2.*(X[1] + phase));
            P(prims::B3, k, j, i) = 0.;
        }
    );
    // Rescale primitive velocities & B field by tscale, and internal energy by the square.
    pmb->par_for("ot_renorm", prims::u, NPRIM-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_VARS {
            P(p, k, j, i) *= tscale * (p == prims::u ? tscale : 1);
        }
    );
}
