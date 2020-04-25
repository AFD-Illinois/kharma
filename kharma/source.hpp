// Source terms for equations of motion
#pragma once

#include "decs.hpp"
#include "phys.hpp"

KOKKOS_INLINE_FUNCTION void get_fluid_source(const Grid &G, const GridVars P, const FourVectors& D,
                      const EOS* eos, const int& k, const int& j, const int& i, Real dU[NPRIM], bool wind=false)
{
    // Get T^mu_nu
    Real mhd[NDIM][NDIM];
    DLOOP1 mhd_calc(P, D, eos, k, j, i, mu, mhd[mu]);

    // Initialize
    PLOOP dU[p] = 0.;

    // Contract mhd stress tensor with connection, and multiply by metric dterminant
    DLOOP3 dU[prims::u + lam] += mhd[mu][nu] * G.conn(j, i, nu, lam, mu);
    DLOOP1 dU[prims::u + mu] *= G.gdet(Loci::center, j, i);

    if (wind) {
        // Need coordinates to evaluate particle addtn rate
        // Note that makes the wind spherical-only
        Real dP[8], dUw[8];
        FourVectors dD;
        // TODO grab this after ensuring embedding coords are spherical
        GReal Xembed[NDIM];
        G.coord_embed(k, j, i, Loci::center, Xembed);
        GReal r = Xembed[1], th = Xembed[2];

        // Particle addition rate: concentrate at poles & center
        Real drhopdt = 2.e-4 * pow(cos(th), 4) / pow(1. + r * r, 2);

        dP[prims::rho] = drhopdt;

        Real Tp = 10.; // New fluid's temperature in units of c^2
        dP[prims::u] = drhopdt * Tp * 3.;

        // Leave everything else: we're inserting only fluid in normal observer frame

        // Add plasma to the T^t_a component of the stress-energy tensor
        // Notice that U already contains a factor of sqrt{-g}
        get_state(G, dP, k, j, i, Loci::center, dD);
        prim_to_flux(G, dP, dD, eos, k, j, i, Loci::center, 0, dUw);

        PLOOP dU[p] += dUw[p];
    }
}