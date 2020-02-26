// Source terms for equations of motion
#pragma once

#include "decs.hpp"
#include "phys.hpp"

// Note this is a single function for the state, so it is *not* INLINE_ETC
void get_fluid_source(const Grid &G, const GridVars P, const GridDerived D,
                      const EOS* eos, GridVars dU, bool wind=false)
{
    Kokkos::parallel_for("fluid_source", G.bulk_ng(),
                         KOKKOS_LAMBDA(const int i, const int j, const int k) {
                             Real mhd[NDIM][NDIM]; // Too much local memory?

                             DLOOP1 mhd_calc(P, D, eos, i, j, k, mu, mhd[mu]);

                             // Contract mhd stress tensor with connection
                             for (int p = 0; p < G.nvar; ++p)
                                 dU(i, j, k, p) = 0.;
                             DLOOP2
                             {
                                 // Put all 4 values into U(UU,U1,U2,U3)
                                 for (int lam = 0; lam < NDIM; lam++)
                                     dU(i, j, k, prims::u + lam) += mhd[mu][nu] * G.conn(i, j, nu, lam, mu);
                             }

                             for (int p = 0; p < G.nvar; ++p)
                                 dU(i, j, k, p) *= G.gdet(Loci::center, i, j);
                         });

    if (wind) {
        GridVars dP("dP", G.gn1, G.gn2, G.gn3, G.nvar);
        GridDerived dD;
        dD.ucon = GridVector("dD_ucon", G.gn1, G.gn2, G.gn3);
        dD.ucov = GridVector("dD_ucov", G.gn1, G.gn2, G.gn3);
        dD.bcon = GridVector("dD_bcon", G.gn1, G.gn2, G.gn3);
        dD.bcov = GridVector("dD_bcov", G.gn1, G.gn2, G.gn3);
        GridVars dUw("dUw", G.gn1, G.gn2, G.gn3, G.nvar);

        // Add a small "wind" source term in RHO,UU
        // Stolen shamelessly from iharm2d_v3
        Kokkos::parallel_for("fluid_source", G.bulk_ng(),
            KOKKOS_LAMBDA(const int i, const int j, const int k)
            {
                // TODO make these local

                // Need coordinates to evaluate particle addtn rate
                // Note that makes the wind spherical-only
                // TODO grab this after ensuring embedding coords are spherical
                GReal Xembed[NDIM];
                G.coord_embed(i, j, k, Loci::center, Xembed);
                GReal r = Xembed[1], th = Xembed[2];

                // Particle addition rate: concentrate at poles & center
                Real drhopdt = 2.e-4 * pow(cos(th), 4) / pow(1. + r * r, 2);

                dP(i, j, k, prims::rho) = drhopdt;

                Real Tp = 10.; // New fluid's temperature in units of c^2
                dP(i, j, k, prims::u) = drhopdt * Tp * 3.;

                // Leave everything else: we're inserting only fluid in normal observer frame

                // Add plasma to the T^t_a component of the stress-energy tensor
                // Notice that U already contains a factor of sqrt{-g}
                get_state(G, dP, i, j, k, Loci::center, dD);
                prim_to_flux(G, dP, dD, eos, i, j, k, Loci::center, 0, dUw);

                for (int p = 0; p < G.nvar; ++p)
                    dU(i, j, k, p) += dUw(i, j, k, p);
            }
        );
    }
}
template<typename DType>
KOKKOS_INLINE_FUNCTION void get_fluid_source(const Grid &G, const GridVars P, const DType D,
                      const EOS* eos, const int i, const int j, const int k, GridVars dU, bool wind=false)
{
    Real mhd[NDIM][NDIM];

    DLOOP1 mhd_calc(P, D, eos, i, j, k, mu, mhd[mu]);

    // Contract mhd stress tensor with connection
    DLOOP2
    {
        // Put all 4 values into U(UU,U1,U2,U3)
        for (int lam = 0; lam < NDIM; lam++)
            dU(i, j, k, prims::u + lam) += mhd[mu][nu] * G.conn(i, j, nu, lam, mu);
    }

    for (int p = 0; p < G.nvar; ++p)
        dU(i, j, k, p) *= G.gdet(Loci::center, i, j);

    if (wind) {
        // Need coordinates to evaluate particle addtn rate
        // Note that makes the wind spherical-only
        Real dP[8], dUw[8];
        Derived dD;
        // TODO grab this after ensuring embedding coords are spherical
        GReal Xembed[NDIM];
        G.coord_embed(i, j, k, Loci::center, Xembed);
        GReal r = Xembed[1], th = Xembed[2];

        // Particle addition rate: concentrate at poles & center
        Real drhopdt = 2.e-4 * pow(cos(th), 4) / pow(1. + r * r, 2);

        dP[prims::rho] = drhopdt;

        Real Tp = 10.; // New fluid's temperature in units of c^2
        dP[prims::u] = drhopdt * Tp * 3.;

        // Leave everything else: we're inserting only fluid in normal observer frame

        // Add plasma to the T^t_a component of the stress-energy tensor
        // Notice that U already contains a factor of sqrt{-g}
        get_state(G, dP, i, j, k, Loci::center, dD);
        prim_to_flux(G, dP, dD, eos, i, j, k, Loci::center, 0, dUw);

        for (int p = 0; p < G.nvar; ++p)
            dU(i, j, k, p) += dUw[p];
    }
}
template<typename DType>
KOKKOS_INLINE_FUNCTION void get_fluid_source(const Grid &G, const GridVars P, const DType D,
                      const EOS* eos, const int i, const int j, const int k, Real dU[], bool wind=false)
{
    Real mhd[NDIM][NDIM];

    DLOOP1 mhd_calc(P, D, eos, i, j, k, mu, mhd[mu]);

    // Contract mhd stress tensor with connection
    DLOOP2
    {
        // Put all 4 values into U(UU,U1,U2,U3)
        for (int lam = 0; lam < NDIM; lam++)
            dU[prims::u + lam] += mhd[mu][nu] * G.conn(i, j, nu, lam, mu);
    }

    for (int p = 0; p < G.nvar; ++p)
        dU[p] *= G.gdet(Loci::center, i, j);

    if (wind) {
        // Need coordinates to evaluate particle addtn rate
        // Note that makes the wind spherical-only
        Real dP[8], dUw[8];
        Derived dD;
        // TODO grab this after ensuring embedding coords are spherical
        GReal Xembed[NDIM];
        G.coord_embed(i, j, k, Loci::center, Xembed);
        GReal r = Xembed[1], th = Xembed[2];

        // Particle addition rate: concentrate at poles & center
        Real drhopdt = 2.e-4 * pow(cos(th), 4) / pow(1. + r * r, 2);

        dP[prims::rho] = drhopdt;

        Real Tp = 10.; // New fluid's temperature in units of c^2
        dP[prims::u] = drhopdt * Tp * 3.;

        // Leave everything else: we're inserting only fluid in normal observer frame

        // Add plasma to the T^t_a component of the stress-energy tensor
        // Notice that U already contains a factor of sqrt{-g}
        get_state(G, dP, i, j, k, Loci::center, dD);
        prim_to_flux(G, dP, dD, eos, i, j, k, Loci::center, 0, dUw);

        for (int p = 0; p < G.nvar; ++p)
            dU[p] += dUw[p];
    }
}