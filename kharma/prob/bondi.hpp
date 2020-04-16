
// BONDI PROBLEM
#pragma once

#include "decs.hpp"
#include "grid.hpp"
#include "coordinates.hpp"
#include "eos.hpp"

using namespace std;

template<typename Vars, typename Range>
void get_prim_bondi(const Grid& G, Vars P, const EOS* eos, const Range range, const Real mdot, const Real rs);

/**
 * Initialization of a Bondi problem with specified sonic point, BH mdot, and horizon radius
 * TODO this can/should be just mdot (and the grid ofc), if this problem is to be used as anything more than a test
 */
GridVarsHost bondi(const Grid& G, const EOS* eos, const Real mdot, const Real rs)
{
#if DEBUG
    cerr << "mdot = " << mdot << endl;
    cerr << "rs   = " << rs << endl;
    // TODO print zone?
#endif

    // Initialize with ghost zones
    // get_prim_bondi will have to be device-side anyway for boundary conditions,
    // and our default coordinates object is there anyway. 
    GridVars p_d("prims_initial_all", G.gn1, G.gn2, G.gn3, G.nvar);
    cerr << "Initialized prims array" << endl;

    get_prim_bondi(G, p_d, eos, G.bulk_ng(), mdot, rs);
    cerr << "Initialized bondi on device" << endl;

    // Copy first to host
    auto p_all = Kokkos::create_mirror(p_d);
    Kokkos::deep_copy(p_all, p_d);
    cerr << "Copied to host" << endl;

    // Then subview and copy into final memory
    // TODO see if this or the offset back in main() is faster
    auto p_s = subview(p_all, Kokkos::make_pair(G.ng, G.ng + G.n1),
                                Kokkos::make_pair(G.ng, G.ng + G.n2),
                                Kokkos::make_pair(G.ng, G.ng + G.n3),
                                ALL());
    GridVarsHost p("prims_initial", G.n1, G.n2, G.n3, G.nvar);
    Kokkos::deep_copy(p, p_s); // Get rid of the stride, it's annoying
    cerr << "Copied on host" << endl;

    return p;
}

// Adapted from M. Chandra
KOKKOS_INLINE_FUNCTION Real get_Tfunc(const Real T, const GReal r, const Real C1, const Real C2, const Real n)
{
    return pow(1. + (1. + n) * T, 2.) * (1. - 2. / r + pow(C1 / r / r / pow(T, n), 2.)) - C2;
}

KOKKOS_INLINE_FUNCTION Real get_T(const GReal r, const Real C1, const Real C2, const Real n)
{
    Real rtol = 1.e-12;
    Real ftol = 1.e-14;
    Real Tmin = 0.6 * (sqrt(C2) - 1.) / (n + 1);
    Real Tmax = pow(C1 * sqrt(2. / r / r / r), 1. / n);

    Real f0, f1, fh;
    Real T0, T1, Th;
    T0 = Tmin;
    f0 = get_Tfunc(T0, r, C1, C2, n);
    T1 = Tmax;
    f1 = get_Tfunc(T1, r, C1, C2, n);
    if (f0 * f1 > 0) return -1;

    Th = (f1 * T0 - f0 * T1) / (f1 - f0);
    fh = get_Tfunc(Th, r, C1, C2, n);
    Real epsT = rtol * (Tmin + Tmax);
    while (fabs(Th - T0) > epsT && fabs(Th - T1) > epsT && fabs(fh) > ftol)
    {
        if (fh * f0 < 0.)
        {
            T0 = Th;
            f0 = fh;
        }
        else
        {
            T1 = Th;
            f1 = fh;
        }

        Th = (f1 * T0 - f0 * T1) / (f1 - f0);
        fh = get_Tfunc(Th, r, C1, C2, n);
    }

    return Th;
}

/**
 * Make primitive velocities out of 4-velocity.  See Gammie '04
 * Returns in the given 
 */
KOKKOS_INLINE_FUNCTION void fourvel_to_prim(const Real gcon[NDIM][NDIM], const Real ucon[NDIM], Real u_prim[NDIM])
{
    Real alpha2 = -1.0 / gcon[0][0];
    // Note gamma/alpha is ucon[0]
    u_prim[1] = ucon[1] + ucon[0] * alpha2 * gcon[0][1];
    u_prim[2] = ucon[2] + ucon[0] * alpha2 * gcon[0][2];
    u_prim[3] = ucon[3] + ucon[0] * alpha2 * gcon[0][3];
}

/**
 * Set time component for consistency given a 3-velocity
 */
KOKKOS_INLINE_FUNCTION void set_ut(const Real gcov[NDIM][NDIM], Real ucon[NDIM])
{
    Real AA, BB, CC;

    AA = gcov[0][0];
    BB = 2. * (gcov[0][1] * ucon[1] +
               gcov[0][2] * ucon[2] +
               gcov[0][3] * ucon[3]);
    CC = 1. + gcov[1][1] * ucon[1] * ucon[1] +
         gcov[2][2] * ucon[2] * ucon[2] +
         gcov[3][3] * ucon[3] * ucon[3] +
         2. * (gcov[1][2] * ucon[1] * ucon[2] +
               gcov[1][3] * ucon[1] * ucon[3] +
               gcov[2][3] * ucon[2] * ucon[3]);

    Real discr = BB * BB - 4. * AA * CC;
    ucon[0] = (-BB - sqrt(discr)) / (2. * AA);
}

/**
 * Get the Bondi solution at a particular zone.  Templated to be host- or device-side.
 * Note this assumes that there are ghost zones!
 */
template<typename Vars, typename Range>
void get_prim_bondi(const Grid& G, Vars P, const EOS* eos, const Range range, const Real mdot, const Real rs)
{
    CoordinateEmbedding cs = *(G.coords); // TODO best way to do that?
    Kokkos::parallel_for("bondi_calc", range,
        KOKKOS_LAMBDA_3D {
            // Solution constants
            // TODO cache these?  State is awful
            Real n = 1. / (eos->gam - 1.);
            Real uc = sqrt(mdot / (2. * rs));
            Real Vc = -sqrt(pow(uc, 2) / (1. - 3. * pow(uc, 2)));
            Real Tc = -n * pow(Vc, 2) / ((n + 1.) * (n * pow(Vc, 2) - 1.));
            Real C1 = uc * pow(rs, 2) * pow(Tc, n);
            Real C2 = pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + pow(C1, 2) / (pow(rs, 4) * pow(Tc, 2 * n)));

            // TODO this seems awkward.  Is this really the way to use my new tooling?
            // Note some awkwardness is the fact we need both native coords and embedding r
            GReal X[NDIM], Xembed[NDIM];
            G.coord(i, j, k, Loci::center, X);
            cs.coord_to_embed(X, Xembed);
            // Any zone inside the horizon gets the horizon's values
            int ii = i;
            while (Xembed[1] < mpark::get<SphKSCoords>(cs.base).rhor())
            {
                ++ii;
                G.coord(ii, j, k, Loci::center, X);
                cs.coord_to_embed(X, Xembed);
            }
            GReal r = Xembed[1];

            Real T = get_T(r, C1, C2, n);
            if (T < 0) T = 0; // If you can't error, NaN
            Real ur = -C1 / (pow(T, n) * pow(r, 2));
            Real rho = pow(T, n);
            Real u = rho * T * n;

            // Convert ur to native coordinates
            Real ucon_bl[NDIM] = {0, ur, 0, 0};
            Real ucon_ks[NDIM], ucon_mks[NDIM], u_prim[NDIM];
            // TODO if using KS etc etc.
            mpark::get<SphKSCoords>(cs.base).vec_from_bl(Xembed, ucon_bl, ucon_ks);
            cs.vec_to_native(X, ucon_ks, ucon_mks);

            // Convert native 4-vector to primitive u-twiddle, see Gammie '04
            Real gcon[NDIM][NDIM];
            cs.gcon_native(X, gcon);
            fourvel_to_prim(gcon, ucon_mks, u_prim);

            P(i, j, k, prims::rho) = rho;
            P(i, j, k, prims::u) = u;
            P(i, j, k, prims::u1) = u_prim[1];
            P(i, j, k, prims::u2) = u_prim[2];
            P(i, j, k, prims::u3) = u_prim[3];
            P(i, j, k, prims::B1) = 0.;
            P(i, j, k, prims::B2) = 0.;
            P(i, j, k, prims::B3) = 0.;
        }
    );
}