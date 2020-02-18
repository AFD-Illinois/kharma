/*
 * self_init: Initialize the fluid with a set analytic distribution rather than a file
 */
#pragma once

#include "decs.hpp"
#include "grid.hpp"
#include "coordinates.hpp"

// Rootfinding with GSL for analytic solution (use?)
// #include <gsl/gsl_errno.h>
// #include <gsl/gsl_math.h>
// #include <gsl/gsl_roots.h>

using namespace std::literals::complex_literals;
using namespace std;

/**
 * Initialization for different analytic wave modes in magnetized plasma:
 * 0. Entropy, static mode
 * 1. Slow mode
 * 2. Alfven wave
 * 3. Fast mode
 */
GridVarsHost mhdmodes(Grid &G, const int nmode)
{
    // TODO check nprim >= 8
    // TODO init
    GridVarsHost p("prims_initial", G.n1, G.n2, G.n3, G.nvar);

    // Mean state
    Real rho0 = 1.;
    Real u0 = 1.;  // TODO set U{n} on the fly for boosted entropy test
    Real B10 = 0.; // This is set later, see below
    Real B20 = 0.;
    Real B30 = 0.;

    // Wavevector (TODO set on the fly)
    Real k1 = 2. * M_PI;
    Real k2 = 2. * M_PI;
    Real k3 = 2. * M_PI;
    // "Faux-2D" planar waves direction
    // Set to 0 for "full" 3D wave
    int dir = 0;
    if (dir == 1)
        k1 = 0;
    if (dir == 2)
        k2 = 0;
    if (dir == 3)
        k3 = 0;

    Real amp = 1.e-4;

    std::complex<Real> omega;
    Real drho, du, du1, du2, du3, dB1, dB2, dB3;

    // Eigenmode definitions
    if (dir == 0)
    {
        // 3D (1,1,1) wave
        B10 = 1.;
        if (nmode == 0)
        { // Entropy
            omega = 2. * M_PI / 5. * 1i;
            drho = 1.;
        }
        else if (nmode == 1)
        { // Slow
            omega = 2.35896379113i;
            drho = 0.556500332363;
            du = 0.742000443151;
            du1 = -0.282334999306;
            du2 = 0.0367010491491;
            du3 = 0.0367010491491;
            dB1 = -0.195509141461;
            dB2 = 0.0977545707307;
            dB3 = 0.0977545707307;
        }
        else if (nmode == 2)
        { // Alfven
            omega = -3.44144232573i;
            du2 = -0.339683110243;
            du3 = 0.339683110243;
            dB2 = 0.620173672946;
            dB3 = -0.620173672946;
        }
        else
        { // Fast
            omega = 6.92915162882i;
            drho = 0.481846076323;
            du = 0.642461435098;
            du1 = -0.0832240462505;
            du2 = -0.224080007379;
            du3 = -0.224080007379;
            dB1 = 0.406380545676;
            dB2 = -0.203190272838;
            dB3 = -0.203190272838;
        }
    }
    else
    {
        // 2D (1,1,0), (1,0,1), (0,1,1) wave
        // Constant field direction
        if (dir == 1)
        {
            B20 = 1.;
        }
        else if (dir == 2)
        {
            B30 = 1.;
        }
        else if (dir == 3)
        {
            B10 = 1.;
        }

        if (nmode == 0)
        { // Entropy
            omega = 2. * M_PI / 5. * 1i;
            drho = 1.;
        }
        else if (nmode == 1)
        { // Slow
            omega = 2.41024185339i;
            drho = 0.558104461559;
            du = 0.744139282078;
            if (dir == 1)
            {
                du2 = -0.277124827421;
                du3 = 0.0630348927707;
                dB2 = -0.164323721928;
                dB3 = 0.164323721928;
            }
            else if (dir == 2)
            {
                du3 = -0.277124827421;
                du1 = 0.0630348927707;
                dB3 = -0.164323721928;
                dB1 = 0.164323721928;
            }
            else if (dir == 3)
            {
                du1 = -0.277124827421;
                du2 = 0.0630348927707;
                dB1 = -0.164323721928;
                dB2 = 0.164323721928;
            }
        }
        else if (nmode == 2)
        { // Alfven
            omega = 3.44144232573i;
            if (dir == 1)
            {
                du1 = 0.480384461415;
                dB1 = 0.877058019307;
            }
            else if (dir == 2)
            {
                du2 = 0.480384461415;
                dB2 = 0.877058019307;
            }
            else if (dir == 3)
            {
                du3 = 0.480384461415;
                dB3 = 0.877058019307;
            }
        }
        else
        { // Fast
            omega = 5.53726217331i;
            drho = 0.476395427447;
            du = 0.635193903263;
            if (dir == 1)
            {
                du2 = -0.102965815319;
                du3 = -0.316873207561;
                dB2 = 0.359559114174;
                dB3 = -0.359559114174;
            }
            else if (dir == 2)
            {
                du3 = -0.102965815319;
                du1 = -0.316873207561;
                dB3 = 0.359559114174;
                dB1 = -0.359559114174;
            }
            else if (dir == 3)
            {
                du1 = -0.102965815319;
                du2 = -0.316873207561;
                dB1 = 0.359559114174;
                dB2 = -0.359559114174;
            }
        }
    }

    // Override tf and the dump and log intervals
    Real tf = 2. * M_PI / fabs(omega.imag());

    Kokkos::parallel_for("mhdmodes_init", G.h_bulk_0(),
                         KOKKOS_LAMBDA_3D {
                             GReal X[NDIM];
                             G.coord(i, j, k, Loci::center, X, false);

                             Real mode = amp * cos(k1 * X[1] + k2 * X[2] + k3 * X[3]);
                             p(i, j, k, prims::rho) = rho0 + drho * mode;
                             p(i, j, k, prims::u) = u0 + du * mode;
                             p(i, j, k, prims::u1) = du1 * mode;
                             p(i, j, k, prims::u2) = du2 * mode;
                             p(i, j, k, prims::u3) = du3 * mode;
                             p(i, j, k, prims::B1) = B10 + dB1 * mode;
                             p(i, j, k, prims::B2) = B20 + dB2 * mode;
                             p(i, j, k, prims::B3) = B30 + dB3 * mode;

#if DEBUG
                             if (i == 11 - 3 && j == 12 - 3 && k == 13 - 3)
                             {
                                 cerr << "Zone is %d %d %d\n", i, j, k);
                                 cerr << "Coord is %f %f %f\n", X[1], X[2], X[3]);
                                 cerr << "Starting prims are %f %f %f %f %f %f %f %f\n", p(i, j, k, prims::rho), p(i, j, k, prims::u),
                                        p(i, j, k, prims::u1), p(i, j, k, prims::u2), p(i, j, k, prims::u3), p(i, j, k, prims::B1),
                                        p(i, j, k, prims::B2), p(i, j, k, prims::B3));
                             }
#endif
                         }
    );

    return p;
}

// BONDI PROBLEM (TODO new file?)

// Cache some constants
Real C1, C2, n;

/**
 * Initialization of a Bondi problem with specified sonic point, BH mdot, and horizon radius
 * TODO this can/should be just mdot (and the grid ofc), if this problem is to be used as anything more than a test
 */
GridVarsHost bondi(Grid &G, const Real mdot, const Real rs)
{

    // TODO check nprim >= 8
    // TODO init
    GridVarsHost p("prims_initial", G.n1, G.n2, G.n3, G.nvar);



    Kokkos::parallel_for("bondi_init", G.h_bulk_0(),
                         KOKKOS_LAMBDA_3D {
                             get_prim_bondi(G, P, i, j, k);
                        }
    );

#if DEBUG
    // TODO print zone?
    cerr <<"a = %e Rhor = %e\n", a, Rhor);

    cerr <<"mdot = %e\n", mdot);
    cerr <<"rs   = %e\n", rs);
    cerr <<"n    = %e\n", n);
    cerr <<"C1   = %e\n", C1);
    cerr <<"C2   = %e\n", C2);
#endif

    return p;
}

// Adapted from M. Chandra
KOKKOS_INLINE_FUNCTION Real get_Tfunc(Real T, GReal r)
{
    return pow(1. + (1. + n) * T, 2.) * (1. - 2. / r + pow(C1 / r / r / pow(T, n), 2.)) - C2;
}

KOKKOS_INLINE_FUNCTION Real get_T(GReal r)
{
    Real rtol = 1.e-12;
    Real ftol = 1.e-14;
    Real Tmin = 0.6 * (sqrt(C2) - 1.) / (n + 1);
    Real Tmax = pow(C1 * sqrt(2. / r / r / r), 1. / n);

    Real f0, f1, fh;
    Real T0, T1, Th;
    T0 = Tmin;
    f0 = get_Tfunc(T0, r);
    T1 = Tmax;
    f1 = get_Tfunc(T1, r);

    Th = (f1 * T0 - f0 * T1) / (f1 - f0);
    fh = get_Tfunc(Th, r);
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
        fh = get_Tfunc(Th, r);
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
KOKKOS_INLINE_FUNCTION void set_ut(Real gcov[NDIM][NDIM], Real ucon[NDIM])
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

KOKKOS_INLINE_FUNCTION void get_prim_bondi(Grid& G, GridVars P, EOS* eos, int i, int j, int k)
{
    static int firstc = 1;
    if (firstc)
    {
        n = 1. / (eos->gam - 1.);

        // Solution constants
        Real uc = sqrt(mdot / (2. * rs));
        Real Vc = -sqrt(pow(uc, 2) / (1. - 3. * pow(uc, 2)));
        Real Tc = -n * pow(Vc, 2) / ((n + 1.) * (n * pow(Vc, 2) - 1.));
        C1 = uc * pow(rs, 2) * pow(Tc, n);
        C2 = pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + pow(C1, 2) / (pow(rs, 4) * pow(Tc, 2 * n)));

        firstc = 0;
    }

    G.ks_coord(i, j, k, &r, &th);

    while (r < Rhor)
    {
        i++;
        G.ks_coord(i, j, k, &r, &th);
    }

    Real T = get_T(r);
    Real ur = -C1 / (pow(T, n) * pow(r, 2));
    Real rho = pow(T, n);
    Real u = rho * T * n;

    // Convert ur to native coordinates
    Real ucon_bl[NDIM] = {0, ur, 0, 0};
    Real ucon_mks[NDIM], u_prim[NDIM];
    G.coords->vec_bl_to_native(X, ucon_bl, ucon_mks);

    // Convert native 4-vector to primitive u-twiddle, see Gammie '04
    Real gcon[NDIM][NDIM];
    G.coords->gcon_native(X, gcon);
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

KOKKOS_INLINE_FUNCTION void bound_gas_prob_x1r(Grid& , GridVars P, int i, int j, int k)
{
    get_prim_bondi(G, P, i, j, k);
}