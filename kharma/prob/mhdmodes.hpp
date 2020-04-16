/*
 * self_init: Initialize the fluid with a set analytic distribution rather than a file
 */
#pragma once

#include "decs.hpp"
#include "coordinates.hpp"
#include "eos.hpp"

using namespace std::literals::complex_literals;
using namespace std;
using namespace parthenon;

/**
 * Initialization for different analytic wave modes in magnetized plasma:
 * 0. Entropy, static mode
 * 1. Slow mode
 * 2. Alfven wave
 * 3. Fast mode
 * 
 * Note this assumes ideal EOS with gamma=4/3!!!
 */
GridVarsHost mhdmodes(MeshBlock *pmb, int nmode)
{
    // TODO check nprim >= 8
    // TODO init
    p = pmb->real_containers.Get("c.c.bulk.prims")

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

    pmb->par_for("mhdmodes_init", pmb->ks, pmb->ke, pmb->js, pmb->je, pmb->is, pmb->ie,
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
        }
    );
    // TODO TODO prim_to_flux this to set cons!!

    return TaskStatus::complete;
}