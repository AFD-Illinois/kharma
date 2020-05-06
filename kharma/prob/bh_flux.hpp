// Add flux to BH horizon
#pragma once

#include "decs.hpp"

// Add flux to BH horizon
// Applicable to any Kerr-space GRMHD sim, run after import/initialization
// Preserves divB==0 with Flux-CT step at end
void torus(MeshBlock *pmb, Grid G, GridVars P, Real BHflux)
{
    // This adds a central flux based on specifying some BHflux
    // Initialize a net magnetic field inside the initial torus
    // TODO do this only for BHflux > SMALL
    ZSLOOP(0, 0, 0, N2, 0, N1)
    {
        Real X[NDIM];
        coord(i, j, k, CORN, X);
        Real r, th;
        bl_coord(X, &r, &th);

        A(i, j) = 0.;

        Real x = r * sin(th);
        Real z = r * cos(th);
        Real a_hyp = 20.;
        Real b_hyp = 60.;
        Real x_hyp = a_hyp * sqrt(1. + pow(z / b_hyp, 2));

        Real q = (pow(x, 2) - pow(x_hyp, 2)) / pow(x_hyp, 2);
        if (x < x_hyp)
        {
            A(i, j) = 10. * q;
        }
    }

    // Evaluate net flux
    Real Phi_proc = 0.;
    ISLOOP(5, N1 - 1)
    {
        JSLOOP(0, N2 - 1)
        {
            int jglobal = j - NG + global_start[1];
            //int j = N2/2+NG;
            int k = NG;
            if (jglobal == N2TOT / 2)
            {
                Real X[NDIM];
                coord(i, j, k, CENT, X);
                Real r, th;
                bl_coord(X, &r, &th);

                if (r < rin)
                {
                    Real B2net = (A(i, j) + A(i, j + 1] - A(i + 1, j) - A(i + 1, j + 1]);
                    // / (2.*dx[1]*G.gdet(Loci::center, j, i));
                    Phi_proc += fabs(B2net) * M_PI / N3CPU; // * 2.*dx[1]*G.gdet(Loci::center, j, i)
                }
            }
        }
    }

    //If left bound in X1.  Note different convention from bhlight!
    if (global_start[0] == 0)
    {
        JSLOOP(0, N2 / 2 - 1)
        {
            int i = 5 + NG;

            Real B1net = -(A(i, j) - A(i, j + 1] + A(i + 1, j) - A(i + 1, j + 1]); // /(2.*dx[2]*G.gdet(Loci::center, j, i));
            Phi_proc += fabs(B1net) * M_PI / N3CPU;                                  // * 2.*dx[2]*G.gdet(Loci::center, j, i)
        }
    }
    Real Phi = mpi_reduce(Phi_proc);

    norm = BHflux / (Phi + SMALL);

    ZLOOP
    {
        // Flux-ct
        P(prims::B1, k, j, i) += -norm * (A(i, j) - A(i, j + 1) + A(i + 1, j) - A(i + 1, j + 1)) / (2. * dx[2] * G.gdet(Loci::center, j, i));
        P(prims::B2, k, j, i) += norm * (A(i, j) + A(i, j + 1) - A(i + 1, j) - A(i + 1, j + 1)) / (2. * dx[1] * G.gdet(Loci::center, j, i));
    }
}