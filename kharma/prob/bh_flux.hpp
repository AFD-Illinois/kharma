// Add flux to BH horizon
#pragma once

#include "decs.hpp"

// Add flux to BH horizon
// Applicable to any Kerr-space GRMHD sim, run after import/initialization
// Preserves divB==0 with Flux-CT step at end
void SeedBHFlux(MeshBlock *pmb, GRCoordinates G, GridVars P, Real BHflux)
{
    // This adds a central flux based on specifying some BHflux
    // Initialize a net magnetic field inside the initial torus
    // TODO do this only for BHflux > SMALL
    pmb->par_for("BHflux_A", 0, N2, 0, N1,
        KOKKOS_LAMBDA_2D {
            Real Xembed[GR_DIM];
            G.coord_embed(k, j, i, Loci::corner, Xembed);
            Real r = Xembed[1], th = Xembed[2];

            Real x = r * sin(th);
            Real z = r * cos(th);
            Real a_hyp = 20.;
            Real b_hyp = 60.;
            Real x_hyp = a_hyp * sqrt(1. + pow(z / b_hyp, 2));

            Real q = (pow(x, 2) - pow(x_hyp, 2)) / pow(x_hyp, 2);
            if (x < x_hyp) {
                A(j, i) = 10. * q;
            } else {
                A(j, i) = 0.;
            }
        }
    );

    // Evaluate net flux
    Real Phi_proc = 0.;
    pmb->par_for("BHflux_B2net", 5, N1 - 1, 0, N2 - 1,
        KOKKOS_LAMBDA_2D {
            // TODO coord, some distance to M_PI/2
            if (jglobal == N2TOT / 2)
            {
                Real Xembed[GR_DIM];
                G.coord(k, j, i, Loci::center, Xembed);
                Real r = Xembed[1], th = Xembed[2];

                if (r < rin)
                {
                    // Commented lines are unnecessary normalizations
                    Real B2net = (A(j, i) + A(j + 1, i) - A(j, i + 1) - A(j + 1, i + 1));
                    // / (2.*dx[1]*G.gdet(Loci::center, j, i));
                    Phi_proc += fabs(B2net) * M_PI / N3CPU;
                    // * 2.*dx[1]*G.gdet(Loci::center, j, i)
                }
            }
        }
    );

    // TODO ask if we're left bound in X1
    if (global_start[0] == 0)
    {
        // TODO probably not globally safe
        pmb->par_for("BHflux_B1net", 0, N2/2-1, 5+NG, 5+NG,
            KOKKOS_LAMBDA_2D {
                Real B1net = -(A(j, i) - A(j + 1, i) + A(j, i + 1) - A(j + 1, i + 1));
                // /(2.*dx[2]*G.gdet(Loci::center, j, i));
                Phi_proc += fabs(B1net) * M_PI / N3CPU;
                // * 2.*dx[2]*G.gdet(Loci::center, j, i)
            }
        );
    }
    Real Phi = mpi_reduce(Phi_proc); // TODO this also needs to be max over meshes!!

    norm = BHflux / (Phi + TINY_NUMBER);

    pmb->par_for("BHflux_B", 0, n3-1, 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA_3D {
            // Flux-ct
            P(prims::B1, k, j, i) += -norm * (A(j, i) - A(j + 1, i) + A(j, i + 1) - A(j + 1, i + 1)) /
                                        (2. * pmb->dx2v(j) * G.gdet(Loci::center, j, i));
            P(prims::B2, k, j, i) += norm * (A(j, i) + A(j + 1, i) - A(j, i + 1) - A(j + 1, i + 1)) /
                                        (2. * pmb->dx1v(i) * G.gdet(Loci::center, j, i));
        }
    );
}