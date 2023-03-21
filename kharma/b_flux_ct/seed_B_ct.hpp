// Seed a torus of some type with a magnetic field according to its density
#pragma once

#include "decs.hpp"
#include "types.hpp"

namespace B_FluxCT
{

/**
 * Seed an axisymmetric initialization with magnetic field proportional to fluid density,
 * or density and radius, to create a SANE or MAD flow
 * Note this function expects a normalized P for which rho_max==1
 *
 * @param rin is the interior radius of the torus
 * @param min_rho_q is the minimum density at which there will be magnetic vector potential
 * @param b_field_type is one of "sane" "ryan" "r3s3" or "gaussian", described below (TODO test or remove opts)
 */
TaskStatus SeedBField(MeshBlockData<Real> *rc, ParameterInput *pin);

/**
 * Add flux to BH horizon
 * Applicable to any Kerr-space GRMHD sim, run after import/initialization
 * Preserves divB==0 with a Flux-CT step at end
 */
//void SeedBHFlux(MeshBlockData<Real> *rc, Real BHflux);

} // namespace B_FluxCT


KOKKOS_INLINE_FUNCTION void get_B_from_A_3D(const GRCoordinates& G, const GridVector& A, const GridVector& B_U, const int& k, const int& j, const int& i)
{
                // Take a flux-ct step from the corner potentials.
                // This needs to be 3D because post-tilt A may not point in the phi direction only

                // A3,2 derivative
                const Real A3c2f = (A(V3, k, j + 1, i)     + A(V3, k, j + 1, i + 1) + 
                                    A(V3, k + 1, j + 1, i) + A(V3, k + 1, j + 1, i + 1)) / 4;
                const Real A3c2b = (A(V3, k, j, i)     + A(V3, k, j, i + 1) +
                                    A(V3, k + 1, j, i) + A(V3, k + 1, j, i + 1)) / 4;
                // A2,3 derivative
                const Real A2c3f = (A(V2, k + 1, j, i)     + A(V2, k + 1, j, i + 1) +
                                    A(V2, k + 1, j + 1, i) + A(V2, k + 1, j + 1, i + 1)) / 4;
                const Real A2c3b = (A(V2, k, j, i)     + A(V2, k, j, i + 1) +
                                    A(V2, k, j + 1, i) + A(V2, k, j + 1, i + 1)) / 4;
                B_U(V1, k, j, i) = (A3c2f - A3c2b) / G.dx2v(j) - (A2c3f - A2c3b) / G.dx3v(k);

                // A1,3 derivative
                const Real A1c3f = (A(V1, k + 1, j, i)     + A(V1, k + 1, j, i + 1) + 
                                    A(V1, k + 1, j + 1, i) + A(V1, k + 1, j + 1, i + 1)) / 4;
                const Real A1c3b = (A(V1, k, j, i)     + A(V1, k, j, i + 1) +
                                    A(V1, k, j + 1, i) + A(V1, k, j + 1, i + 1)) / 4;
                // A3,1 derivative
                const Real A3c1f = (A(V3, k, j, i + 1)     + A(V3, k + 1, j, i + 1) +
                                    A(V3, k, j + 1, i + 1) + A(V3, k + 1, j + 1, i + 1)) / 4;
                const Real A3c1b = (A(V3, k, j, i)     + A(V3, k + 1, j, i) +
                                    A(V3, k, j + 1, i) + A(V3, k + 1, j + 1, i)) / 4;
                B_U(V2, k, j, i) = (A1c3f - A1c3b) / G.dx3v(k) - (A3c1f - A3c1b) / G.dx1v(i);

                // A2,1 derivative
                const Real A2c1f = (A(V2, k, j, i + 1)     + A(V2, k, j + 1, i + 1) + 
                                    A(V2, k + 1, j, i + 1) + A(V2, k + 1, j + 1, i + 1)) / 4;
                const Real A2c1b = (A(V2, k, j, i)     + A(V2, k, j + 1, i) +
                                    A(V2, k + 1, j, i) + A(V2, k + 1, j + 1, i)) / 4;
                // A1,2 derivative
                const Real A1c2f = (A(V1, k, j + 1, i)     + A(V1, k, j + 1, i + 1) +
                                    A(V1, k + 1, j + 1, i) + A(V1, k + 1, j + 1, i + 1)) / 4;
                const Real A1c2b = (A(V1, k, j, i)     + A(V1, k, j, i + 1) +
                                    A(V1, k + 1, j, i) + A(V1, k + 1, j, i + 1)) / 4;
                B_U(V3, k, j, i) = (A2c1f - A2c1b) / G.dx1v(i) - (A1c2f - A1c2b) / G.dx2v(j);
}

KOKKOS_INLINE_FUNCTION void get_B_from_A_2D(const GRCoordinates& G, const GridVector& A, const GridVector& B_U, const int& k, const int& j, const int& i)
{
                // A3,2 derivative
                const Real A3c2f = (A(V3, k, j + 1, i) + A(V3, k, j + 1, i + 1)) / 2;
                const Real A3c2b = (A(V3, k, j, i)     + A(V3, k, j, i + 1)) / 2;
                B_U(V1, k, j, i) = (A3c2f - A3c2b) / G.dx2v(j);

                // A3,1 derivative
                const Real A3c1f = (A(V3, k, j, i + 1) + A(V3, k, j + 1, i + 1)) / 2;
                const Real A3c1b = (A(V3, k, j, i)     + A(V3, k, j + 1, i)) / 2;
                B_U(V2, k, j, i) = - (A3c1f - A3c1b) / G.dx1v(i);

                B_U(V3, k, j, i) = 0;
}
