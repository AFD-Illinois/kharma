// Common functions for problems:
// Make analytic 3-velocities into 4-velocities, and back

#include "decs.hpp"

/**
 * Set time component for a consistent 4-velocity given a 3-velocity
 */
KOKKOS_INLINE_FUNCTION void set_ut(const Real gcov[GR_DIM][GR_DIM], Real ucon[GR_DIM])
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
 * Make primitive velocities u-twiddle out of 4-velocity.  See Gammie '04
 */
KOKKOS_INLINE_FUNCTION void fourvel_to_prim(const Real gcon[GR_DIM][GR_DIM], const Real ucon[GR_DIM], Real u_prim[GR_DIM])
{
    Real alpha2 = -1.0 / gcon[0][0];
    // Note gamma/alpha is ucon[0]
    u_prim[1] = ucon[1] + ucon[0] * alpha2 * gcon[0][1];
    u_prim[2] = ucon[2] + ucon[0] * alpha2 * gcon[0][2];
    u_prim[3] = ucon[3] + ucon[0] * alpha2 * gcon[0][3];
}