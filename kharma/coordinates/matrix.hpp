/**
 * Reasonably fast functions for handling 4x4 matrices and symbols
 */
#pragma once

#include "decs.hpp"

/**
 * NOT GR AWARE
 */
KOKKOS_INLINE_FUNCTION Real dot(const Real v1[GR_DIM], const Real v2[GR_DIM])
{
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3];
}

KOKKOS_INLINE_FUNCTION Real MINOR(const Real m[16], int r0, int r1, int r2, int c0, int c1, int c2)
{
  return m[4*r0+c0]*(m[4*r1+c1]*m[4*r2+c2] - m[4*r2+c1]*m[4*r1+c2]) -
         m[4*r0+c1]*(m[4*r1+c0]*m[4*r2+c2] - m[4*r2+c0]*m[4*r1+c2]) +
         m[4*r0+c2]*(m[4*r1+c0]*m[4*r2+c1] - m[4*r2+c0]*m[4*r1+c1]);
}

KOKKOS_INLINE_FUNCTION void adjoint(const Real m[16], Real adjOut[16])
{
  adjOut[ 0] =  MINOR(m,1,2,3,1,2,3);
  adjOut[ 1] = -MINOR(m,0,2,3,1,2,3);
  adjOut[ 2] =  MINOR(m,0,1,3,1,2,3);
  adjOut[ 3] = -MINOR(m,0,1,2,1,2,3);

  adjOut[ 4] = -MINOR(m,1,2,3,0,2,3);
  adjOut[ 5] =  MINOR(m,0,2,3,0,2,3);
  adjOut[ 6] = -MINOR(m,0,1,3,0,2,3);
  adjOut[ 7] =  MINOR(m,0,1,2,0,2,3);

  adjOut[ 8] =  MINOR(m,1,2,3,0,1,3);
  adjOut[ 9] = -MINOR(m,0,2,3,0,1,3);
  adjOut[10] =  MINOR(m,0,1,3,0,1,3);
  adjOut[11] = -MINOR(m,0,1,2,0,1,3);

  adjOut[12] = -MINOR(m,1,2,3,0,1,2);
  adjOut[13] =  MINOR(m,0,2,3,0,1,2);
  adjOut[14] = -MINOR(m,0,1,3,0,1,2);
  adjOut[15] =  MINOR(m,0,1,2,0,1,2);
}

KOKKOS_INLINE_FUNCTION Real determinant(const Real m[16])
{
  return m[0]*MINOR(m,1,2,3,1,2,3) -
         m[1]*MINOR(m,1,2,3,0,2,3) +
         m[2]*MINOR(m,1,2,3,0,1,3) -
         m[3]*MINOR(m,1,2,3,0,1,2);
}

KOKKOS_INLINE_FUNCTION Real invert(const Real *m, Real *invOut)
{
  adjoint(m, invOut);

  Real det = determinant(m);
  Real inv_det = 1. / det;
  for (int i = 0; i < 16; ++i) {
    invOut[i] *= inv_det;
  }

  return det;
}


/**
 * Parity calculation.
 * Due to Norm Hardy; in principle good for general n,
 * but in practice specified for speed/compiler
 */
KOKKOS_INLINE_FUNCTION int pp(int P[4])
{
  int x;
  int p = 0;
  int v[4];

  for (int j = 0; j < 4; j++) v[j] = 0;

  for (int j = 0; j < 4; j++) {
    if (v[j]) {
      p++;
    } else {
      x = j;
      do {
        x = P[x];
        v[x] = 1;
      } while (x != j);
    }
  }

  if (p % 2 == 0) {
    return 1;
  } else {
    return -1;
  }
}

// Completely antisymmetric 4D symbol
KOKKOS_INLINE_FUNCTION int antisym(int a, int b, int c, int d)
{
  // Check for valid permutation
  if (a < 0 || a > 3) return 100;
  if (b < 0 || b > 3) return 100;
  if (c < 0 || c > 3) return 100;
  if (d < 0 || d > 3) return 100;

  // Entries different? 
  if (a == b) return 0;
  if (a == c) return 0;
  if (a == d) return 0;
  if (b == c) return 0;
  if (b == d) return 0;
  if (c == d) return 0;

  // Determine parity of permutation
  int p[4] = {a, b, c, d};

  return pp(p);
}
