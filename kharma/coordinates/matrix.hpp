/* 
 *  File: matrix.hpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *  
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include "decs.hpp"

/**
 * Reasonably fast functions for handling 4x4 matrices and symbols
 * NOT GR AWARE
 * 
 * TODO
 * * Take matrix refs instead of ponters
 * * template over matrix types float vs double
 * * Make actual use of decs.hpp inclusion here: GR_DIM, DLOOP, etc
 */

/**
 * Dot, avoid for loops. NOT GR AWARE
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

/**
 * Matrix inversion. Call with pointers, i.e. &matrix_name[0][0]
 */
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
 * Due to Norm Hardy; good for general n
 */
template<int n>
KOKKOS_INLINE_FUNCTION int pp(int P[n])
{
  int x;
  int p = 0;
  int v[n] = {0};

  for (int j = 0; j < n; j++) {
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
  // Entries different? 
  if (a == b) return 0;
  if (a == c) return 0;
  if (a == d) return 0;
  if (b == c) return 0;
  if (b == d) return 0;
  if (c == d) return 0;

  // Determine parity of permutation
  int p[4] = {a, b, c, d};

  return pp<4>(p);
}

KOKKOS_INLINE_FUNCTION int antisym(int a, int b, int c)
{
  // Entries different? 
  if (a == b) return 0;
  if (a == c) return 0;
  if (b == c) return 0;

  // Determine parity of permutation
  int p[3] = {a, b, c};

  return pp<3>(p);
}

/**
 * Multiply two 4x4 matrices.
*/
KOKKOS_INLINE_FUNCTION void matrix_multiply(const GReal A[GR_DIM][GR_DIM], const GReal B[GR_DIM][GR_DIM], GReal result[GR_DIM][GR_DIM])
{
  for (int i = 0; i < GR_DIM; ++i) {
    for (int j = 0; j < GR_DIM; ++j) {
      for (int k = 0; k < GR_DIM; ++k) {
          result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

/**
 * Compute matrix exponential (4x4) to second order.
 * Needed to apply similarity transformation to modified gravity metrics
*/
KOKKOS_INLINE_FUNCTION void exp_taylor_series_second_order(const GReal M[GR_DIM][GR_DIM],  GReal expM[GR_DIM][GR_DIM])
{
    // Identity matrix
    double I[GR_DIM][GR_DIM] = { {1, 0, 0, 0},
                       {0, 1, 0, 0},
                       {0, 0, 1, 0},
                       {0, 0, 0, 1} };

    // Compute M^2
    // Matrix to hold M^2
    double M2[GR_DIM][GR_DIM] = {0};
    // Perform matrix multiplication
    matrix_multiply(M, M, M2);

    // expM = I + M + (1/2) * M^2
    for (int i = 0; i < GR_DIM; ++i) {
        for (int j = 0; j < GR_DIM; ++j) {
            expM[i][j] = I[i][j] + M[i][j] + 0.5 * M2[i][j];
        }
    }
}

/**
 * Compute matrix exponential (4x4) to fourth order.
 * Needed to apply similarity transformation to modified gravity metrics
*/
KOKKOS_INLINE_FUNCTION void exp_taylor_series_fourth_order(const GReal M[GR_DIM][GR_DIM],  GReal expM[GR_DIM][GR_DIM])
{
    // Identity matrix
    double I[GR_DIM][GR_DIM] = { {1, 0, 0, 0},
                       {0, 1, 0, 0},
                       {0, 0, 1, 0},
                       {0, 0, 0, 1} };

    // Compute M^2
    // Matrices to hold M^2, M^3, and M^4
    double M2[GR_DIM][GR_DIM] = {0};
    double M3[GR_DIM][GR_DIM] = {0};
    double M4[GR_DIM][GR_DIM] = {0};
    // Perform matrix multiplication
    matrix_multiply(M, M, M2);
    matrix_multiply(M2, M, M3);
    matrix_multiply(M3, M, M4);

    // expM = I + M + (1/2) * M^2
    for (int i = 0; i < GR_DIM; ++i) {
        for (int j = 0; j < GR_DIM; ++j) {
            expM[i][j] = I[i][j] + M[i][j] + ((1./2.) * M2[i][j]) + ((1./6.) * M3[i][j]) + ((1./24.) * M4[i][j]);
        }
    }
}

/**
 * Compute transpose of a 4x4 matrix
*/
KOKKOS_INLINE_FUNCTION void transpose_matrix(const double X[4][4], double transX[4][4]) {
    for (int i = 0; i < GR_DIM; ++i) {
        for (int j = 0; j < GR_DIM; ++j) {
            transX[j][i] = X[i][j];
        }
    }
}