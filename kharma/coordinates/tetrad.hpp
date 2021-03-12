/* 
 *  File: tetrad.hpp
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
#include "matrix.hpp"

/** supporting routines **/

/**
 * @brief Convert vector K from coordinate to tetrad frame
 * Input and output vectors are contravariant (index up)
 */
KOKKOS_INLINE_FUNCTION void coordinate_to_tetrad(double Ecov[GR_DIM][GR_DIM], double K[GR_DIM],
                          double K_tetrad[GR_DIM])
{
    DLOOP1 K_tetrad[mu] = 0;
    DLOOP2 K_tetrad[mu] += Ecov[mu][nu] * K[nu];
}

/**
 * @brief Convert vector K from tetrad to coordinate frame
 * Input and output vectors are contravariant (index up)
 */
KOKKOS_INLINE_FUNCTION void tetrad_to_coordinate(double Econ[GR_DIM][GR_DIM], double K_tetrad[GR_DIM],
                          double K[GR_DIM])
{
    DLOOP1 K_tetrad[mu] = 0;
    DLOOP2 K_tetrad[mu] += Econ[nu][mu] * K[nu];
}

/**
 * Copy the trial vector into a tetrad basis vector,
 * checking to see if it is null, and if it is null
 * setting to some default value
 */
KOKKOS_INLINE_FUNCTION void set_Econ_from_trial(double Econ[GR_DIM], int defdir, double trial[GR_DIM])
{
    double norm = 0.;
    VLOOP norm += fabs(trial[v+1]);

    // Default to defdir if norm of trial vector is too small
    if (norm <= SMALL)
        DLOOP1 Econ[mu] = delta(mu, defdir);
    else
        DLOOP1 Econ[mu] = trial[mu];
}

/**
 * Check the handedness of a tetrad basis.
 * Basis is assumed to be in form e^\mu_{(a)} = Econ[a][mu]
 * levi_(ijkl) e0^i e1^j e2^k e3^l will be +1 if spatial
 * components are right-handed, -1 if left-handed.
 * experience suggests that roundoff produces errors of
 * order 10^{-12} in the result.
 */
KOKKOS_INLINE_FUNCTION int check_handedness(double Econ[GR_DIM][GR_DIM], double Gcov[GR_DIM][GR_DIM], double *dot)
{
    double Gcon[GR_DIM][GR_DIM];
    double g = invert(&Gcov[0][0], &Gcon[0][0]);
    if (g < 0.) return 1;

    /* check handedness */
    *dot = 0.;
    DLOOP4 *dot += antisym(mu, nu, lam, kap) * Econ[0][mu] * Econ[1][nu] * Econ[2][lam] * Econ[3][kap];
    *dot *= g;

    return 0;
}

/*
 * project out vconb from vcona
 * both arguments are index up (contravariant)
 * covariant metric is third argument.
 * overwrite the first argument on return
 */
KOKKOS_INLINE_FUNCTION void project_out(double vcona[GR_DIM], double vconb[GR_DIM], double Gcov[GR_DIM][GR_DIM])
{
    double vconb_sq = 0., adotb = 0.;
    DLOOP2 {
        vconb_sq += vconb[mu] * vconb[nu] * Gcov[mu][nu];
        adotb += vcona[mu] * vconb[nu] * Gcov[mu][nu];
    }
    DLOOP1 vcona[mu] -= vconb[mu] * adotb / vconb_sq;
}

/** tetrad making routines **/

/* 
 * econ/ecov index key:
 * Econ[k][l]
 * k: index attached to tetrad basis
 * index down
 * l: index attached to coordinate basis
 * index up
 * Ecov switches both indices
 *
 * make orthonormal basis for plasma frame.
 * e^0 along U
 * e^2 along b
 * e^3 along spatial part of K
 * 
 * Returns flag for whether the tetrad is suspicious.
 * Ideally ipole should crash on these errors but there are a lot of corner cases...
 */
KOKKOS_INLINE_FUNCTION int make_plasma_tetrad(double Ucon[GR_DIM], double Kcon[GR_DIM], double Bcon[GR_DIM],
                                              double Gcov[GR_DIM][GR_DIM], double Econ[GR_DIM][GR_DIM],
                                              double Ecov[GR_DIM][GR_DIM])
{
    // Start w/ time component parallel to U
    set_Econ_from_trial(Econ[0], 0, Ucon);
    normalize(Econ[0], Gcov);

    // Now use the trial vector in basis vector 3
    // We cast a suspicious eye on the trial vector...
    set_Econ_from_trial(Econ[3], 3, Kcon);
    // project out Econ[0]
    project_out(Econ[3], Econ[0], Gcov);
    normalize(Econ[3], Gcov);

    // Repeat for x2 unit basis vector
    set_Econ_from_trial(Econ[2], 2, Bcon);
    // Project out Econ[0],Econ[3]
    project_out(Econ[2], Econ[0], Gcov);
    project_out(Econ[2], Econ[3], Gcov);
    normalize(Econ[2], Gcov);

    // Whatever is left is Econ[1]
    // So start from any trial vector
    DLOOP1 Econ[1][mu] = 1.;
    // Project out Econ[0],Econ[3],Econ[2]
    project_out(Econ[1], Econ[0], Gcov);
    project_out(Econ[1], Econ[2], Gcov);
    project_out(Econ[1], Econ[3], Gcov);
    normalize(Econ[1], Gcov);

    // Check the resulting tetrad: handedness, orthogonality
    // TODO standard error codes/enum here
    int oddflag = 0;
    double dot = 0.;
    int hand_flag = check_handedness(Econ, Gcov, &dot);

    if (hand_flag)
        oddflag |= 16;

    // TODO may need to ease up here when using MKS3/eKS "exotic" coordinates
    if (fabs(fabs(dot) - 1.) > 1.e-10)
        oddflag |= 1;

    // We expect dot = 1 in a right-handed system
    // If not, flip Econ[1] to make system right-handed
    if (dot < 0.)
        DLOOP1 Econ[1][mu] *= -1.;

    // Lower each contravariant vector
    DLOOP1 flip_index(Econ[mu], Gcov, Ecov[mu]);

    // Flip Econ[0]
    DLOOP1 Ecov[0][mu] *= -1.;

    // TODO run check_ortho here under debug

    return oddflag;
}