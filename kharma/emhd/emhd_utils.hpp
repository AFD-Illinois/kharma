/* 
 *  File: emhd.hpp
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
 * Utilities for the EMHD source terms, things we might conceivably use somewhere else,
 * or use *from* somewhere else instead of here.
 * 
 * 1. Slopes at faces using various linear reconstructions.  Since this is unrelated to
 *    reconstructing all prims, and only called zone-wise, the "same" recon algos are reimplemented here
 * 2. Calculate gradient of each component of ucov & 
 */

namespace EMHD {

// Linear MC slope limiter
KOKKOS_INLINE_FUNCTION Real linear_monotonized_cd(Real x1, Real x2, Real x3, Real dx)
{
    const Real Dqm = 2 * (x2 - x1) / dx;
    const Real Dqp = 2 * (x3 - x2) / dx;
    const Real Dqc = 0.5 * (x3 - x1) / dx;

    if (Dqm * Dqp <= 0) {
        return 0;
    } else {
        if ((m::abs(Dqm) < m::abs(Dqp)) && (fabs (Dqm) < m::abs(Dqc))) {
            return Dqm;
        } else if (m::abs(Dqp) < m::abs(Dqc)) {
            return Dqp;
        } else {
            return Dqc;
        }
    }
}

// Linear Van Leer slope limiter
KOKKOS_INLINE_FUNCTION Real linear_van_leer(Real x1, Real x2, Real x3, Real dx)
{
    const Real Dqm = (x2 - x1) / dx;
    const Real Dqp = (x3 - x2) / dx;

    const Real extrema = Dqm * Dqp;

    if (extrema <= 0) {
        return 0;
    } else {
        return (2 * extrema / (Dqm + Dqp)); 
    }
}

/**
 * Compute slope of scalars at faces
 */
template<typename Global>
KOKKOS_INLINE_FUNCTION Real slope_calc_scalar(const GRCoordinates& G, const Global& A, const int& dir,
                                              const int& b, const int& k, const int& j, const int& i, 
                                              ReconstructionType recon=ReconstructionType::linear_mc)
{
    // TODO could generic-ize this, but with two options, screw it
    if (recon != ReconstructionType::linear_vl) {
        if (dir == 1) return linear_monotonized_cd(A(b, k, j, i-1), A(b, k, j, i), A(b, k, j, i+1), G.dx1v(i));
        if (dir == 2) return linear_monotonized_cd(A(b, k, j-1, i), A(b, k, j, i), A(b, k, j+1, i), G.dx2v(j));
        if (dir == 3) return linear_monotonized_cd(A(b, k-1, j, i), A(b, k, j, i), A(b, k+1, j, i), G.dx3v(k));
    } else {
        if (dir == 1) return linear_van_leer(A(b, k, j, i-1), A(b, k, j, i), A(b, k, j, i+1), G.dx1v(i));
        if (dir == 2) return linear_van_leer(A(b, k, j-1, i), A(b, k, j, i), A(b, k, j+1, i), G.dx2v(j));
        if (dir == 3) return linear_van_leer(A(b, k-1, j, i), A(b, k, j, i), A(b, k+1, j, i), G.dx3v(k));
    }
    return 0.;
}

/**
 * Compute slope of all  vectors at faces
 */
template<typename Global>
KOKKOS_INLINE_FUNCTION Real slope_calc_vector(const GRCoordinates& G, const Global& A, const int& mu,
                                              const int& dir, const int& b, const int& k, const int& j, const int& i, 
                                              ReconstructionType recon=ReconstructionType::linear_mc)
{
    // TODO could generic-ize this, but with two options, screw it
    if (recon != ReconstructionType::linear_vl) {
        if (dir == 1) return linear_monotonized_cd(A(b, mu, k, j, i-1), A(b, mu, k, j, i), A(b, mu, k, j, i+1), G.dx1v(i));
        if (dir == 2) return linear_monotonized_cd(A(b, mu, k, j-1, i), A(b, mu, k, j, i), A(b, mu, k, j+1, i), G.dx2v(j));
        if (dir == 3) return linear_monotonized_cd(A(b, mu, k-1, j, i), A(b, mu, k, j, i), A(b, mu, k+1, j, i), G.dx3v(k));
    } else {
        if (dir == 1) return linear_van_leer(A(b, mu, k, j, i-1), A(b, mu, k, j, i), A(b, mu, k, j, i+1), G.dx1v(i));
        if (dir == 2) return linear_van_leer(A(b, mu, k, j-1, i), A(b, mu, k, j, i), A(b, mu, k, j+1, i), G.dx2v(j));
        if (dir == 3) return linear_van_leer(A(b, mu, k-1, j, i), A(b, mu, k, j, i), A(b, mu, k+1, j, i), G.dx3v(k));
    }
    return 0.;
}

// Compute gradient of four velocities and temperature
// Called by emhd_explicit_sources
template<typename Global>
KOKKOS_INLINE_FUNCTION void gradient_calc(const GRCoordinates& G, const Global& P,
                                          const GridVector& ucov_s, const GridScalar& theta_s,
                                          const int& b, const int& k, const int& j, const int& i, 
                                          const bool& do_3d, const bool& do_2d,
                                          Real grad_ucov[GR_DIM][GR_DIM], Real grad_Theta[GR_DIM])
{
    // Compute gradient of ucov
    DLOOP1 {
        grad_ucov[0][mu] = 0;

        // slope in direction nu of component mu
        grad_ucov[1][mu] = slope_calc_vector(G, ucov_s, mu, 1, b, k, j, i);
        if (do_2d) {
            grad_ucov[2][mu] = slope_calc_vector(G, ucov_s, mu, 2, b, k, j, i);
        } else {
            grad_ucov[2][mu] = 0.;
        }
        if (do_3d) {
            grad_ucov[3][mu] = slope_calc_vector(G, ucov_s, mu, 3, b, k, j, i);
        } else {
            grad_ucov[3][mu] = 0.;
        }
    }
    DLOOP3 grad_ucov[mu][nu] -= G.conn(j, i, lam, mu, nu) * ucov_s(lam, k, j, i);

    // Compute temperature gradient
    // Time derivative component is computed in time_derivative_sources
    grad_Theta[0] = 0;
    grad_Theta[1] = slope_calc_scalar(G, theta_s, 1, b, k, j, i);
    if (do_2d) {
        grad_Theta[2] = slope_calc_scalar(G, theta_s, 2, b, k, j, i);
    } else {
        grad_Theta[2] = 0.;
    } 
    if (do_3d) {
        grad_Theta[3] = slope_calc_scalar(G, theta_s, 3, b, k, j, i);
    } else {
        grad_Theta[3] = 0.;
    }
}

} // namespace EMHD
