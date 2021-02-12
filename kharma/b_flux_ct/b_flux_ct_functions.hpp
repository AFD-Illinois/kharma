/* 
 *  File: b_flux_ct_functions.hpp
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

#include "mhd_functions.hpp"

namespace B_FluxCT
{

/**
 * Convenience function as in GRMHD functions
 */
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const GridVector B_P,
                                    const int& k, const int& j, const int& i,
                                    GridVector B_flux, const Loci loc = Loci::center)
{
    VLOOP B_flux(v, k, j, i) = B_P(v, k, j, i) * G.gdet(loc, j, i);
}
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Real B_P[NVEC],
                                    const int& k, const int& j, const int& i,
                                    GridVector B_flux, const Loci loc = Loci::center)
{
    VLOOP B_flux(v, k, j, i) = B_P[v] * G.gdet(loc, j, i);
}
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const GridVector B_P,
                                    const int& k, const int& j, const int& i,
                                    Real B_flux[NVEC], const Loci loc = Loci::center)
{
    VLOOP B_flux[v] = B_P(v, k, j, i) * G.gdet(loc, j, i);
}
KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const Real B_P[NVEC],
                                    const int& k, const int& j, const int& i,
                                    Real B_flux[NVEC], const Loci loc = Loci::center)
{
    VLOOP B_flux[v] = B_P[v] * G.gdet(loc, j, i);
}

/**
 * Turn the primitive B field into the local conserved flux
 */
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const FourVectors D, const Real B_P[NVEC],
                                           const int& k, const int& j, const int& i, const Loci loc, const int dir,
                                           Real B_flux[NVEC])
{
    Real gdet = G.gdet(loc, j, i);
    if (dir == 0) {
        VLOOP B_flux[v] = B_P[v] * gdet;
    } else {
        // Dual of Maxwell tensor
        B_flux[0] = (D.bcon[1] * D.ucon[dir] -
                    D.bcon[dir] * D.ucon[1]) * gdet;
        B_flux[1] = (D.bcon[2] * D.ucon[dir] -
                    D.bcon[dir] * D.ucon[2]) * gdet;
        B_flux[2] = (D.bcon[3] * D.ucon[dir] -
                    D.bcon[dir] * D.ucon[3]) * gdet;
    }
}

} // namespace B_FluxCT