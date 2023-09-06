/* 
 *  File: seed_B.hpp
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
#include "types.hpp"

TaskStatus SeedBField(MeshData<Real> *md, ParameterInput *pin);

TaskStatus NormalizeBField(MeshData<Real> *md, ParameterInput *pin);

// Internal representation of the field initialization preference for quick switch
// Avoids string comparsion in kernels
enum BSeedType{constant, monopole, monopole_cube, sane, mad, mad_quadrupole, r3s3, r5s5, gaussian, bz_monopole, vertical};

#define SEEDA_ARGS GReal *x, double rho, double rin, double min_A, double A0

// This will also act as the default implementation for unspecified types,
// which should all be filled as B field by seed_b below.
// So, we want to set it to something dramatic.
template<BSeedType T>
KOKKOS_INLINE_FUNCTION Real seed_a(SEEDA_ARGS) { return 0./0.;}

// EHT comparison SANE
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::sane>(SEEDA_ARGS)
{
    return m::max(rho - min_A, 0.);
}

// used in testing to exactly agree with harmpi
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::bz_monopole>(SEEDA_ARGS)
{
    return 1. - m::cos(x[2]);
}

// BR's smoothed poloidal in-torus, EHT standard MAD
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::mad>(SEEDA_ARGS)
{
    return m::max(m::pow(x[1] / rin, 3) * m::pow(sin(x[2]), 3) *
            m::exp(-x[1] / 400) * rho - min_A, 0.);
}

// MAD, but turned into a quadrupole
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::mad_quadrupole>(SEEDA_ARGS)
{
    return m::max(pow(x[1] / rin, 3) * m::pow(sin(x[2]), 3) *
            m::exp(-x[1] / 400) * rho - min_A, 0.) * m::cos(x[2]);
}

// Just the r^3 sin^3 th term
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::r3s3>(SEEDA_ARGS)
{
    return m::max(m::pow(x[1] / rin, 3) * m::pow(m::sin(x[2]), 3) * rho - min_A, 0.);
}

// Bump power to r^5 sin^5 th term, quieter MAD
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::r5s5>(SEEDA_ARGS)
{
    return m::max(m::pow(x[1] / rin, 5) * m::pow(m::sin(x[2]), 5) * rho - min_A, 0.);
}

// Pure vertical threaded field of gaussian strength with FWHM 2*rin (i.e. HM@rin)
// centered at BH center
// Block is to avoid compiler whinging about initialization
template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::gaussian>(SEEDA_ARGS)
{
    const Real xf = (x[1] / rin) * m::sin(x[2]);
    const Real sigma = 2 / m::sqrt(2 * m::log(2));
    const Real u = xf / m::abs(sigma);
    return (1 / (m::sqrt(2 * M_PI) * m::abs(sigma))) * m::exp(-u * u / 2);
}

template<>
KOKKOS_INLINE_FUNCTION Real seed_a<BSeedType::vertical>(SEEDA_ARGS)
{
    return A0 * x[1] * m::sin(x[2]) / 2.;
}

#define SEEDB_ARGS GReal *x, GReal gdet, double b10, double b20, double b30, double &B1, double &B2, double &B3

template<BSeedType T>
KOKKOS_INLINE_FUNCTION void seed_b(SEEDB_ARGS) {}

template<>
KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::constant>(SEEDB_ARGS)
{
    B1 = b10;
    B2 = b20;
    B3 = b30;
}

template<>
KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::monopole>(SEEDB_ARGS)
{
    B1 = b10 / gdet;
    B2 = 0.;
    B3 = 0.;
}

template<>
KOKKOS_INLINE_FUNCTION void seed_b<BSeedType::monopole_cube>(SEEDB_ARGS)
{
    B1 = 1 / (x[1]*x[1]*x[1]);
    B2 = 0.;
    B3 = 0.;
}
