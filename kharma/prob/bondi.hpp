/* 
 *  File: bondi.hpp
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

#include "gr_coordinates.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "coordinate_utils.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

/**
 * Initialize a Bondi problem over the domain
 */
TaskStatus InitializeBondi(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin);

/**
 * Set all values on a given domain to the Bondi inflow analytic steady-state solution.
 * Use the template version when possible, which just calls through
 */
TaskStatus SetBondiImpl(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse);

template<IndexDomain domain>
TaskStatus SetBondi(std::shared_ptr<MeshBlockData<Real>>& rc, bool coarse=false) {
    return SetBondiImpl(rc, domain, coarse);
}

/**
 * Supporting functions for Bondi flow calculations
 * 
 * Adapted from M. Chandra
 * Modified by Hyerin Cho and Ramesh Narayan
 */
KOKKOS_INLINE_FUNCTION Real get_Tfunc(const Real T, const GReal r, const Real C1, const Real C2, const Real n)
{
    const Real A = 1. + (1. + n) * T;
    const Real B = C1 / (r * r * m::pow(T, n));
    return A * A * (1. - 2. / r + B * B) - C2;
}

KOKKOS_INLINE_FUNCTION Real get_T(const GReal r, const Real C1, const Real C2, const Real n, const Real rs)
{
    Real rtol = 1.e-12;
    Real ftol = 1.e-14;
    Real Tinf = (m::sqrt(C2) - 1.) / (n + 1); // temperature at infinity
    Real Tnear = m::pow(C1 * m::sqrt(2. / (r*r*r)), 1. / n); // temperature near the BH

    // There are two branches of solutions (see Michel et al. 1971) and the two branches cross at rs.
    // These bounds are set to select the inflowing solution only.
    Real Tmin = (r < rs) ? Tinf  : m::max(Tnear,Tinf);
    Real Tmax = (r < rs) ? Tnear : 1.0;

    Real f0, f1, fh;
    Real T0, T1, Th;
    T0 = Tmin;
    f0 = get_Tfunc(T0, r, C1, C2, n);
    T1 = Tmax;
    f1 = get_Tfunc(T1, r, C1, C2, n);
    // TODO(BSP) where does this trigger an error?  Can we make it clearer?
    if (f0 * f1 > 0) return -1.;

    Th = (T0 + T1) / 2.; // a simple bisection method which is stable and fast
    fh = get_Tfunc(Th, r, C1, C2, n);
    Real epsT = rtol * (Tmin + Tmax);
    while (m::abs(Th - T0) > epsT && m::abs(Th - T1) > epsT && m::abs(fh) > ftol)
    {
        if (fh * f0 > 0.) {
            T0 = Th;
            f0 = fh;
        } else {
            T1 = Th;
            f1 = fh;
        }

        Th = (T0 + T1) / 2.; 
        fh = get_Tfunc(Th, r, C1, C2, n);
    }

    return Th;
}

KOKKOS_INLINE_FUNCTION void get_bondi_soln(const Real &r, const Real &rs, const Real &mdot, const Real &gam,
                                            Real &rho, Real &u, Real &ur)
{
    // Solution constants
    // These don't depend on which zone we're calculating
    const Real n = 1. / (gam - 1.);
    const Real uc = m::sqrt(1. / (2. * rs));
    const Real Vc = m::sqrt(uc * uc / (1. - 3. * uc * uc));
    const Real Tc = -n * Vc * Vc / ((n + 1.) * (n * Vc * Vc - 1.));
    const Real C1 = uc * rs * rs * m::pow(Tc, n);
    const Real A = 1. + (1. + n) * Tc;
    const Real C2 = A * A * (1. - 2. / rs + uc * uc);
    const Real K  = m::pow(4 * M_PI * C1 / mdot, 1/n);
    const Real Kn = m::pow(K, n);

    const Real T = get_T(r, C1, C2, n, rs);
    const Real Tn = m::pow(T, n);

    rho = Tn / Kn;
    u = rho * T * n;
    ur = -C1 / (Tn * r * r);
}
