/* 
 *  File: eos.hpp
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
 * Class representing an equation of state.  Currently still very tied to ideal EOSs
 */
class EOS {
    // TODO when gam does not need to be public, we are ready for new eqns of state
    public:
        Real gam;
        KOKKOS_FUNCTION EOS(Real gamma): gam(gamma) {};

        KOKKOS_FUNCTION virtual Real p(Real rho, Real u) const = 0;
        // Special version of above for Mignone & McKinney variable inversion w/ consolidated state variable w
        KOKKOS_FUNCTION virtual Real p_w(Real rho, Real w) const = 0;
};

class GammaLaw : public EOS {
    public:
        KOKKOS_FUNCTION GammaLaw(Real gamma): EOS(gamma) {}

        KOKKOS_FUNCTION Real p(Real rho, Real u)   const {return (gam - 1) * u;};
        KOKKOS_FUNCTION Real p_w(Real rho, Real w) const {return (w - rho) * (gam - 1) / gam;}
};

// Host functions for creating/deleting a device-side EOS function
// TODO move to a .cpp & a separate namespace
inline EOS* CreateEOS(Real gamma) {
        EOS *eos;
        eos = (EOS*)Kokkos::kokkos_malloc(sizeof(GammaLaw));
        Kokkos::parallel_for("CreateEOSObject", 1,
            KOKKOS_LAMBDA(const int&) {
                new ((GammaLaw*)eos) GammaLaw(gamma);
            }
        );
        return eos;
}
inline void DelEOS(EOS* eos) {
    Kokkos::parallel_for("DestroyEOSObject", 1,
        KOKKOS_LAMBDA(const int&) {
            eos->~EOS();
        }
    );
    Kokkos::kokkos_free(eos);
}