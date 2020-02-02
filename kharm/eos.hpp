/**
 * Equation of state
 */
#pragma once

#include "decs.hpp"

// For now, don't subclass.  Kokkos has problems with virtual functions

class EOS {
    public:
        Real gam;
        EOS(Real gamma): gam(gamma) {};
        KOKKOS_INLINE_FUNCTION virtual Real p(Real rho, Real u) const {return (gam - 1) * u;}
        // Special version of above for Mignone & McKinney variable inversion w/ consolidated state variable w
        KOKKOS_INLINE_FUNCTION virtual Real p_w(Real rho, Real w) const {return (w - rho) * (gam - 1) / gam;}
};

// class Gammalaw : public EOS {
//     public:
//         Gammalaw(Real gamma) {gam = gamma;};
//         KOKKOS_INLINE_FUNCTION Real p(Real rho, Real u) const {return (gam - 1) * u;};
//         KOKKOS_INLINE_FUNCTION Real p_w(Real rho, Real w) const {return (w - rho) * (gam - 1) / gam;}
// };