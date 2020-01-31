/**
 * Equation of state
 */
#pragma once

#include "decs.hpp"
class EOS {
    public:
        Real gam; // TODO eliminate this, and with it the last gamma-law dependence in HARM
        KOKKOS_INLINE_FUNCTION virtual Real p(Real rho, Real u) const {return 0.0;};
        // Special version of above for Mignone & McKinney variable inversion w/ consolidated state variable w
        KOKKOS_INLINE_FUNCTION Real p_w(Real rho, Real u) const {return 0.0;};
};

class Gammalaw : public EOS {
    public:
        Real gam;
        Gammalaw(Real gamma): gam(gamma) {};
        KOKKOS_INLINE_FUNCTION Real p(Real rho, Real u) const {return (gam - 1) * u;};
        KOKKOS_INLINE_FUNCTION Real p_w(Real rho, Real w) const {return (w - rho) * (gam - 1) / gam;};
};