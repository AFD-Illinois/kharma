/**
 * Equation of state
 */
#pragma once

#include "decs.hpp"

// TODO when gam does not need to be public, we are ready for new eqns of state
class EOS {
    public:
        Real gam;
        KOKKOS_FUNCTION EOS(Real gamma): gam(gamma) {};

        KOKKOS_FUNCTION virtual Real p(Real rho, Real u) const;
        // Special version of above for Mignone & McKinney variable inversion w/ consolidated state variable w
        KOKKOS_FUNCTION virtual Real p_w(Real rho, Real w) const;
};

class GammaLaw : public EOS {
    public:
        KOKKOS_FUNCTION GammaLaw(Real gamma): EOS(gamma) {}

        KOKKOS_FUNCTION Real p(Real rho, Real u) const {return (gam - 1) * u;};
        KOKKOS_FUNCTION Real p_w(Real rho, Real w) const {return (w - rho) * (gam - 1) / gam;}
};