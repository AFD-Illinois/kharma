/**
 * Equation of state
 */

#include "decs.hpp"

class EOS {
    public:
        Real gam; // TODO eliminate this, and with it the last gamma-law dependence in HARM
        KOKKOS_INLINE_FUNCTION virtual Real p(Real u, Real rho) const {return 0.0;};
};

class Gammalaw : public EOS {
    public:
        Real gam;
        Gammalaw(Real gamma): gam(gamma) {};
        KOKKOS_INLINE_FUNCTION Real p(Real u, Real rho) const {return (gam - 1) * u;};
};