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
// TODO move to cpp
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