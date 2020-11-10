// Fishbone-Moncrief torus initialization functions
#pragma once

#include "decs.hpp"
#include <parthenon/parthenon.hpp>

#include "eos.hpp"

/**
 * Initialize a wide variety of different fishbone-moncrief torii.
 *
 * @param rin is the torus innermost radius, in r_g
 * @param rmax is the radius of maximum density of the F-M torus in r_g
 */
void InitializeFMTorus(std::shared_ptr<MeshBlock> pmb, const GRCoordinates& G, GridVars P, const EOS* eos,
                       GReal rin, GReal rmax, Real kappa=1.e-3);
/**
 * Perturb the internal energy by a uniform random proportion per cell.
 * Resulting internal energies will be between u \pm u*u_jitter/2
 * i.e. u_jitter=0.1 -> \pm 5% randomization, 0.95u to 1.05u
 *
 * @param u_jitter see description
 * @param rng_seed is added to the MPI rank to seed the GSL RNG
 */
void PerturbU(std::shared_ptr<MeshBlock> pmb, GridVars P, Real u_jitter, int rng_seed);

// Device-side expressions for FM variables
KOKKOS_INLINE_FUNCTION Real lnh_calc(const GReal a, const Real l, const GReal rin, const GReal r, const GReal th)
{
    Real sth = sin(th);
    Real cth = cos(th);

    Real r2 = pow(r, 2);
    Real a2 = pow(a, 2);
    Real DD = r2 - 2. * r + a2;
    Real AA = pow(r2 + a2, 2) - DD * a2 * sth * sth;
    Real SS = r2 + a2 * cth * cth;

    Real thin = M_PI / 2.;
    Real sthin = sin(thin);
    Real cthin = cos(thin);

    Real rin2 = pow(rin, 2);
    Real DDin = rin2 - 2. * rin + a2;
    Real AAin = pow(rin2 + a2, 2) - DDin * a2 * sthin * sthin;
    Real SSin = rin2 + a2 * cthin * cthin;

    if (r >= rin)
    {
        return
            0.5 *
                log((1. +
                        sqrt(1. +
                            4. * (l * l * SS * SS) * DD / (AA * AA * sth * sth))) /
                    (SS * DD / AA)) -
            0.5 * sqrt(1. +
                        4. * (l * l * SS * SS) * DD /
                            (AA * AA * sth * sth)) -
            2. * a * r * l / AA -
            (0.5 *
                    log((1. +
                        sqrt(1. +
                            4. * (l * l * SSin * SSin) * DDin /
                                (AAin * AAin * sthin * sthin))) /
                        (SSin * DDin / AAin)) -
                0.5 * sqrt(1. +
                        4. * (l * l * SSin * SSin) * DDin / (AAin * AAin * sthin * sthin)) -
                2. * a * rin * l / AAin);
    }
    else
    {
        return 1.;
    }
}

KOKKOS_INLINE_FUNCTION Real lfish_calc(const GReal a, const GReal r)
{
    return (((pow(a, 2) - 2. * a * sqrt(r) + pow(r, 2)) *
             ((-2. * a * r *
               (pow(a, 2) - 2. * a * sqrt(r) +
                pow(r,
                    2))) /
                  sqrt(2. * a * sqrt(r) + (-3. + r) * r) +
              ((a + (-2. + r) * sqrt(r)) * (pow(r, 3) + pow(a, 2) *
                                                            (2. + r))) /
                  sqrt(1 + (2. * a) / pow(r, 1.5) - 3. / r))) /
            (pow(r, 3) * sqrt(2. * a * sqrt(r) + (-3. + r) * r) *
             (pow(a, 2) + (-2. + r) * r)));
}
