// Fishbone-Moncrief torus initialization functions
#pragma once

#include "decs.hpp"
#include "types.hpp"

/**
 * Initialize a wide variety of different fishbone-moncrief torii.
 *
 * @param rin is the torus innermost radius, in r_g
 * @param rmax is the radius of maximum density of the F-M torus in r_g
 */
TaskStatus InitializeFMTorus(MeshBlockData<Real> *rc, ParameterInput *pin);
/* Need a different initialization function since we have additional fields (q, dP)
 * for the EMHD problem that are declared at runtime*/
TaskStatus InitializeFMTorusEMHD(MeshBlockData<Real> *rc, ParameterInput *pin);
/**
 * Perturb the internal energy by a uniform random proportion per cell.
 * Resulting internal energies will be between u \pm u*u_jitter/2
 * i.e. u_jitter=0.1 -> \pm 5% randomization, 0.95u to 1.05u
 *
 * @param u_jitter see description
 * @param rng_seed is added to the MPI rank to seed the GSL RNG
 */
TaskStatus PerturbU(MeshBlockData<Real> *rc, ParameterInput *pin);

/**
 * Torus solution for ln h, See Fishbone and Moncrief eqn. 3.6. 
 */
KOKKOS_INLINE_FUNCTION Real lnh_calc(const GReal a, const Real l, const GReal rin, const GReal r, const GReal th)
{
    Real sth = sin(th);
    Real cth = cos(th);

    Real r2 = m::pow(r, 2);
    Real a2 = m::pow(a, 2);
    Real DD = r2 - 2. * r + a2;
    Real AA = m::pow(r2 + a2, 2) - DD * a2 * sth * sth;
    Real SS = r2 + a2 * cth * cth;

    Real thin = M_PI / 2.;
    Real sthin = sin(thin);
    Real cthin = cos(thin);

    Real rin2 = m::pow(rin, 2);
    Real DDin = rin2 - 2. * rin + a2;
    Real AAin = m::pow(rin2 + a2, 2) - DDin * a2 * sthin * sthin;
    Real SSin = rin2 + a2 * cthin * cthin;

    if (r >= rin) {
        return
            0.5 *
                log((1. +
                        m::sqrt(1. +
                            4. * (l * l * SS * SS) * DD / (AA * AA * sth * sth))) /
                    (SS * DD / AA)) -
            0.5 * m::sqrt(1. +
                        4. * (l * l * SS * SS) * DD /
                            (AA * AA * sth * sth)) -
            2. * a * r * l / AA -
            (0.5 *
                    log((1. +
                        m::sqrt(1. +
                            4. * (l * l * SSin * SSin) * DDin /
                                (AAin * AAin * sthin * sthin))) /
                        (SSin * DDin / AAin)) -
                0.5 * m::sqrt(1. +
                        4. * (l * l * SSin * SSin) * DDin / (AAin * AAin * sthin * sthin)) -
                2. * a * rin * l / AAin);
    } else {
        return 1.;
    }
}

/**
 * This function calculates specific the angular momentum of the
 * Fishbone-Moncrief solution in the midplane, as a function of radius.
 * (see Fishbone & Moncrief eqn. 3.8)
 * It improves on (3.8) by requiring no sign changes for
 * co-rotating (a > 0) vs counter-rotating (a < 0) disks.
 */
KOKKOS_INLINE_FUNCTION Real lfish_calc(const GReal a, const GReal r)
{
    return (((m::pow(a, 2) - 2. * a * m::sqrt(r) + m::pow(r, 2)) *
             ((-2. * a * r *
               (m::pow(a, 2) - 2. * a * m::sqrt(r) +
                m::pow(r,
                    2))) /
                  m::sqrt(2. * a * m::sqrt(r) + (-3. + r) * r) +
              ((a + (-2. + r) * m::sqrt(r)) * (m::pow(r, 3) + m::pow(a, 2) *
                                                            (2. + r))) /
                  m::sqrt(1 + (2. * a) / m::pow(r, 1.5) - 3. / r))) /
            (m::pow(r, 3) * m::sqrt(2. * a * m::sqrt(r) + (-3. + r) * r) *
             (m::pow(a, 2) + (-2. + r) * r)));
}

/**
 * Torus solution for density at a given location.
 * 
 * This function is *not* used for the actual initialization (where rho is calculated
 * alongside the other primitive variables).  Rather, it is for:
 * 1. Normalization, in which the max of this function over the domain is calculated.
 * 2. B field initialization, which requires density the untilted disk for simplicity
 */
KOKKOS_INLINE_FUNCTION Real fm_torus_rho(const GReal a, const GReal rin, const GReal rmax, const Real gam,
                                         const Real kappa, const GReal r, const GReal th)
{
    Real l = lfish_calc(a, rmax);
    Real lnh = lnh_calc(a, l, rin, r, th);
    if (lnh >= 0. && r >= rin) {
        // Calculate rho
        Real hm1 = exp(lnh) - 1.;
        return m::pow(hm1 * (gam - 1.) / (kappa * gam),
                            1. / (gam - 1.));
    } else {
        return 0;
    }
}
