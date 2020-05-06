/******************************************************************************
 *                                                                            *
 * PROBLEM.C                                                                  *
 *                                                                            *
 * INITIAL CONDITIONS FOR FISHBONE-MONCRIEF TORUS                             *
 *                                                                            *
 ******************************************************************************/
#pragma once

#include "decs.hpp"
#include "eos.hpp"
#include "prob_common.hpp"

#include <random>

// Functions to initialize and manipulate the solution
void InitializeFMTorus(MeshBlock *pmb, Grid& G, GridVars P, const EOS* eos,
                       GReal rin, GReal rmax, Real kappa);
void PerturbU(MeshBlock *pmb, GridVars P, Real u_jitter, int rng_seed);

// Device-side expressions for analytic torus
KOKKOS_INLINE_FUNCTION Real lfish_calc(const GReal a, const GReal r);
KOKKOS_INLINE_FUNCTION Real lnh_calc(const GReal a, const Real l, const GReal rin, const GReal r, const GReal th);

/**
 * Initialize a wide variety of different fishbone-moncrief torii.
 *
 * @param rin is the torus innermost radius, in r_g
 * @param rmax is the radius of maximum density of the F-M torus in r_g
 */
void InitializeFMTorus(MeshBlock *pmb, Grid& G, GridVars P, const EOS* eos,
                       GReal rin, GReal rmax, Real kappa=1.e-3)
{
    // Get coordinate system pointers for later
    // Only compatible with KS coords as base
    SphKSCoords ks = mpark::get<SphKSCoords>(G.coords->base);
    SphBLCoords bl = SphBLCoords(ks.a);

    // Fishbone-Moncrief parameters
    Real l = lfish_calc(ks.a, rmax);

    pmb->par_for("fm_torus_init", 0, pmb->ncells3-1, 0, pmb->ncells2-1, 0, pmb->ncells1-1,
        KOKKOS_LAMBDA_3D {
            GReal X[NDIM], Xembed[NDIM];
            G.coord(k, j, i, Loci::center, X);
            G.coord_embed(k, j, i, Loci::center, Xembed);
            GReal r = Xembed[1], th = Xembed[2];
            GReal sth = sin(th);
            GReal cth = cos(th);

            Real lnh = lnh_calc(ks.a, l, rin, r, th);

            // Region inside magnetized torus; u^i is calculated in
            // Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
            // so it needs to be transformed at the end
            // everything outside/else is initialized to 0 already
            if (lnh > 0. && r > rin) {
                GReal sth = sin(th);
                GReal cth = cos(th);

                Real r2 = pow(r, 2);
                Real a2 = pow(ks.a, 2);
                Real DD = r2 - 2. * r + a2;
                Real AA = pow(r2 + a2, 2) - DD * a2 * sth * sth;
                Real SS = r2 + a2 * cth * cth;

                // Calculate rho and u
                Real hm1 = exp(lnh) - 1.;
                Real rho = pow(hm1 * (eos->gam - 1.) / (kappa * eos->gam),
                                    1. / (eos->gam - 1.));
                Real u = kappa * pow(rho, eos->gam) / (eos->gam - 1.);

                // Calculate u^phi
                Real expm2chi = SS * SS * DD / (AA * AA * sth * sth);
                Real up1 = sqrt((-1. + sqrt(1. + 4. * l * l * expm2chi)) / 2.);
                Real up = 2. * ks.a * r * sqrt(1. + up1 * up1) / sqrt(AA * SS * DD) +
                            sqrt(SS / AA) * up1 / sth;

                // Convert u^phi from 3-velocity to 4-velocity
                Real ucon_bl[NDIM] = {0., 0, 0, up};
                Real gcov_bl[NDIM][NDIM];
                bl.gcov_embed(Xembed, gcov_bl);
                set_ut(gcov_bl, ucon_bl);

                // Then transform that 4-vector to KS, then to native
                Real ucon_ks[NDIM], ucon_mks[NDIM], u_prim[NDIM];
                ks.vec_from_bl(Xembed, ucon_bl, ucon_ks);
                G.coords->con_vec_to_native(X, ucon_ks, ucon_mks);

                // Convert native 4-vector to primitive u-twiddle, see Gammie '04
                Real gcon[NDIM][NDIM];
                G.gcon(Loci::center, j, i, gcon);
                fourvel_to_prim(gcon, ucon_mks, u_prim);

                P(prims::rho, k, j, i) = rho;
                P(prims::u, k, j, i) = u;
                P(prims::u1, k, j, i) = u_prim[1];
                P(prims::u2, k, j, i) = u_prim[2];
                P(prims::u3, k, j, i) = u_prim[3];
            }
        }
    );

    // Find rho_max "analytically" by looking over the whole mesh domain for the maximum in the midplane
    // Given xout is O(10-100) the given step should be pretty quick
    Real rho_max = 0, u_max = 0;

    GReal xin = log(rin);
    GReal xout = pmb->pmy_mesh->mesh_size.x1max;
    GReal x = xin, dx = 0.001;
    while (x < xout) {
        GReal r = exp(x);
        GReal th = M_PI/2;

        // Abbreviated version of the full primitives calculation
        Real lnh = lnh_calc(ks.a, l, rin, r, th);

        GReal sth = sin(th);
        GReal cth = cos(th);

        Real r2 = pow(r, 2);
        Real a2 = pow(ks.a, 2);
        Real DD = r2 - 2. * r + a2;
        Real AA = pow(r2 + a2, 2) - DD * a2 * sth * sth;
        Real SS = r2 + a2 * cth * cth;

        // Calculate rho
        Real hm1 = exp(lnh) - 1.;
        Real rho = pow(hm1 * (eos->gam - 1.) / (kappa * eos->gam),
                            1. / (eos->gam - 1.));
        Real u = kappa * pow(rho, eos->gam) / (eos->gam - 1.);

        // Record max
        if (rho > rho_max) rho_max = rho;
        if (u > u_max) u_max = u;

        x += dx;
    }

    pmb->par_for("fm_torus_normalize", 0, pmb->ncells3-1, 0, pmb->ncells2-1, 0, pmb->ncells1-1,
        KOKKOS_LAMBDA_3D {
            P(prims::rho, k, j, i) /= rho_max;
            P(prims::u, k, j, i) /= rho_max;
        }
    );
}

/**
 * Perturb the internal energy by a uniform random proportion per cell.
 * Resulting internal energies will be between u \pm u*u_jitter/2
 * i.e. u_jitter=0.1 -> \pm 5% randomization, 0.95u to 1.05u
 *
 * @param u_jitter see description
 * @param rng_seed is added to the MPI rank to seed the GSL RNG
 */
void PerturbU(MeshBlock *pmb, GridVars P, Real u_jitter, int rng_seed=31337)
{
    // Initialize RNG
    std::mt19937 gen(rng_seed);
    std::uniform_real_distribution<Real> dis(-u_jitter/2, u_jitter/2);

    for(int k=0; k < pmb->ncells3; k++)
        for(int j=0; j < pmb->ncells2; j++)
            for(int i=0; i < pmb->ncells3; i++)
                P(prims::u, k, j, i) *= 1. + u_jitter * dis(gen);

    // TODO option for parallel
    // pmb->par_for("perturb_u", 0, pmb->ncells3-1, 0, pmb->ncells2-1, 0, pmb->ncells1-1,
    //     KOKKOS_LAMBDA_3D {
    //         P(prims::u, k, j, i) *= 1. + u_jitter * dis(gen);
    //     }
    // );
}

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
