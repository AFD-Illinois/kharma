/* 
 *  File: fm_torus.cpp
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

#include "fm_torus.hpp"

#include "mpi.hpp"
#include "prob_common.hpp"

#include <random>
#include "Kokkos_Random.hpp"

void InitializeFMTorus(MeshBlock *pmb, const GRCoordinates& G, GridVars P, const Real& gam,
                       GReal rin, GReal rmax, Real kappa)
{
    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // Get coordinate system pointers for later
    // Only compatible with KS coords as base (TODO BL for fun?)
    SphKSCoords ksc = mpark::get<SphKSCoords>(G.coords.base);
    SphBLCoords bl = SphBLCoords(ksc.a);

    // Fishbone-Moncrief parameters
    Real l = lfish_calc(ksc.a, rmax);

    if (pmb->packages.Get("GRMHD")->Param<int>("verbose") > 0) {
        // Example of pulling stuff host-side. Maybe make all initializations print stuff like this?
        double gam_host;
        Kokkos::Max<Real> gam_reducer(gam_host);
        pmb->par_reduce("fm_torus_init", 0, 0, KOKKOS_LAMBDA_1D_REDUCE { local_result = gam; }, gam_reducer);
        cout << "Initializing Fishbone-Moncrief torus:" << gam_host << endl;
        cout << "rin = " << rin << endl;
        cout << "rmax = " << rmax << endl;
        cout << "kappa = " << kappa << endl;
        cout << "fluid gamma = " << gam_host << endl;

    }

    pmb->par_for("fm_torus_init", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            GReal Xnative[GR_DIM], Xembed[GR_DIM];
            G.coord(k, j, i, Loci::center, Xnative);
            G.coord_embed(k, j, i, Loci::center, Xembed);
            GReal r = Xembed[1], th = Xembed[2];
            GReal sth = sin(th);
            GReal cth = cos(th);

            Real lnh = lnh_calc(ksc.a, l, rin, r, th);

            // Region inside magnetized torus; u^i is calculated in
            // Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
            // so it needs to be transformed at the end
            // everything outside is left 0 to be added by the floors
            if (lnh >= 0. && r >= rin) {
                Real r2 = r*r;
                Real a2 = ksc.a * ksc.a;
                Real DD = r2 - 2. * r + a2;
                Real AA = pow(r2 + a2, 2) - DD * a2 * sth * sth;
                Real SS = r2 + a2 * cth * cth;

                // Calculate rho and u
                Real hm1 = exp(lnh) - 1.;
                Real rho = pow(hm1 * (gam - 1.) / (kappa * gam),
                                    1. / (gam - 1.));
                Real u = kappa * pow(rho, gam) / (gam - 1.);

                // Calculate u^phi
                Real expm2chi = SS * SS * DD / (AA * AA * sth * sth);
                Real up1 = sqrt((-1. + sqrt(1. + 4. * l * l * expm2chi)) / 2.);
                Real up = 2. * ksc.a * r * sqrt(1. + up1 * up1) / sqrt(AA * SS * DD) +
                            sqrt(SS / AA) * up1 / sth;

                // Convert u^phi from 3-velocity to 4-velocity
                Real ucon_bl[GR_DIM] = {0., 0., 0., up};
                Real gcov_bl[GR_DIM][GR_DIM];
                bl.gcov_embed(Xembed, gcov_bl);
                set_ut(gcov_bl, ucon_bl);

                // Then transform that 4-vector to KS, then to native
                Real ucon_ks[GR_DIM], ucon_mks[GR_DIM];
                ksc.vec_from_bl(Xembed, ucon_bl, ucon_ks);
                G.coords.con_vec_to_native(Xnative, ucon_ks, ucon_mks);

                // Convert native 4-vector to primitive u-twiddle, see Gammie '04
                Real gcon[GR_DIM][GR_DIM], u_prim[GR_DIM];
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
    // Done device-side for speed (for large 2D meshes this may get bad) but may work fine in HostSpace
    // Note this covers the full domain from each rank: it doesn't need a grid so it's not a memory problem,
    // and an MPI synch as is done for beta_min would be a headache
    GReal x1min = pmb->pmy_mesh->mesh_size.x1min;
    GReal x1max = pmb->pmy_mesh->mesh_size.x1max;
    //GReal x2min = pmb->pmy_mesh->mesh_size.x2min;
    //GReal x2max = pmb->pmy_mesh->mesh_size.x2max;
    GReal dx = 0.001;
    int nx1 = (x1max - x1min) / dx;
    //int nx2 = (x2max - x2min) / dx;

    if (pmb->packages.Get("GRMHD")->Param<int>("verbose") > 0) {
        cout << "Calculating maximum density:" << endl;
        cout << "a = " << ksc.a << endl;
        cout << "dx = " << dx << endl;
        cout << "x1min->x1max: " << x1min << " " << x1max << endl;
        cout << "nx1 = " << nx1 << endl;
        //cout << "x2min->x2max: " << x2min << " " << x2max << endl;
        //cout << "nx2 = " << nx2 << endl;
    }

    Real rho_max = 0;
    Kokkos::Max<Real> max_reducer(rho_max);
    pmb->par_reduce("fm_torus_maxrho", 0, nx1,
        KOKKOS_LAMBDA_1D_REDUCE {
            //GReal x2 = x2min + j*dx;
            GReal x1 = x1min + i*dx;
            GReal x2 = 0.5;

            GReal Xnative[GR_DIM] = {0,x1,x2,0};
            GReal Xembed[GR_DIM];
            G.coords.coord_to_embed(Xnative, Xembed);
            GReal r = Xembed[1], th = Xembed[2];

            // Abbreviated version of the full primitives calculation
            //printf("lnh calc with %g %g %g %g %g\n", ksc.a, l, rin, r, th);
            Real lnh = lnh_calc(ksc.a, l, rin, r, th);
            // if (lnh >= 0. || r >= rin) {
            //     printf("a: %g l: %g lnh: %g r: %g th: %g\n", ksc.a, l, lnh, r, th);
            // }
            if (lnh >= 0. && r >= rin) {
                // Calculate rho
                Real hm1 = exp(lnh) - 1.;
                Real rho = pow(hm1 * (gam - 1.) / (kappa * gam),
                                    1. / (gam - 1.));
                //Real u = kappa * pow(rho, gam) / (gam - 1.);

                // Record max.  Maybe more efficient to bail earlier?  Meh.
                //printf("lnh: %g rho: %g\n", lnh, rho);
                if (rho > local_result) local_result = rho;
                //if (u > u_max) u_max = u;
            }
        }
    , max_reducer);

    if (pmb->packages.Get("GRMHD")->Param<int>("verbose") > 0) {
        cout << "Initial maximum density is " << rho_max << endl;
    }

    pmb->par_for("fm_torus_normalize", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            P(prims::rho, k, j, i) /= rho_max;
            P(prims::u, k, j, i) /= rho_max;
        }
    );
}

void PerturbU(MeshBlock *pmb, GridVars P, Real u_jitter, int rng_seed=31337)
{
    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    // Should only jitter physical zones...

    // SERIAL VERSION -- better determinism guarantee but CPU only
    // Initialize RNG
    // std::mt19937 gen(rng_seed);
    // std::uniform_real_distribution<Real> dis(-u_jitter/2, u_jitter/2);

    // for(int k=0; k < n3; k++)
    //     for(int j=0; j < n2; j++)
    //         for(int i=0; i < n3; i++)
    //             P(prims::u, k, j, i) *= 1. + dis(gen);


    // Kokkos version.  This would probably be much faster (and more deterministic?) as 2D internal, 1D outside
    // TODO check determinism...
    typedef typename Kokkos::Random_XorShift64_Pool<> RandPoolType;
    RandPoolType rand_pool(rng_seed);
    typedef typename RandPoolType::generator_type gen_type;
    pmb->par_for("perturb_u", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            gen_type rgen = rand_pool.get_state();
            if (P(prims::rho, k, j, i) > 1.e-5) {
                P(prims::u, k, j, i) *= 1. + Kokkos::rand<gen_type, Real>::draw(rgen, -u_jitter/2, u_jitter/2);
            }
            rand_pool.free_state(rgen);
        }
    );
}
