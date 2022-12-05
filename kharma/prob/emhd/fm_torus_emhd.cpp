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

#include "../fm_torus.hpp"

#include "mpi.hpp"
#include "prob_common.hpp"
#include "types.hpp"

#include <random>
#include "Kokkos_Random.hpp"

TaskStatus InitializeFMTorusEMHD(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag(rc, "Initializing torus problem");

    auto pmb        = rc->GetBlockPointer();
    GridScalar rho  = rc->Get("prims.rho").data;
    GridScalar u    = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P  = rc->Get("prims.B").data;

    // This problem init is exclusively for the EMHD torus; get copies of q and dP
    const bool use_emhd   = pin->GetOrAddBoolean("emhd", "on", true);
    GridVector q          = rc->Get("prims.q").data;
    GridVector dP         = rc->Get("prims.dP").data;
    const auto& emhd_pars = pmb->packages.Get("EMHD")->AllParams();

    const GReal rin      = pin->GetOrAddReal("torus", "rin", 6.0);
    const GReal rmax     = pin->GetOrAddReal("torus", "rmax", 12.0);
    const Real kappa     = pin->GetOrAddReal("torus", "kappa", 1.e-3);
    const GReal tilt_deg = pin->GetOrAddReal("torus", "tilt", 0.0);
    const GReal tilt     = tilt_deg / 180. * M_PI;
    const Real gam       = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    IndexDomain domain = IndexDomain::interior;
    const int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    const int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    const int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // Get coordinate systems
    // G clearly holds a reference to an existing system G.coords.base,
    // but we don't know if it's KS or BL coordinates
    // Since we can't create a system and assign later, we just
    // rebuild copies of both based on the BH spin "a"
    const auto& G              = pmb->coords;
    const bool use_ks          = G.coords.is_ks();
    const GReal a              = G.coords.get_a();
    const SphBLCoords blcoords = SphBLCoords(a);
    const SphKSCoords kscoords = SphKSCoords(a);

    // Fishbone-Moncrief parameters
    Real l = lfish_calc(a, rmax);

    pmb->par_for("fm_torus_init", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            GReal Xnative[GR_DIM], Xembed[GR_DIM], Xmidplane[GR_DIM];
            G.coord(k, j, i, Loci::center, Xnative);
            G.coord_embed(k, j, i, Loci::center, Xembed);
            // What are our corresponding "midplane" values for evaluating the function?
            rotate_polar(Xembed, tilt, Xmidplane);

            GReal r   = Xmidplane[1], th = Xmidplane[2];
            GReal sth = sin(th);
            GReal cth = cos(th);

            Real lnh = lnh_calc(a, l, rin, r, th);

            // Region inside magnetized torus; u^i is calculated in
            // Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
            // so it needs to be transformed at the end
            // everything outside is left 0 to be added by the floors
            if (lnh >= 0. && r >= rin) {
                Real r2 = r*r;
                Real a2 = a*a;
                Real DD = r2 - 2. * r + a2;
                Real AA = m::pow(r2 + a2, 2) - DD * a2 * sth * sth;
                Real SS = r2 + a2 * cth * cth;

                // Calculate rho and u
                Real hm1   = exp(lnh) - 1.;
                Real rho_l = m::pow(hm1 * (gam - 1.) / (kappa * gam), 1. / (gam - 1.));
                Real u_l   = kappa * m::pow(rho_l, gam) / (gam - 1.);

                // Calculate u^phi
                Real expm2chi = SS * SS * DD / (AA * AA * sth * sth);
                Real up1      = m::sqrt((-1. + m::sqrt(1. + 4. * l * l * expm2chi)) / 2.);
                Real up       = 2. * a * r * m::sqrt(1. + up1 * up1) / m::sqrt(AA * SS * DD) +
                                m::sqrt(SS / AA) * up1 / sth;

                const Real ucon_tilt[GR_DIM] = {0., 0., 0., up};
                Real ucon_bl[GR_DIM];
                rotate_polar_vec(Xmidplane, ucon_tilt, -tilt, Xembed, ucon_bl);

                Real gcov_bl[GR_DIM][GR_DIM];
                blcoords.gcov_embed(Xembed, gcov_bl);
                set_ut(gcov_bl, ucon_bl);

                // Then transform that 4-vector to KS if necessary,
                // and then to native coordinates
                Real ucon_native[GR_DIM];
                if (use_ks) {
                    Real ucon_ks[GR_DIM];
                    kscoords.vec_from_bl(Xembed, ucon_bl, ucon_ks);
                    G.coords.con_vec_to_native(Xnative, ucon_ks, ucon_native);
                } else {
                    G.coords.con_vec_to_native(Xnative, ucon_bl, ucon_native);
                }

                // Convert native 4-vector to primitive u-twiddle, see Gammie '04
                Real gcon[GR_DIM][GR_DIM], u_prim[NVEC];
                G.gcon(Loci::center, j, i, gcon);
                fourvel_to_prim(gcon, ucon_native, u_prim);

                rho(k, j, i) = rho_l;
                u(k, j, i)   = u_l;
                uvec(0, k, j, i) = u_prim[0];
                uvec(1, k, j, i) = u_prim[1];
                uvec(2, k, j, i) = u_prim[2];
                // EMHD variables
                q(k, j, i)  = 0.;
                dP(k, j, i) = 0.;
            }
        }
    );

    // Find rho_max "analytically" by looking over the whole mesh domain for the maximum in the midplane
    // Done device-side for speed (for large 2D meshes this may get bad) but may work fine in HostSpace
    // Note this covers the full domain on each rank: it doesn't need a grid so it's not a memory problem,
    // and an MPI synch as is done for beta_min would be a headache
    GReal x1min = pmb->pmy_mesh->mesh_size.x1min;
    GReal x1max = pmb->pmy_mesh->mesh_size.x1max;
    // Add back 2D if torus solution may not be largest in midplane (before tilt ofc)
    //GReal x2min = pmb->pmy_mesh->mesh_size.x2min;
    //GReal x2max = pmb->pmy_mesh->mesh_size.x2max;
    GReal dx = 0.001;
    int nx1  = (x1max - x1min) / dx;
    //int nx2 = (x2max - x2min) / dx;

    // If we print diagnostics, do so only from block 0 as the others do exactly the same thing
    // Since this is initialization, we are guaranteed to have a block 0
    if (pmb->gid == 0 && pmb->packages.Get("GRMHD")->Param<int>("verbose") > 0) {
        std::cout << "Calculating maximum density:" << std::endl;
        std::cout << "a = " << a << std::endl;
        std::cout << "dx = " << dx << std::endl;
        std::cout << "x1min->x1max: " << x1min << " " << x1max << std::endl;
        std::cout << "nx1 = " << nx1 << std::endl;
        //cout << "x2min->x2max: " << x2min << " " << x2max << std::endl;
        //cout << "nx2 = " << nx2 << std::endl;
    }

    Real rho_max = 0;
    Kokkos::Max<Real> max_reducer(rho_max);
    pmb->par_reduce("fm_torus_maxrho", 0, nx1,
        KOKKOS_LAMBDA_1D_REDUCE {
            GReal x1 = x1min + i*dx;
            //GReal x2 = x2min + j*dx;
            GReal Xnative[GR_DIM] = {0,x1,0,0};
            GReal Xembed[GR_DIM];
            G.coords.coord_to_embed(Xnative, Xembed);
            const GReal r = Xembed[1];
            // Regardless of native coordinate shenanigans,
            // set th=pi/2 since the midplane is densest in the solution
            const GReal rho = fm_torus_rho(a, rin, rmax, gam, kappa, r, M_PI/2.);
            // TODO umax for printing/recording?

            // Record max
            if (rho > local_result) local_result = rho;
        }
    , max_reducer);

    // Record and print normalization factor
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rho_norm")))
        pmb->packages.Get("GRMHD")->AllParams().Add("rho_norm", rho_max);
    if (pmb->gid == 0 && pmb->packages.Get("GRMHD")->Param<int>("verbose") > 0) {
        std::cout << "Initial maximum density is " << rho_max << std::endl;
    }

    pmb->par_for("fm_torus_normalize", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            rho(k, j, i) /= rho_max;
            u(k, j, i)   /= rho_max;
        }
    );

    return TaskStatus::complete;
}
