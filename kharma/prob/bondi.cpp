/* 
 *  File: bondi.cpp
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

// BONDI PROBLEM

#include "decs.hpp"

#include "gr_coordinates.hpp"
#include "eos.hpp"
#include "phys.hpp"
#include "prob_common.hpp"

#include "mesh/mesh.hpp"

using namespace std;

KOKKOS_INLINE_FUNCTION void get_prim_bondi(const GRCoordinates& G, const CoordinateEmbedding& coords, GridVars P, const EOS* eos, const SphBLCoords& bl,  const SphKSCoords& ks, 
                                            const Real mdot, const Real rs, const int& k, const int& j, const int& i);

/**
 * Initialization of a Bondi problem with specified sonic point, BH mdot, and horizon radius
 * TODO this can/should be just mdot (and the grid ofc), if this problem is to be used as anything more than a test
 */
void InitializeBondi(MeshBlock *pmb, const GRCoordinates& G, GridVars P,
                     const EOS* eos, const Real mdot, const Real rs)
{
    FLAG("Initializing Bondi problem");

    SphKSCoords ks = mpark::get<SphKSCoords>(G.coords.base);
    SphBLCoords bl = SphBLCoords(ks.a);
    CoordinateEmbedding cs = G.coords;

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    pmb->par_for("init_bondi", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            get_prim_bondi(G, cs, P, eos, bl, ks, mdot, rs, k, j, i);
        }
    );
    FLAG("Initialized Bondi");
}

void ApplyBondiBoundary(std::shared_ptr<MeshBlockData<Real>>& rc)
{
    auto pmb = rc->GetBlockPointer();
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GRCoordinates G = pmb->coords;

    FLAG("Applying Bondi X1R boundary");

    Real mdot = pmb->packages["GRMHD"]->Param<Real>("mdot");
    Real rs = pmb->packages["GRMHD"]->Param<Real>("rs");
    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    // Just the X1 right boundary
    SphKSCoords ks = mpark::get<SphKSCoords>(G.coords.base);
    SphBLCoords bl = SphBLCoords(ks.a);
    CoordinateEmbedding cs = G.coords;

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    pmb->par_for("bondi_boundary", kb.s, kb.e, jb.s, jb.e, ib.e+1, n1-1,
        KOKKOS_LAMBDA_3D {
            get_prim_bondi(G, cs, P, eos, bl, ks, mdot, rs, k, j, i);
            p_to_u(G, P, eos, k, j, i, U);
        }
    );
}
// Adapted from M. Chandra
KOKKOS_INLINE_FUNCTION Real get_Tfunc(const Real T, const GReal r, const Real C1, const Real C2, const Real n)
{
    return pow(1. + (1. + n) * T, 2.) * (1. - 2. / r + pow(C1 / pow(r,2) / pow(T, n), 2.)) - C2;
}

KOKKOS_INLINE_FUNCTION Real get_T(const GReal r, const Real C1, const Real C2, const Real n)
{
    Real rtol = 1.e-12;
    Real ftol = 1.e-14;
    Real Tmin = 0.6 * (sqrt(C2) - 1.) / (n + 1);
    Real Tmax = pow(C1 * sqrt(2. / pow(r,3)), 1. / n);

    Real f0, f1, fh;
    Real T0, T1, Th;
    T0 = Tmin;
    f0 = get_Tfunc(T0, r, C1, C2, n);
    T1 = Tmax;
    f1 = get_Tfunc(T1, r, C1, C2, n);
    if (f0 * f1 > 0) return -1;

    Th = (f1 * T0 - f0 * T1) / (f1 - f0);
    fh = get_Tfunc(Th, r, C1, C2, n);
    Real epsT = rtol * (Tmin + Tmax);
    while (fabs(Th - T0) > epsT && fabs(Th - T1) > epsT && fabs(fh) > ftol)
    {
        if (fh * f0 < 0.) {
            T0 = Th;
            f0 = fh;
        } else {
            T1 = Th;
            f1 = fh;
        }

        Th = (f1 * T0 - f0 * T1) / (f1 - f0);
        fh = get_Tfunc(Th, r, C1, C2, n);
    }

    return Th;
}

/**
 * Get the Bondi solution at a particular zone.  Can ideally be host- or device-side, but careful of EOS.
 * Note this assumes that there are ghost zones!
 */
KOKKOS_INLINE_FUNCTION void get_prim_bondi(const GRCoordinates& G, const CoordinateEmbedding& coords, GridVars P, const EOS* eos, const SphBLCoords& bl,  const SphKSCoords& ks, 
                                            const Real mdot, const Real rs, const int& k, const int& j, const int& i)
{
    // Solution constants
    // Ideally these could be cached but preformance isn't an issue here
    Real n = 1. / (eos->gam - 1.);
    Real uc = sqrt(mdot / (2. * rs));
    Real Vc = -sqrt(pow(uc, 2) / (1. - 3. * pow(uc, 2)));
    Real Tc = -n * pow(Vc, 2) / ((n + 1.) * (n * pow(Vc, 2) - 1.));
    Real C1 = uc * pow(rs, 2) * pow(Tc, n);
    Real C2 = pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + pow(C1, 2) / (pow(rs, 4) * pow(Tc, 2 * n)));

    GReal X[GR_DIM], Xembed[GR_DIM];
    G.coord(k, j, i, Loci::center, X);
    coords.coord_to_embed(X, Xembed);
    Real Rhor = ks.rhor();
    // Any zone inside the horizon gets the horizon's values
    // TODO There may be a more stable way to initialize inside the EH...
    int ii = i;
    while (Xembed[1] < Rhor)
    {
        ++ii;
        G.coord(k, j, ii, Loci::center, X);
        coords.coord_to_embed(X, Xembed);
    }
    GReal r = Xembed[1];

    Real T = get_T(r, C1, C2, n);
    //if (T < 0) T = 0; // If you can't error, NaN
    Real ur = -C1 / (pow(T, n) * pow(r, 2));
    Real rho = pow(T, n);
    Real u = rho * T * n;

    // Set u^t to make u^r a 4-vector
    Real ucon_bl[GR_DIM] = {0, ur, 0, 0};
    Real gcov_bl[GR_DIM][GR_DIM];
    bl.gcov_embed(Xembed, gcov_bl);
    set_ut(gcov_bl, ucon_bl);

    // Then transform that 4-vector to KS, then to native
    Real ucon_ks[GR_DIM], ucon_mks[GR_DIM];
    ks.vec_from_bl(Xembed, ucon_bl, ucon_ks);
    coords.con_vec_to_native(X, ucon_ks, ucon_mks);

    // Convert native 4-vector to primitive u-twiddle, see Gammie '04
    Real gcon[GR_DIM][GR_DIM], u_prim[GR_DIM];
    G.gcon(Loci::center, j, i, gcon);
    fourvel_to_prim(gcon, ucon_mks, u_prim);

#if DEBUG && 0
    // TODO print in device-compatible way
    static int fails = 0;
    if (T < 0 || rho < 0 || u < 0) {
        Real ucov_mks[GR_DIM];
        G.lower(ucon_mks, ucov_mks, k, j, i, Loci::center);
        Real uu_mks = dot(ucon_mks, ucov_mks);

        cerr << "Bondi Bad in zone " << i << " " << j << " " << k << std::endl;
        cerr << "Native X:  " << X[1] << " " << X[2] << " " << X[3] << std::endl;
        cerr << "Embed X:  " << Xembed[1] << " " << Xembed[2] << " " << Xembed[3] << std::endl;
        cerr << "gam, C1, C2, n, T: " << eos->gam << " " << C1 << " " << C2 << " " << n << " " << T << std::endl;
        cerr << "Tmin, Tmax: " << 0.6 * (sqrt(C2) - 1.) / (n + 1) << " " << pow(C1 * sqrt(2. / pow(r,3)), 1. / n) << std::endl;
        cerr << "rho, u:  " << rho << " " << u << std::endl;
        cerr << "u [BL]:  " << ucon_bl[0] << " " << ucon_bl[1] << " " << ucon_bl[2] << " " << ucon_bl[3] << std::endl;
        cerr << "u [KS]:  " << ucon_ks[0] << " " << ucon_ks[1] << " " << ucon_ks[2] << " " << ucon_ks[3] << std::endl;
        cerr << "u [native]:  " << ucon_mks[0] << " " << ucon_mks[1] << " " << ucon_mks[2] << " " << ucon_mks[3] << std::endl;
        cerr << "u.u [native]:  " << uu_mks << std::endl;
        cerr << "u [prim]:  " << u_prim[1] << " " << u_prim[2] << " " << u_prim[3] << std::endl;
        fails++;
    }
    if(fails > 100) {
        exit(-1);
    }
#endif

    P(prims::rho, k, j, i) = rho;
    P(prims::u, k, j, i) = u;
    P(prims::u1, k, j, i) = u_prim[1];
    P(prims::u2, k, j, i) = u_prim[2];
    P(prims::u3, k, j, i) = u_prim[3];
    P(prims::B1, k, j, i) = 0.;
    P(prims::B2, k, j, i) = 0.;
    P(prims::B3, k, j, i) = 0.;
}