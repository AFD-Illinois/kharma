/* 
 *  File: bondi.hpp
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
#pragma once

#include "decs.hpp"

#include "gr_coordinates.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "prob_common.hpp"
#include "types.hpp"
#include "hdf5_utils.h"

#include <parthenon/parthenon.hpp>

/**
 * Initialization of a Bondi problem with specified sonic point and BH accretion rate mdot
 * TODO mdot and rs are redundant and should be merged into one parameter
 */
TaskStatus InitializeBondi(MeshBlockData<Real> *rc, ParameterInput *pin);

/**
 * Set all values on a given domain to the Bondi inflow analytic steady-state solution
 * 
 * Used for initialization and boundary conditions
 */
TaskStatus SetBondi(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);

/**
 * Supporting functions for Bondi flow calculations
 * 
 * Adapted from M. Chandra
 * Modified by Hyerin Cho and Ramesh Narayan
 */
KOKKOS_INLINE_FUNCTION Real get_Tfunc(const Real T, const GReal r, const Real C1, const Real C2, const Real n)
{
    return m::pow(1. + (1. + n) * T, 2.) * (1. - 2. / r + m::pow(C1 / m::pow(r,2) / m::pow(T, n), 2.)) - C2;
}
KOKKOS_INLINE_FUNCTION Real get_T(const GReal r, const Real C1, const Real C2, const Real n, const Real rs)
{
    Real rtol = 1.e-12;
    Real ftol = 1.e-14;
    Real Tinf = (m::sqrt(C2) - 1.) / (n + 1); // temperature at infinity
    Real Tnear = m::pow(C1 * m::sqrt(2. / m::pow(r,3)), 1. / n); // temperature near the BH
    Real Tmin, Tmax;

    // There are two branches of solutions (see Michel et al. 1971) and the two branches cross at rs.
    // These bounds are set to only select the inflowing solution only.
    if (r<rs) {
        Tmin = Tinf;
        Tmax = Tnear;
    }
    else {
        Tmin = m::max(Tnear,Tinf);
        Tmax = 1.;
    }

    Real f0, f1, fh;
    Real T0, T1, Th;
    T0 = Tmin;
    f0 = get_Tfunc(T0, r, C1, C2, n);
    T1 = Tmax;
    f1 = get_Tfunc(T1, r, C1, C2, n);
    if (f0 * f1 > 0) return -1;

    Th = (T0 + T1) / 2.; // a simple bisection method which is stable and fast
    fh = get_Tfunc(Th, r, C1, C2, n);
    Real epsT = rtol * (Tmin + Tmax);
    while (m::abs(Th - T0) > epsT && m::abs(Th - T1) > epsT && m::abs(fh) > ftol)
    {
        if (fh * f0 > 0.) {
            T0 = Th;
            f0 = fh;
        } else {
            T1 = Th;
            f1 = fh;
        }

        Th = (T0 + T1) / 2.; 
        fh = get_Tfunc(Th, r, C1, C2, n);
    }

    return Th;
}

/**
 * Get the Bondi solution at a particular zone
 * Note this assumes that there are ghost zones!
 * 
 * TODO could put this back into SetBondi
 */
KOKKOS_INLINE_FUNCTION void get_prim_bondi(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
                                           const Real& gam, const SphBLCoords& bl,  const SphKSCoords& ks, 
                                           const Real mdot, const Real rs, const Real r_shell, const Real ur_frac, const Real uphi, const int& k, const int& j, const int& i)
{
    // Solution constants
    // Ideally these could be cached but preformance isn't an issue here
    Real n = 1. / (gam - 1.);
    Real uc = m::sqrt(mdot / (2. * rs));
    Real Vc = -m::sqrt(m::pow(uc, 2) / (1. - 3. * m::pow(uc, 2)));
    Real Tc = -n * m::pow(Vc, 2) / ((n + 1.) * (n * m::pow(Vc, 2) - 1.));
    Real C1 = uc * m::pow(rs, 2) * m::pow(Tc, n);
    Real C2 = m::pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + m::pow(C1, 2) / (m::pow(rs, 4) * m::pow(Tc, 2 * n)));

    GReal Xnative[GR_DIM], Xembed[GR_DIM];
    G.coord(k, j, i, Loci::center, Xnative);
    G.coord_embed(k, j, i, Loci::center, Xembed);
    GReal r = Xembed[1];
    //GReal th = Xembed[2];
    // Unless we're doing a Schwarzchild problem & comparing solutions,
    // be a little cautious about initializing the Ergosphere zones
    if (ks.a > 0.1 && r < 2) return;

    Real T = get_T(r, C1, C2, n, rs);
    Real ur = -C1 / (m::pow(T, n) * m::pow(r, 2)) * ur_frac;

    // Set u^t to make u^r a 4-vector
    Real ucon_bl[GR_DIM] = {0, ur, 0, 0};

    // values at infinity (obtained by putting r = rshell)
    Real rho, u, rho0, T0, u0;
    T0 = get_T(r_shell, C1, C2, n, rs);
    rho0 = m::pow(T0, n);
    u0 = rho0 * T0 * n;

    Real rb = rs * rs; // Bondi radius
    // interpolation between inner and outer regimes
    rho = rho0 * (r + rb) / r;
    //T = T0 * (r + rb) / r; // use the same analytic temperature solution since T already goes like ~1/r
    u = rho * T * n;

    ucon_bl[3]=uphi*m::pow(r,-3./2.); // (04/13/23) a fraction of the kepler //*m::sin(th); // 04/04/23 set it to some small angular velocity. smallest at the poles
    Real gcov_bl[GR_DIM][GR_DIM];
    bl.gcov_embed(Xembed, gcov_bl);
    set_ut(gcov_bl, ucon_bl);

    // Then transform that 4-vector to KS, then to native
    Real ucon_ks[GR_DIM], ucon_mks[GR_DIM];
    ks.vec_from_bl(Xembed, ucon_bl, ucon_ks);
    coords.con_vec_to_native(Xnative, ucon_ks, ucon_mks);

    // Convert native 4-vector to primitive u-twiddle, see Gammie '04
    Real gcon[GR_DIM][GR_DIM], u_prim[NVEC];
    G.gcon(Loci::center, j, i, gcon);
    fourvel_to_prim(gcon, ucon_mks, u_prim);

    if (!isnan(rho)) P(m_p.RHO, k, j, i) = rho;
    if (!isnan(u)) P(m_p.UU, k, j, i) = u;
    if (!isnan(u_prim[0])) P(m_p.U1, k, j, i) = u_prim[0];
    if (!isnan(u_prim[1])) P(m_p.U2, k, j, i) = u_prim[1];
    if (!isnan(u_prim[2])) P(m_p.U3, k, j, i) = u_prim[2];
}

KOKKOS_INLINE_FUNCTION void XtoindexGizmo(const GReal XG[GR_DIM],
                                    const GridScalar& rarr, const int length, int& i, GReal& del)
{
    Real dx2, dx2_min;
    dx2_min=m::pow(XG[1]-rarr(0),2); //100000.; //arbitrarily large number

    i = 0; // initialize

    for (int itemp = 0; itemp < length; itemp++) {
        if (rarr(itemp) < XG[1]) { // only look for smaller side
            dx2 = m::pow(XG[1]-rarr(itemp),2); //pow(XG[1]-rarr[itemp],2.);

            // simplest interpolation (Hyerin 07/26/22)
            if (dx2<dx2_min){
                dx2_min=dx2;
                i = itemp;
            }
        }
    }
    
    // interpolation (11/14/2022) TODO: write a case where indices hit the boundaries of the data file
    del = (XG[1]-rarr(i))/(rarr(i+1)-rarr(i));

    //if (m::abs(dx2_min/m::pow(XG[1],2))>1.e-8) printf("XtoindexGizmo: dx2 frac diff large = %g at r= %g \n",m::sqrt(dx2_min)/XG[1], XG[1]); this is interpolation anyway
}
/**
 * Get the GIZMO output values at a particular zone
 * Note this assumes that there are ghost zones!
 * TODO: Hyerin: maybe combine with get_prim_bondi 
 */
KOKKOS_INLINE_FUNCTION void get_prim_gizmo_shell(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
                                           const Real& gam, const SphBLCoords& bl,  const SphKSCoords& ks, 
                                           const Real r_shell, const Real rs, Real vacuum_logrho, Real vacuum_log_u_over_rho,
                                           const GridScalar& rarr, const GridScalar& rhoarr, const GridScalar& Tarr, const GridScalar& vrarr, const int length,
                                           const int& k, const int& j, const int& i)
{
    // Solution constants for velocity prescriptions
    // Ideally these could be cached but preformance isn't an issue here
    Real mdot = 1.; // mdot and rs defined arbitrarily
    Real n = 1. / (gam - 1.);
    Real uc = sqrt(mdot / (2. * rs));
    Real Vc = -sqrt(pow(uc, 2) / (1. - 3. * pow(uc, 2)));
    Real Tc = -n * pow(Vc, 2) / ((n + 1.) * (n * pow(Vc, 2) - 1.));
    Real C1 = uc * pow(rs, 2) * pow(Tc, n);
    Real C2 = pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + pow(C1, 2) / (pow(rs, 4) * pow(Tc, 2 * n)));

    Real smallrho=pow(10.,vacuum_logrho); // pow(10.,-4.);
    Real smallu = smallrho*pow(10.,vacuum_log_u_over_rho);

    //Real T = smallu/(smallrho*n);

    //Real rs = 1./sqrt(T); //1000.;
    GReal Xnative[GR_DIM], Xembed[GR_DIM];
    G.coord(k, j, i, Loci::center, Xnative);
    G.coord_embed(k, j, i, Loci::center, Xembed);
    GReal r = Xembed[1];

    Real rho, u;
    int itemp;
    GReal del;

    // Unless we're doing a Schwarzchild problem & comparing solutions,
    // be a little cautious about initializing the Ergosphere zones
    if (ks.a > 0.1 && r < 2) return;

    Real T = get_T(r, C1, C2, n, rs);
    Real ur = -C1 / (pow(T, n) * pow(r, 2));
    //Real ucon_bl[GR_DIM];
    //ucon_bl[0] = 0.;
    //ucon_bl[2] = 0.;
    //ucon_bl[3] = 0.;
    Real ucon_bl[GR_DIM] = {0, 0, 0, 0};
    if (r<r_shell*0.9){
        rho = smallrho;
        u = smallu;
        ucon_bl[1] = ur;
    } else {
        XtoindexGizmo(Xembed, rarr, length, itemp, del);
        // linear interpolation
        if (del < 0 ) { // when r is smaller than GIZMO's range
            del = 0; // just copy over the smallest r values
        }
        rho = rhoarr(itemp)*(1.-del)+rhoarr(itemp+1)*del;
        u = rho*(Tarr(itemp)*(1.-del)+Tarr(itemp+1)*del)*n;
        //ucon_bl[1] = 0.; // 10/23/2022 test zero velocity for the bondi shell
    }

    // Set u^t to make u^r a 4-vector
    Real gcov_bl[GR_DIM][GR_DIM];
    bl.gcov_embed(Xembed, gcov_bl);
    set_ut(gcov_bl, ucon_bl);

    // Then transform that 4-vector to KS, then to native
    Real ucon_ks[GR_DIM], ucon_mks[GR_DIM];
    ks.vec_from_bl(Xembed, ucon_bl, ucon_ks);
    coords.con_vec_to_native(Xnative, ucon_ks, ucon_mks);

    // Convert native 4-vector to primitive u-twiddle, see Gammie '04
    Real gcon[GR_DIM][GR_DIM], u_prim[NVEC];
    G.gcon(Loci::center, j, i, gcon);
    fourvel_to_prim(gcon, ucon_mks, u_prim);

    P(m_p.RHO, k, j, i) = rho;
    P(m_p.UU, k, j, i) = u;
    P(m_p.U1, k, j, i) = u_prim[0];
    P(m_p.U2, k, j, i) = u_prim[1];
    P(m_p.U3, k, j, i) = u_prim[2];
}

KOKKOS_INLINE_FUNCTION void XtoindexGizmo3D(const GReal XG[GR_DIM],
                                    const GridVector& coordarr, const hsize_t length, int& i, GReal& del)
{
    Real dx2, dx2_min;
    //Real XG_cart[3] = {XG[1]*m::sin(XG[2])*m::cos(XG[3]), XG[1]*m::sin(XG[2])*m::sin(XG[3]), XG[1]*m::cos(XG[2])};

    //Real x = coordarr(0,0)*m::sin(coordarr(0,1))*m::cos(coordarr(0,2));
    //Real y = coordarr(0,0)*m::sin(coordarr(0,1))*m::sin(coordarr(0,2));
    //Real z = coordarr(0,0)*m::cos(coordarr(0,1));
    //dx2_min=m::pow(XG_cart[0]-x,2)+m::pow(XG_cart[1]-y,2)+m::pow(XG_cart[2]-z,2); 
    //dx2_min=m::pow(coordarr(0,0)/XG[1]-1.,2.)+m::pow((coordarr(0,1)-XG[2])/M_PI,2.)+m::pow((coordarr(0,2)-XG[3])/(2.*M_PI),2.); // sum of fractional diff^2 for each r, th, phi
    dx2_min=m::pow(coordarr(0,0)-XG[1],2.)+m::pow((coordarr(0,1)-XG[2])/M_PI,2.)+m::pow((coordarr(0,2)-XG[3])/(2.*M_PI),2.); // sum of diff^2 for each r, th, phi

    i = 0; // initialize

    for (int itemp = 0; itemp < length; itemp++) {
        if (coordarr(itemp,0) <= XG[1]) { // only look for smaller side
            //x = coordarr(itemp,0)*m::sin(coordarr(itemp,1))*m::cos(coordarr(itemp,2));
            //y = coordarr(itemp,0)*m::sin(coordarr(itemp,1))*m::sin(coordarr(itemp,2));
            //z = coordarr(itemp,0)*m::cos(coordarr(itemp,1));
            //dx2 = m::pow(XG_cart[0]-x,2)+m::pow(XG_cart[1]-y,2)+m::pow(XG_cart[2]-z,2); 
            //dx2 = m::pow(coordarr(itemp,0)/XG[1]-1.,2.)+m::pow((coordarr(itemp,1)-XG[2])/M_PI,2.)+m::pow((coordarr(itemp,2)-XG[3])/(2.*M_PI),2.);
            dx2 = m::pow(coordarr(itemp,0)-XG[1],2.)+m::pow((coordarr(itemp,1)-XG[2])/M_PI,2.)+m::pow((coordarr(itemp,2)-XG[3])/(2.*M_PI),2.);

            // simplest interpolation (Hyerin 07/26/22)
            if (dx2<dx2_min){
                dx2_min=dx2;
                i = itemp;
            }
        }
    }
    
    // No interpolation! Warn if the data points are not exactly on top of each other
    if (m::abs(dx2_min)>1.e-8) printf("XtoindexGizmo3D: dx2 frac diff large = %g at (r,th,phi)=(%lf %lf %lf) fitted=(%lf %lf %lf) \n",m::sqrt(dx2_min), XG[1], XG[2], XG[3], coordarr(i,0),coordarr(i,1),coordarr(i,2));
}
/**
 * Get the GIZMO output values at a particular zone for 3D GIZMO data
 * Note this assumes that there are ghost zones!
 * TODO: Hyerin: maybe combine with get_prim_bondi and get_prim_gizmo_shell
 */
KOKKOS_INLINE_FUNCTION void get_prim_gizmo_shell_3d(const GRCoordinates& G, const CoordinateEmbedding& coords, const VariablePack<Real>& P, const VarMap& m_p,
                                           const Real& gam, const SphBLCoords& bl,  const SphKSCoords& ks, 
                                           const Real r_shell, const Real rs, Real vacuum_logrho, Real vacuum_log_u_over_rho,
                                           const GridVector& coordarr, const GridScalar& rhoarr, const GridScalar& Tarr, const GridVector& varr, const hsize_t length,
                                           const int& k, const int& j, const int& i)
{
    // Solution constants for velocity prescriptions
    // Ideally these could be cached but preformance isn't an issue here
    Real mdot = 1.; // mdot and rs defined arbitrarily
    Real n = 1. / (gam - 1.);
    Real uc = sqrt(mdot / (2. * rs));
    Real Vc = -sqrt(pow(uc, 2) / (1. - 3. * pow(uc, 2)));
    Real Tc = -n * pow(Vc, 2) / ((n + 1.) * (n * pow(Vc, 2) - 1.));
    Real C1 = uc * pow(rs, 2) * pow(Tc, n);
    Real C2 = pow(1. + (1. + n) * Tc, 2) * (1. - 2. * mdot / rs + pow(C1, 2) / (pow(rs, 4) * pow(Tc, 2 * n)));

    Real smallrho=pow(10.,vacuum_logrho); // pow(10.,-4.);
    Real smallu = smallrho*pow(10.,vacuum_log_u_over_rho);

    //Real T = smallu/(smallrho*n);

    //Real rs = 1./sqrt(T); //1000.;
    GReal Xnative[GR_DIM], Xembed[GR_DIM];//, Xembed_corner[GR_DIM];
    G.coord(k, j, i, Loci::center, Xnative);
    G.coord_embed(k, j, i, Loci::center, Xembed);
    //G.coord_embed(k, j, i, Loci::corner, Xembed_corner); // TODO: get cell centered values from KungYi
    GReal r = Xembed[1];
    GReal th = Xembed[2];

    Real rho, u, T, ur, uth, uphi;
    int itemp;
    GReal del;

    // Unless we're doing a Schwarzchild problem & comparing solutions,
    // be a little cautious about initializing the Ergosphere zones
    if (ks.a > 0.1 && r < 2) return;

    T = get_T(r, C1, C2, n, rs);
    ur = -C1 / (pow(T, n) * pow(r, 2));
    Real ucon_bl[GR_DIM] = {0, 0, 0, 0};
    if (r<r_shell*0.9){
        rho = smallrho;
        u = smallu;
        ucon_bl[1] = ur;
    } else {
        XtoindexGizmo3D(Xembed, coordarr, length, itemp, del);
        // DO NOT INTERPOLATE, it is assumed GIZMO data is right on the grid
        rho = rhoarr(itemp);
        u = rho*(Tarr(itemp))*n;
        ur = varr(itemp,0);
        uth = varr(itemp,1)/r;
        uphi = varr(itemp,2)/(r*m::sin(th));
        // Newtonian limit
        ucon_bl[1] = ur;
        ucon_bl[2] = uth;
        ucon_bl[3] = uphi;
    }

    // Set u^t to make u^r a 4-vector
    Real gcov_bl[GR_DIM][GR_DIM];
    bl.gcov_embed(Xembed, gcov_bl);
    set_ut(gcov_bl, ucon_bl);

    // Then transform that 4-vector to KS, then to native
    Real ucon_ks[GR_DIM], ucon_mks[GR_DIM];
    ks.vec_from_bl(Xembed, ucon_bl, ucon_ks);
    coords.con_vec_to_native(Xnative, ucon_ks, ucon_mks);

    // Convert native 4-vector to primitive u-twiddle, see Gammie '04
    Real gcon[GR_DIM][GR_DIM], u_prim[NVEC];
    G.gcon(Loci::center, j, i, gcon);
    fourvel_to_prim(gcon, ucon_mks, u_prim);

    P(m_p.RHO, k, j, i) = rho;
    P(m_p.UU, k, j, i) = u;
    P(m_p.U1, k, j, i) = u_prim[0];
    P(m_p.U2, k, j, i) = u_prim[1];
    P(m_p.U3, k, j, i) = u_prim[2];
}
