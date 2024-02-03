/* 
 *  File: kastaun.hpp
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
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#pragma once

// General template
// We define a specialization based on the Inverter::Type parameter
#include "invert_template.hpp"

#include "floors.hpp"
#include "grmhd_functions.hpp"
#include "kharma_utils.hpp"

// The following functions are stolen from AthenaK, ideal_c2p_mhd.hpp
// They have been lightly adapted to fit into KHARMA

namespace Inverter {

// structs to store primitive/conserved variables in one-dimension
// (density, velocity/momentum, internal/total energy, [transverse magnetic field])
struct HydPrim1D {
  Real d, vx, vy, vz, e;
};
struct HydCons1D {
  Real d, mx, my, mz, e;
};
struct MHDPrim1D {
  Real d, vx, vy, vz, e, bx, by, bz;
};
struct MHDCons1D {
  Real d, mx, my, mz, e, bx, by, bz;
};

struct EOS_Data {
  Real gamma;        // ratio of specific heats for ideal gas
  Real dfloor, pfloor, tfloor, sfloor;  // density, pressure, temperature, entropy floors
};

//----------------------------------------------------------------------------------------
//! \fn Real equation_49()
//! \brief Inline function to compute function fa(mu) defined in eq. 49 of Kastaun et al.
//! The root fa(mu)==0 of this function corresponds to the upper bracket for
//! solving equation_44
KOKKOS_INLINE_FUNCTION
Real equation_49(const Real mu, const Real b2, const Real rp, const Real r, const Real q)
{
  Real const x = 1.0/(1.0 + mu*b2);             // (26)
  Real rbar = (x*x*r*r + mu*x*(1.0 + x)*rp*rp); // (38)
  return mu*m::sqrt(1.0 + rbar) - 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real equation_44()
//! \brief Inline function to compute function f(mu) defined in eq. 44 of Kastaun et al.
//! The ConsToPrim algorithms finds the root of this function f(mu)=0
KOKKOS_INLINE_FUNCTION
Real equation_44(const Real mu, const Real b2, const Real rpar, const Real r, const Real q,
                const Real u_d,  EOS_Data eos)
{
  Real const x = 1./(1.+mu*b2);                    // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);   // (38)
  Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar- rpar*rpar)); // (31)
  Real z2 = (mu*mu*rbar/(m::abs(1.- SQR(mu)*rbar))); // (32)
  Real w = m::sqrt(1.+z2);
  Real const wd = u_d/w;                           // (34)
  Real eps = w*(qbar - mu*rbar) + z2/(w+1.);
  Real const gm1 = eos.gamma - 1.0;
  Real epsmin = m::max(eos.pfloor/(wd*gm1), eos.sfloor*pow(wd, gm1)/gm1);
  eps = m::max(eps, epsmin);
  Real const h = 1.0 + eos.gamma*eps;              // (43)
  return mu - 1./(h/w + rbar*mu);                  // (45)
}




//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRMHD()
//! \brief Converts single state of conserved variables into primitive variables for
//! special relativistic MHD with an ideal gas EOS. Note input CONSERVED state contains
//! cell-centered magnetic fields, but PRIMITIVE state returned via arguments does not.
KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRMHD(MHDCons1D &u, const EOS_Data &eos, Real s2, Real b2, Real rpar,
                          HydPrim1D &w, bool &dfloor_used, bool &efloor_used,
                          bool &c2p_failure, int &max_iter)
{
    // Parameters
    const int max_iterations = 25;
    const Real tol = 1.0e-12;
    const Real gm1 = eos.gamma - 1.0;

    // apply density floor, without changing momentum or energy
    if (u.d < eos.dfloor) {
        u.d = eos.dfloor;
        dfloor_used = true;
    }

    // apply energy floor
    if (u.e < (eos.pfloor/gm1 + 0.5*b2)) {
        u.e = eos.pfloor/gm1 + 0.5*b2;
        efloor_used = true;
    }

    // Recast all variables (eq 22-24)
    Real q = u.e/u.d;
    Real r = m::sqrt(s2)/u.d;
    Real isqrtd = 1.0/m::sqrt(u.d);
    Real bx = u.bx*isqrtd;
    Real by = u.by*isqrtd;
    Real bz = u.bz*isqrtd;

    // normalize b2 and rpar as well since they contain b
    b2 /= u.d;
    rpar *= isqrtd;

    // Need to find initial bracket. Requires separate solve
    Real zm = 0.;
    Real zp = 1.; // This is the lowest specific enthalpy admitted by the EOS

    // Evaluate master function (eq 49) at bracket values
    Real fm = equation_49(zm, b2, rpar, r, q);
    Real fp = equation_49(zp, b2, rpar, r, q);

    // For simplicity on the GPU, find roots using the false position method
    int iterations = max_iterations;
    // If bracket within tolerances, don't bother doing any iterations
    if ((m::abs(zm-zp) < tol) || ((m::abs(fm) + m::abs(fp)) < 2.0*tol)) {
        iterations = -1;
    }
    Real z = 0.5*(zm + zp);

    int iter;
    for (iter=0; iter<iterations; ++iter) {
        z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
        Real f = equation_49(z, b2, rpar, r, q);
        // Quit if convergence reached
        // NOTE(@ermost): both z and f are of order unity
        if ((m::abs(zm-zp) < tol) || (m::abs(f) < tol)) {
            break;
        }
        // assign zm-->zp if root bracketed by [z,zp]
        if (f*fp < 0.0) {
            zm = zp;
            fm = fp;
            zp = z;
            fp = f;
        } else {  // assign zp-->z if root bracketed by [zm,z]
            fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
            zp = z;
            fp = f;
        }
    }
    max_iter = (iter > max_iter) ? iter : max_iter;

    // Found brackets. Now find solution in bounded interval, again using the
    // false position method
    zm = 0.;
    zp = z;

    // Evaluate master function (eq 44) at bracket values
    fm = equation_44(zm, b2, rpar, r, q, u.d, eos);
    fp = equation_44(zp, b2, rpar, r, q, u.d, eos);

    iterations = max_iterations;
    if ((m::abs(zm-zp) < tol) || ((m::abs(fm) + m::abs(fp)) < 2.0*tol)) {
        iterations = -1;
    }
    z = 0.5*(zm + zp);

    for (iter=0; iter<iterations; ++iter) {
        z = (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
        Real f = equation_44(z, b2, rpar, r, q, u.d, eos);
        // Quit if convergence reached
        // NOTE: both z and f are of order unity
        if ((m::abs(zm-zp) < tol) || (m::abs(f) < tol)) {
            break;
        }
        // assign zm-->zp if root bracketed by [z,zp]
        if (f*fp < 0.0) {
            zm = zp;
            fm = fp;
            zp = z;
            fp = f;
        } else {  // assign zp-->z if root bracketed by [zm,z]
            fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
            zp = z;
            fp = f;
        }
    }
    max_iter = (iter > max_iter) ? iter : max_iter;

    // check if convergence is established within max_iterations.  If not, trigger a C2P
    // failure and return floored density, pressure, and primitive velocities. Future
    // development may trigger averaging of (successfully inverted) neighbors in the event
    // of a C2P failure.
    if (max_iter == max_iterations) {
        w.d = eos.dfloor;
        w.e = eos.pfloor/gm1;
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        c2p_failure = true;
        return;
    }

    // iterations ended, compute primitives from resulting value of z
    Real &mu = z;
    Real const x = 1./(1.+mu*b2);                               // (26)
    Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);              // (38)
    Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar - rpar*rpar)); // (31)
    Real z2 = (mu*mu*rbar/(m::abs(1.- SQR(mu)*rbar)));            // (32)
    Real lor = m::sqrt(1.0 + z2);

    // compute density then apply floor
    Real dens = u.d/lor;
    if (dens < eos.dfloor) {
        dens = eos.dfloor;
        dfloor_used = true;
    }

    // compute specific internal energy density then apply floors
    Real eps = lor*(qbar - mu*rbar) + z2/(lor + 1.0);
    Real epsmin = m::max(eos.pfloor/(dens*gm1), eos.sfloor*pow(dens, gm1)/gm1);
    if (eps <= epsmin) {
        eps = epsmin;
        efloor_used = true;
    }

    // set parameters required for velocity inversion
    Real const h = 1.0 + eos.gamma*eps;  // (43)
    Real const conv = lor/(h*lor + b2);  // (C26)

    // set primitive variables
    w.d  = dens;
    w.vx = conv*(u.mx/u.d + bx*rpar/(h*lor));  // (C26)
    w.vy = conv*(u.my/u.d + by*rpar/(h*lor));  // (C26)
    w.vz = conv*(u.mz/u.d + bz*rpar/(h*lor));  // (C26)
    w.e  = dens*eps;
}

//----------------------------------------------------------------------------------------
//! \fn void TransformToSRMHD()
//! \brief Converts single state of conserved variables in GR MHD into conserved
//! variables for special relativistic MHD with an ideal gas EOS. This allows
//! the ConsToPrim() function in GR MHD to use SingleP2C_IdealSRMHD() function.

KOKKOS_INLINE_FUNCTION
void TransformToSRMHD(const MHDCons1D &u, Real glower[][4], Real gupper[][4],
                      Real &s2, Real &b2, Real &rpar, MHDCons1D &u_sr) {
    // Need to multiply the conserved density by alpha, so that it
    // contains a lorentz factor
    Real alpha = m::sqrt(-1.0/gupper[0][0]);
    u_sr.d = u.d*alpha;

    // We are evolving T^t_t, but the SR C2P algorithm is only consistent with
    // alpha^2 T^{tt}.  Therefore compute T^{tt} = g^0\mu T^t_\mu
    // We are also evolving T^t_t + D as conserved variable, so must convert to E
    u_sr.e = gupper[0][0]*(u.e - u.d) +
            gupper[0][1]*u.mx + gupper[0][2]*u.my + gupper[0][3]*u.mz;

    // This is only true if m::sqrt{-g}=1!
    u_sr.e *= (-1./gupper[0][0]);  // Multiply by alpha^2

    // Subtract density for consistency with the rest of the algorithm
    u_sr.e -= u_sr.d;

    // Need to treat the conserved momenta. Also they lack an alpha
    // This is only true if m::sqrt{-g}=1!
    Real m1l = u.mx*alpha;
    Real m2l = u.my*alpha;
    Real m3l = u.mz*alpha;

    // Need to raise indices on u_m1, which transforms using the spatial 3-metric.
    // Store in u_sr.  This is slightly more involved
    //
    // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
    //       g^0i = beta^i/alpha^2
    //       g^00 = -1/ alpha^2
    // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
    u_sr.mx = ((gupper[1][1] - gupper[0][1]*gupper[0][1]/gupper[0][0])*m1l +
                (gupper[1][2] - gupper[0][1]*gupper[0][2]/gupper[0][0])*m2l +
                (gupper[1][3] - gupper[0][1]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

    u_sr.my = ((gupper[2][1] - gupper[0][2]*gupper[0][1]/gupper[0][0])*m1l +
                (gupper[2][2] - gupper[0][2]*gupper[0][2]/gupper[0][0])*m2l +
                (gupper[2][3] - gupper[0][2]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

    u_sr.mz = ((gupper[3][1] - gupper[0][3]*gupper[0][1]/gupper[0][0])*m1l +
                (gupper[3][2] - gupper[0][3]*gupper[0][2]/gupper[0][0])*m2l +
                (gupper[3][3] - gupper[0][3]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

    // Compute (S^i S_i) (eqn C2)
    s2 = (m1l*u_sr.mx) + (m2l*u_sr.my) + (m3l*u_sr.mz);

    // load magnetic fields into SR conserved state. Also they lack an alpha
    // This is only true if m::sqrt{-g}=1!
    u_sr.bx = alpha*u.bx;
    u_sr.by = alpha*u.by;
    u_sr.bz = alpha*u.bz;

    b2 = glower[1][1]*SQR(u_sr.bx) + glower[2][2]*SQR(u_sr.by) + glower[3][3]*SQR(u_sr.bz) +
        2.0*(u_sr.bx*(glower[1][2]*u_sr.by + glower[1][3]*u_sr.bz) +
                        glower[2][3]*u_sr.by*u_sr.bz);
    rpar = (u_sr.bx*m1l +  u_sr.by*m2l +  u_sr.bz*m3l)/u_sr.d;
}

/**
 * Robust inversion scheme from Kastaun et al. 2020 (by way of AthenaK, obviously)
 */
template <>
KOKKOS_INLINE_FUNCTION int u_to_p<Type::kastaun>(const GRCoordinates &G, const VariablePack<Real>& U, const VarMap& m_u,
                                              const Real& gam, const int& k, const int& j, const int& i,
                                              const VariablePack<Real>& P, const VarMap& m_p,
                                              const Loci loc)
{
    // Prep
    MHDCons1D u = {U(m_u.RHO, k, j, i), U(m_u.U1, k, j, i), U(m_u.U2, k, j, i), U(m_u.U3, k, j, i),
                   U(m_u.UU, k, j, i), U(m_u.B1, k, j, i), U(m_u.B2, k, j, i), U(m_u.B3, k, j, i)};
    GReal gcon[GR_DIM][GR_DIM], gcov[GR_DIM][GR_DIM];
    G.gcon(Loci::center, j, i, gcon);
    G.gcov(Loci::center, j, i, gcov);
    // Output params
    Real s2, b2, rpar;
    MHDCons1D u_sr;
    // TODO(BSP) skip this call in Minkowski, calculate s2/b2/rpar separately 
    TransformToSRMHD(u, gcov, gcon, s2, b2, rpar, u_sr);

    EOS_Data eos;
    eos.gamma = gam;
    eos.dfloor = 1.e-10; // density
    eos.pfloor = 1.e-10; // pressure
    eos.tfloor = 0.; // temperature
    eos.sfloor = 0.; // entropy

    HydPrim1D w;
    bool dfloor_used = false, efloor_used = false, c2p_failure = false;
    int max_iter;
    SingleC2P_IdealSRMHD(u_sr, eos, s2, b2, rpar,
                          w, dfloor_used, efloor_used,
                          c2p_failure, max_iter);
    P(m_p.RHO, k, j, i) = w.d;
    P(m_p.UU, k, j, i) = w.e;
    P(m_p.U1, k, j, i) = w.vx;
    P(m_p.U2, k, j, i) = w.vy;
    P(m_p.U3, k, j, i) = w.vz;
    int flag = 0;
    if (dfloor_used) flag |= Floors::FFlag::GEOM_RHO;
    if (efloor_used) flag |= Floors::FFlag::GEOM_U;
    if (c2p_failure) flag += static_cast<int>(Status::max_iter);
    return flag;
}

}
