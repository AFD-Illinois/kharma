/* 
 *  File: coordinate_systems.hpp
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

// See note in coordintate_embedding
#include <mpark/variant.hpp>
//#include <variant>
//namespace mpark = std;

#include "decs.hpp"

#include "matrix.hpp"
#include "coordinate_utils.hpp"
#include "kharma_utils.hpp"
#include "root_find.hpp"

#define LEGACY_TH 1

/**
 * Embedding/Base systems implemented:
 * Minkowski space: Cartesian and Spherical coordinates
 * Kerr Space: Spherical KS and BL coordinates
 * 
 * Transformations:
 * Nulls in Cartesian and Spherical coordinates
 * "Modified": r=exp(x1), th=pi*x2 + (1-hslope)*etc
 * "Funky" modified: additional non-invertible cylindrization of th
 * 
 * TODO Cartesian KS base
 * TODO snake coordinate transform for Cartesian Minkowski
 * TODO CMKS, MKS3 transforms, proper Cartesian<->Spherical functions stolen from e.g. coordinate_utils.hpp
 * TODO overhaul the LEGACY_TH stuff
 * TODO currently avoids returning gcov which might be singular,
 *      is this the correct play vs handling in inversions?
 */

/**
 * EMBEDDING SYSTEMS:
 * These are the usual systems of coordinates for different spacetimes.
 * Each system/class must define at least gcov_embed, returning the metric in terms of their own coordinates Xembed
 * Some extra convenience classes have been defined for some systems.
 */
// ____________________________________________________________________________________________________________________________

/**
 * Cartesian Coordinates over flat space.
 */
class CartMinkowskiCoords {
    public:
        static constexpr char name[] = "CartMinkowskiCoords";
        static constexpr bool spherical = false;
        static constexpr GReal a = 0.0;
        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            DLOOP2 gcov[mu][nu] = (mu == nu) - 2*(mu == 0 && nu == 0);
        }
};
// ____________________________________________________________________________________________________________________________

/**
 * Spherical coordinates for flat space
 */
class SphMinkowskiCoords {
    public:
        static constexpr char name[] = "SphMinkowskiCoords";
        static constexpr bool spherical = true;
        static constexpr GReal a = 0.0;
        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = m::max(Xembed[1], SMALL);
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            const GReal sth = m::sin(th);

            gzero2(gcov);
            gcov[0][0] = 1.;
            gcov[1][1] = 1.;
            gcov[2][2] = r*r;
            gcov[3][3] = sth*sth*r*r;
        }
};
// ____________________________________________________________________________________________________________________________

/**
 * Spherical Kerr-Schild coordinates
 */
class SphKSCoords {
    public:
        static constexpr char name[] = "SphKSCoords";
        // BH Spin is a property of KS
        const GReal a;
        static constexpr bool spherical = true;

        KOKKOS_FUNCTION SphKSCoords(GReal spin): a(spin) {};

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal sin2 = sth*sth;
            const GReal rho2 = r*r + a*a*cth*cth;

            gcov[0][0] = -1. + 2.*r/rho2;
            gcov[0][1] = 2.*r/rho2;
            gcov[0][2] = 0.;
            gcov[0][3] = -2.*a*r*sin2/rho2;

            gcov[1][0] = 2.*r/rho2;
            gcov[1][1] = 1. + 2.*r/rho2;
            gcov[1][2] = 0.;
            gcov[1][3] = -a*sin2*(1. + 2.*r/rho2);

            gcov[2][0] = 0.;
            gcov[2][1] = 0.;
            gcov[2][2] = rho2;
            gcov[2][3] = 0.;

            gcov[3][0] = -2.*a*r*sin2/rho2;
            gcov[3][1] = -a*sin2*(1. + 2.*r/rho2);
            gcov[3][2] = 0.;
            gcov[3][3] = sin2*(rho2 + a*a*sin2*(1. + 2.*r/rho2));
        }

        // For converting from BL
        KOKKOS_INLINE_FUNCTION void vec_from_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
        {
            GReal r = Xembed[1];
            Real trans[GR_DIM][GR_DIM];
            DLOOP2 trans[mu][nu] = (mu == nu);
            trans[0][1] = 2.*r/(r*r - 2.*r + a*a);
            trans[3][1] = a/(r*r - 2.*r + a*a);

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
        }

        KOKKOS_INLINE_FUNCTION void vec_to_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
        {
            GReal r = Xembed[1];
            GReal rtrans[GR_DIM][GR_DIM], trans[GR_DIM][GR_DIM];
            DLOOP2 rtrans[mu][nu] = (mu == nu);
            rtrans[0][1] = 2.*r/(r*r - 2.*r + a*a);
            rtrans[3][1] = a/(r*r - 2.*r + a*a);

            invert(&rtrans[0][0], &trans[0][0]);

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
        }
};
// ____________________________________________________________________________________________________________________________
/**
 * Spherical Kerr-Schild coordinates w/ external gravity term
 */
class SphKSExtG {
    public:
        static constexpr char name[] = "SphKSExtG";
        // BH Spin is a property of KS
        const GReal a;
        static constexpr bool spherical = true;

        static constexpr GReal A = 4.24621057e-9; //1.46797639e-8;
        static constexpr GReal B = 1.35721335; //1.29411117;

        KOKKOS_FUNCTION SphKSExtG(GReal spin): a(spin) {};

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal sin2 = sth*sth;
            const GReal rho2 = r*r + a*a*cth*cth;

            const GReal Phi_g = (A / (B - 1.)) * (m::pow(r, B-1.) - m::pow(2, B-1.));

            gcov[0][0] = -1. + 2.*r/rho2 - 2. * Phi_g;
            gcov[0][1] = 2.*r/rho2 - 2. * Phi_g;
            gcov[0][2] = 0.;
            gcov[0][3] = -2.*a*r*sin2/rho2;

            gcov[1][0] = 2.*r/rho2 - 2. * Phi_g;
            gcov[1][1] = 1. + 2.*r/rho2 - 2. * Phi_g;
            gcov[1][2] = 0.;
            gcov[1][3] = -a*sin2*(1. + 2.*r/rho2);

            gcov[2][0] = 0.;
            gcov[2][1] = 0.;
            gcov[2][2] = rho2;
            gcov[2][3] = 0.;

            gcov[3][0] = -2.*a*r*sin2/rho2;
            gcov[3][1] = -a*sin2*(1. + 2.*r/rho2);
            gcov[3][2] = 0.;
            gcov[3][3] = sin2*(rho2 + a*a*sin2*(1. + 2.*r/rho2));
        }

        // For converting from BL
        // TODO will we ever need a from_ks?
        KOKKOS_INLINE_FUNCTION void vec_from_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
        {
            GReal r = Xembed[1];
            Real trans[GR_DIM][GR_DIM];
            DLOOP2 trans[mu][nu] = (mu == nu);

            // external gravity from GIZMO
            const GReal Phi_g = (A/(B-1.)) * (m::pow(r,B-1.)-m::pow(2,B-1.));

            trans[0][1] = (2./r - 2.*Phi_g)/(1. - 2./r + 2.*Phi_g);
            trans[3][1] = a/(r*r - 2.*r + a*a);

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
        }

        KOKKOS_INLINE_FUNCTION void vec_to_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
        {
            GReal r = Xembed[1];
            GReal rtrans[GR_DIM][GR_DIM], trans[GR_DIM][GR_DIM];
            DLOOP2 rtrans[mu][nu] = (mu == nu);

            const GReal Phi_g = (A / (B-1.)) * (m::pow(r, B-1.) - m::pow(2, B-1.));

            rtrans[0][1] = (2./r - 2.*Phi_g)/(1. - 2./r + 2.*Phi_g);
            rtrans[3][1] = a/(r*r - 2.*r + a*a);

            invert(&rtrans[0][0], &trans[0][0]);

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
        }
};

// ____________________________________________________________________________________________________________________________

/**
 * Boyer-Lindquist coordinates as an embedding system
 */
class SphBLCoords {
    public:
        static constexpr char name[] = "SphBLCoords";
        // BH Spin is a property of BL
        const GReal a;
        static constexpr bool spherical = true;

        KOKKOS_FUNCTION SphBLCoords(GReal spin): a(spin) {}

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            const GReal cth = m::cos(th), sth = m::sin(th);

            const GReal sin2 = sth*sth;
            const GReal a2 = a*a;
            const GReal r2 = r*r;
            // TODO(BSP) this and gcov_embed for KS should look more similar...
            const GReal mmu = 1. + a2*cth*cth/r2; // mu is taken as an index

            gzero2(gcov);
            gcov[0][0]  = -(1. - 2./(r*mmu));
            gcov[0][3]  = -2.*a*sin2/(r*mmu);
            gcov[1][1]   = mmu/(1. - 2./r + a2/r2);
            gcov[2][2]   = r2*mmu;
            gcov[3][0]  = -2.*a*sin2/(r*mmu);
            gcov[3][3]   = sin2*(r2 + a2 + 2.*a2*sin2/(r*mmu));
        }

        // TODO(BSP) vec to/from ks, put guaranteed ks/bl fns into embedding

};
// ____________________________________________________________________________________________________________________________

/**
 * Boyer-Lindquist coordinates as an embedding system
 */
class SphBLExtG {
    public:
        static constexpr char name[] = "SphBLExtG";
        // BH Spin is a property of BL
        const GReal a;
        static constexpr bool spherical = true;

        static constexpr GReal A = 4.24621057e-9; //1.46797639e-8;
        static constexpr GReal B = 1.35721335; //1.29411117;

        KOKKOS_FUNCTION SphBLExtG(GReal spin): a(spin) {}

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            const GReal cth = m::cos(th), sth = m::sin(th);

            const GReal sin2 = sth*sth;
            const GReal a2 = a*a;
            const GReal r2 = r*r;
            const GReal mmu = 1. + a2*cth*cth/r2; // mu is taken as an index

            const GReal Phi_g = (A / (B-1.)) * (m::pow(r, B-1.) - m::pow(2, B-1.));

            gzero2(gcov);
            gcov[0][0]  = -(1. - 2./(r*mmu)) - 2. * Phi_g;;
            gcov[0][3]  = -2.*a*sin2/(r*mmu);
            gcov[1][1]   = mmu / (1. - 2./r + 2.*Phi_g);
            gcov[2][2]   = r2*mmu;
            gcov[3][0]  = -2.*a*sin2/(r*mmu);
            gcov[3][3]   = sin2*(r2 + a2 + 2.*a2*sin2/(r*mmu));
        }
}; 
// ____________________________________________________________________________________________________________________________
/**
 * Changes Made. DCS theory KS coordinates. 
 */
class DCSKSCoords {
    public:
        static constexpr char name[] = "DCSKSCoords";
        // BH Spin is a property of KS
        const GReal a;
        const GReal zeta;
        static constexpr bool spherical = true;

        KOKKOS_FUNCTION DCSKSCoords(GReal spin, GReal z): a(spin), zeta(z) {}

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            
            // Assign gcov matrix to zero. 
            gzero2(gcov);

            // Compute Kerr metric with new "radial" coordinate
            GReal r_corr;
            radius_correction_dcs(a, zeta, r, th, r_corr);
            GReal gcov_kerr[GR_DIM][GR_DIM] = {0};
            gcov_kerr_gr(r_corr, th, a, gcov_kerr);

            // Compute correction and take exponential
            GReal metric_corr[GR_DIM][GR_DIM] = {0};
            GReal metric_corr_trans[GR_DIM][GR_DIM] = {0};
            metric_correction_dcs(a, zeta, r, th, metric_corr);
            transpose_matrix(metric_corr, metric_corr_trans);
            GReal exp_metric_corr[GR_DIM][GR_DIM] = {0};
            GReal exp_metric_corr_trans[GR_DIM][GR_DIM] = {0};
            exp_taylor_series_second_order(metric_corr, exp_metric_corr);
            exp_taylor_series_second_order(metric_corr_trans, exp_metric_corr_trans);

            // Matrix multiply to obtain metric
            GReal temp_matrix_mul[GR_DIM][GR_DIM] = {0};
            matrix_multiply(gcov_kerr, exp_metric_corr, temp_matrix_mul);
            matrix_multiply(exp_metric_corr_trans, temp_matrix_mul, gcov);
        }

    // New Transformation matrix here !!! 

        KOKKOS_INLINE_FUNCTION void vec_from_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
        {
            GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            Real trans[GR_DIM][GR_DIM];
            DLOOP2 trans[mu][nu] = (mu == nu);

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal a2  = a * a;
            const GReal a3  = a * a2;
            const GReal a4  = a2 * a2;
            const GReal a5  = a * a4;

            trans[0][0] = 1;
            trans[0][1] = (2 * r) / ((-2 + r) * r + a2) + 
                        (((1824966 - 623743*r) * a2) / (192192 * m::pow((-2 + r),2) * r) + 
                        ((-70735682160 - 11191562400 * r + 47551988940 * m::pow(r,2) - 24050897266 * m::pow(r,3) + 
                        3159486233 * m::pow(r,4)) * a4) / (3724680960 * m::pow((-2 + r),3) * m::pow(r,3))) * zeta;
            trans[0][2] = 0;
            trans[0][3] = 0;

            trans[1][0] = 0 ;
            trans[1][1] = 1 ;
            trans[1][2] = 0 ;
            trans[1][3] = 0 ;

            trans[2][0] = 0 ;
            trans[2][1] = 0 ;
            trans[2][2] = 1 ;
            trans[2][3] = 0 ;

            trans[3][0] = 0;
            trans[3][1] =  a / ((-2 + r) * r + a2) + 
                      ((-709 * a) / (224 * (-2 + r) * m::pow(r,2)) + 
                      ((1216644 - 254563 * r + 119571 * m::pow(r,2)) * a3)/
                      (192192 * m::pow((-2 + r),2) * m::pow(r,3)) + ((-70735682160 + 19733723760 * r - 
                      13180164712 * m::pow(r,2) + 7112046966 * m::pow(r,3) - 2578268285 * m::pow(r,4)) * a5)/
                      (7449361920 * m::pow((-2 + r),3) * m::pow(r,4))) * zeta;
            trans[3][2] = 0;
            trans[3][3] = 1;

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
        }
};

// ____________________________________________________________________________________________________________________________

/**
 * Changes Made. DCS theory BL coordinates. 
 */
class DCSBLCoords {
    public:
        static constexpr char name[] = "DCSBLCoords";

        // BH Spin is a property of KS
        const GReal a;
        const GReal zeta;
        static constexpr bool spherical = true;

        KOKKOS_FUNCTION DCSBLCoords(GReal spin, GReal z): a(spin), zeta(z) {}

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            
            // Assign gcov matrix to zero.
            gzero2(gcov); 

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal s2t = sth*sth;
            const GReal c2t = cth*cth; 
            const GReal c4t = c2t*c2t;
            // const GReal rho2 = r*r + a*a*cth*cth;
            const GReal a2 = a * a;
            const GReal a3 = a * a2 ;
            const GReal a4 = a2 * a2;
            const GReal a5 = a * a4 ;
            // const GReal DD = 1. - 2./r + ep2/(r*r);
            // const GReal mu = 1. + ep2*cth*cth/(r*r);

            gcov[0][0] = -1 + (2*r)/(m::pow(r,2) + c2t*a2) + 
     (((57513456*c2t - 2376*r*(2252 + 1031*c2t) + 358*m::pow(r,4)*(72 + 1667*c2t) - 
          240*m::pow(r,2)*(12881 + 13496*c2t) + 7*m::pow(r,5)*(-542 + 80291*c2t) + 
          7*m::pow(r,6)*(-542 + 80291*c2t) - 4*m::pow(r,3)*(399191 + 1094689*c2t))*
         a2)/(168168*m::pow(r,9)) + 
       ((-6725487328560*c4t - 6*m::pow(r,7)*(-65706777 + 482059160*c2t) - 
          6*m::pow(r,8)*(-65706777 + 482059160*c2t) + 880*r*c2t*(457031182 + 
            705483437*c2t) + m::pow(r,4)*(6697024816 - 7899576234*c2t - 
            20900576036*c4t) + m::pow(r,6)*(637514559 - 6751780650*c2t - 
            11301772795*c4t) - 3*m::pow(r,5)*(-1189461411 + 645503803*c2t + 
            3617103760*c4t) + 8*m::pow(r,2)*(859910810 + 17173037420*c2t + 
            53803121793*c4t) + 4*m::pow(r,3)*(2145558371 + 11748411201*c2t + 
            75381202555*c4t))*a4)/(2444321880*m::pow(r,11)))*zeta;

            gcov[0][3] = (-2*r*(1 - c2t)*a)/(m::pow(r,2) + c2t*a2) + 
     (-1/7*((189 + 120*r + 70*m::pow(r,2))*(-1 + c2t)*a)/m::pow(r,6) + 
       ((-1 + c2t)*(3450807360*c2t + m::pow(r,5)*(17299155 - 49653375*c2t) + 
          47520*r*(-6756 + 59879*c2t) - 104*m::pow(r,3)*(564161 + 173915*c2t) + 
          6*m::pow(r,6)*(457841 + 2310145*c2t) - 30*m::pow(r,4)*(-1082725 + 5558373*c2t) + 
          32*m::pow(r,2)*(-4927439 + 28914115*c2t))*a3)/(10090080*m::pow(r,9)) - 
       ((-1 + c2t)*(1506509161597440*c4t + 4040960*r*c2t*
           (-22294204 + 173679445*c2t) + 2*m::pow(r,8)*(-80412903443 + 
            588521457535*c2t) + m::pow(r,7)*(-474922692703 + 1322888510075*c2t) - 
          448*m::pow(r,3)*(5280220585 - 6953291298*c2t + 117686275521*c4t) + 
          6*m::pow(r,6)*(-136614673432 + 395122411427*c2t + 132060794257*c4t) + 
          896*m::pow(r,2)*(-1719821620 - 19785006773*c2t + 140513803419*c4t) - 
          28*m::pow(r,5)*(58696057625 - 148765679968*c2t + 292875133523*c4t) - 
          112*m::pow(r,4)*(20394874372 - 92750674255*c2t + 294800362507*c4t))*
         a5)/(547528101120*m::pow(r,11)))*zeta;

            gcov[1][1] =  (m::pow(r,2) + c2t*a2)/(-2*r + m::pow(r,2) + a2) + 
     (((66594528*c2t + m::pow(r,7)*(596015 - 686735*c2t) - 4752*r*(71 + 580*c2t) + 
          42*m::pow(r,5)*(-2558 + 15301*c2t) + 4*m::pow(r,4)*(-37036 + 57677*c2t) + 
          14*m::pow(r,6)*(-542 + 80291*c2t) - 48*m::pow(r,2)*(7513 + 90609*c2t) - 
          4*m::pow(r,3)*(100031 + 1624369*c2t))*a2)/(336336*m::pow((-2 + r),2)*
         m::pow(r,7)) - ((-23484015434880*c4t - 84*m::pow(r,10)*(-41239343 + 42306935*c2t) + 
          137280*r*c2t*(14110424 + 113722771*c2t) + 
          24*m::pow(r,9)*(-354382178 + 778207705*c2t) + 
          m::pow(r,7)*(383283900 + 71418009360*c2t - 27396512900*c4t) + 
          15*m::pow(r,8)*(2388050243 - 4049661346*c2t + 599177895*c4t) + 
          40*m::pow(r,6)*(-117290815 - 191181339*c2t + 635842970*c4t) + 
          352*m::pow(r,2)*(-33172623 - 378313744*c2t + 783039587*c4t) + 
          8*m::pow(r,5)*(-652016916 + 3067706031*c2t + 1704483638*c4t) + 
          16*m::pow(r,3)*(-899129902 - 4204697795*c2t + 3397074817*c4t) - 
          8*m::pow(r,4)*(2047359111 + 20418344123*c2t + 79181634110*c4t))*
         a4)/(9777287520*m::pow((-2 + r),3)*m::pow(r,9)))*zeta;

            gcov[2][2] = (m::pow(r,2) + c2t*a2)/(1 - c2t) + 
     (((33297264*c2t + 2376*r*(-71 + 6427*c2t) + 28*m::pow(r,5)*(-6209 + 8907*c2t) - 
          14*m::pow(r,4)*(17162 + 10275*c2t) + 35*m::pow(r,6)*(-2592 + 19621*c2t) - 
          8*m::pow(r,3)*(41549 + 64801*c2t) + 60*m::pow(r,2)*(-4411 + 91011*c2t))*
         a2)/(336336*m::pow(r,6)*(-1 + c2t)) + 
       ((-5871003858720*c4t - 84*m::pow(r,8)*(-1067592 + 42306935*c2t) - 
          34320*r*c2t*(-8567 + 57343775*c2t) + 72*m::pow(r,7)*(-19411211 + 
            61970205*c2t) + 15*m::pow(r,6)*(-142897849 + 397996074*c2t + 
            599177895*c4t) + 8*m::pow(r,5)*(-297446889 + 1924459035*c2t + 
            1069270100*c4t) - 176*m::pow(r,2)*(2654514 + 73677257*c2t + 
            2451022214*c4t) + 16*m::pow(r,3)*(-90179947 + 109284393*c2t + 
            4638623694*c4t) + 4*m::pow(r,4)*(-628224397 + 2843517045*c2t + 
            5924922075*c4t))*a4)/(9777287520*m::pow(r,8)*(-1 + c2t)))*
      zeta;

            gcov[3][0] = gcov[0][3]; 

            gcov[3][3] = (1 - c2t)*(m::pow(r,2) + a2 + (2*r*(1 - c2t)*a2)/
        (m::pow(r,2) + c2t*a2)) + 
     (((-1 + c2t)*(596015*m::pow(r,6) + m::pow(r,3)*(3866096 - 4716896*c2t) + 
          m::pow(r,4)*(2182824 - 2566942*c2t) + 3027024*(6 + 5*c2t) + 
          1584*r*(9299 + 235*c2t) - 120*m::pow(r,2)*(-90189 + 46889*c2t) - 
          14*m::pow(r,5)*(-85687 + 80291*c2t))*a2)/(336336*m::pow(r,6)) + 
       ((-1 + c2t)*(-866026203*m::pow(r,9) + 835958082960*c2t*(-1 + c2t) + 
          12*m::pow(r,8)*(-177191089 + 241029580*c2t) + 30*m::pow(r,7)*(-138133897 + 
            244918412*c2t) - 605880*r*(-128364 + 2462533*c2t + 88342*c4t) + 
          m::pow(r,6)*(-7864782813 + 1955574510*c2t + 11301772795*c4t) + 
          m::pow(r,5)*(-15610926579 - 16038673926*c2t + 39789815228*c4t) + 
          8*m::pow(r,2)*(3932616732 - 120339610907*c2t + 54914983595*c4t) + 
          4*m::pow(r,3)*(577408088 - 111691905294*c2t + 83313603371*c4t) + 
          m::pow(r,4)*(-21259108927 - 110710463772*c2t + 150600485259*c4t))*
         a4)/(2444321880*m::pow(r,9)))*zeta;

        }

};
// ____________________________________________________________________________________________________________________________
/**
 * Changes Made. EDGB theory KS coordinates. 
 */
class EDGBKSCoords {
    public:
        static constexpr char name[] = "EDGBKSCoords";

        // BH Spin is a property of KS
        const GReal a;
        const GReal zeta;
        static constexpr bool spherical = true;

        KOKKOS_FUNCTION EDGBKSCoords(GReal spin, GReal z): a(spin), zeta(z) {}

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            
            // Assign gcov matrix to zero. 
            gzero2(gcov);

            // Compute Kerr metric with new "radial" coordinate
            GReal r_corr;
            radius_correction_edgb(a, zeta, r, th, r_corr);
            GReal gcov_kerr[GR_DIM][GR_DIM] = {0};
            gcov_kerr_gr(r_corr, th, a, gcov_kerr);

            // Compute correction and take exponential
            GReal metric_corr[GR_DIM][GR_DIM] = {0};
            GReal metric_corr_trans[GR_DIM][GR_DIM] = {0};
            metric_correction_edgb(a, zeta, r, th, metric_corr);
            transpose_matrix(metric_corr, metric_corr_trans);
            GReal exp_metric_corr[GR_DIM][GR_DIM] = {0};
            GReal exp_metric_corr_trans[GR_DIM][GR_DIM] = {0};
            exp_taylor_series_second_order(metric_corr, exp_metric_corr);
            exp_taylor_series_second_order(metric_corr_trans, exp_metric_corr_trans);

            // Matrix multiply to obtain metric
            GReal temp_matrix_mul[GR_DIM][GR_DIM] = {0};
            matrix_multiply(gcov_kerr, exp_metric_corr, temp_matrix_mul);
            matrix_multiply(exp_metric_corr_trans, temp_matrix_mul, gcov);
        }

        KOKKOS_INLINE_FUNCTION void vec_from_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
        {
            GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            Real trans[GR_DIM][GR_DIM];
            DLOOP2 trans[mu][nu] = (mu == nu);

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal s2t = sth*sth;
            const GReal c2t = cth*cth; 
            const GReal c4t = c2t*c2t;
            const GReal a2  = a * a;
            const GReal a3  = a * a2 ;
            const GReal a4  = a2 * a2;
            const GReal a5  = a * a4;

            trans[0][0] = 1;
            trans[0][1] = (2*r)/((-2 + r)*r + a2) + 
                      ((11242 - 7855*r)/(2310*m::pow((-2 + r),2)*r) + 
                      ((-17537520 + 3485040*r + 30628260*m::pow(r,2) - 19496986*m::pow(r,3) + 3623153*m::pow(r,4))*
                      a2)/(1801800*m::pow((-2 + r),3)*m::pow(r,3)) + 
                      ((634437323520 - 1171045337280*r + 151318998656*m::pow(r,2) + 
                      1023918648648*m::pow(r,3) - 1045655284580*m::pow(r,4) + 378373431510*m::pow(r,5) - 
                      44447836077*m::pow(r,6))*a4)/(32590958400*m::pow((-2 + r),4)*m::pow(r,4)))*zeta;
            trans[0][2] = 0;
            trans[0][3] = 0;

            trans[1][0] = 0;
            trans[1][1] = 1; 
            trans[1][2] = 0;
            trans[1][3] = 0;


            trans[2][0] = 0 ;
            trans[2][1] = 0 ; 
            trans[2][2] = 1 ;
            trans[2][3] = 0 ;

   
            trans[3][0] = 0 ;
            trans[3][1] = a/((-2 + r)*r + a2) + 
                        (-1/4620*((3694 + 387*r)*a)/(m::pow((-2 + r),2)*m::pow(r,2)) + 
                        ((5762640 - 23759784*r + 27326618*m::pow(r,2) - 7572433*m::pow(r,3))*a3)/
                        (3603600*m::pow((-2 + r),3)*m::pow(r,3)) - ((156351948480 - 864993989664*r + 
                        633264793160*m::pow(r,2) + 245119239332*m::pow(r,3) - 252582370834*m::pow(r,4) + 
                        40939158983*m::pow(r,5))*a5)/(65181916800*m::pow((-2 + r),4)*m::pow(r,4)))*zeta; 
            trans[3][2] = 0 ; 
            trans[3][3] = 1 ;

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu]; // CHANGES MADE, (DID NOT INCLUDE THIS BEFORE I THINK)

        }
};

// ____________________________________________________________________________________________________________________________
/**
 * Changes Made. EDGB theory KS coordinates. 
 */

class EDGBBLCoords {
    public:
        static constexpr char name[] = "EDGBBLCoords";

        // BH Spin is a property of KS
        const GReal a;
        const GReal zeta;
        static constexpr bool spherical = true;

        KOKKOS_FUNCTION EDGBBLCoords(GReal spin, GReal z): a(spin), zeta(z) {}

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            
            // Assign gcov matrix to zero.
            gzero2(gcov); 

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal s2t = sth*sth;
            const GReal c2t = cth*cth; 
            const GReal c4t = c2t*c2t;
            // const GReal rho2 = r*r + a*a*cth*cth;
            const GReal a2 = a * a;
            const GReal a3 = a * a2 ;
            const GReal a4 = a2 * a2;
            const GReal a5 = a * a4 ;

            gcov[0][0] = -1 + (2*r)/(m::pow(r,2) + c2t*a2) + 
     ((43680 + 784*r + 428*m::pow(r,2) - 9606*m::pow(r,3) - 366*m::pow(r,4))/(1155*m::pow(r,7)) + 
       ((-4156807200*c2t + m::pow(r,4)*(2986194 - 22915196*c2t) + 
          4800*r*(-56803 + 158331*c2t) + 300*m::pow(r,2)*(-561266 + 1580411*c2t) - 
          7*m::pow(r,5)*(-2988737 + 4454096*c2t) - 7*m::pow(r,6)*(-736487 + 4454096*c2t) + 
          m::pow(r,3)*(-87538336 + 493638556*c2t))*a2)/(7882875*m::pow(r,9)) + 
       ((933554332838400*c4t - 537600*r*c2t*(-104697713 + 358584514*c2t) + 
          15*m::pow(r,8)*(7761861021 + 14872864789*c2t) + 
          15*m::pow(r,7)*(26773253421 + 14872864789*c2t) - 
          560*m::pow(r,2)*(-1047470079 - 39007247069*c2t + 161053631378*c4t) + 
          16*m::pow(r,4)*(30573921681 - 209341163160*c2t + 346098152155*c4t) + 
          4*m::pow(r,5)*(70591601799 - 1005064298479*c2t + 881660687734*c4t) + 
          2*m::pow(r,6)*(52374110171 - 302129028817*c2t + 1316162201140*c4t) - 
          8*m::pow(r,3)*(-83105751265 - 718589156631*c2t + 6738918795806*c4t))*
         a4)/(285170886000*m::pow(r,11)))*zeta;

            gcov[0][3] =  (-2*r*(1 - c2t)*a)/(m::pow(r,2) + c2t*a2) + 
     ((-2*(-21840 + 1456*r + 710*m::pow(r,2) + 5188*m::pow(r,3) + 337*m::pow(r,4))*(-1 + c2t)*
         a)/(1155*m::pow(r,7)) - ((-1 + c2t)*(8313614400*c2t + 
          m::pow(r,3)*(171933532 - 765585912*c2t) + 9600*r*(56803 + 120948*c2t) + 
          200*m::pow(r,2)*(1629458 + 740749*c2t) - m::pow(r,5)*(36396163 + 9768085*c2t) + 
          m::pow(r,6)*(-8132356 + 58450110*c2t) - 2*m::pow(r,4)*(280491 + 84874529*c2t))*
         a3)/(15765750*m::pow(r,9)) + 
       ((-1 + c2t)*(2800662998515200*c4t + 268800*r*c2t*(628186278 + 
            1772766269*c2t) - 5*m::pow(r,7)*(-244629458729 + 86745869725*c2t) + 
          5*m::pow(r,8)*(71324820365 + 99157435211*c2t) + 
          1680*m::pow(r,2)*(1047470079 + 38803012129*c2t + 22664337382*c4t) + 
          4*m::pow(r,5)*(251789024733 - 3601843302492*c2t + 256322116355*c4t) + 
          6*m::pow(r,6)*(63797540059 - 736067474880*c2t + 1272929793775*c4t) - 
          16*m::pow(r,4)*(-109479126093 + 612510146580*c2t + 1626768235846*c4t) - 
          48*m::pow(r,3)*(-46816941485 - 400319353593*c2t + 2738551436968*c4t))*
         a5)/(855512658000*m::pow(r,11)))*zeta;

            gcov[1][1] = (m::pow(r,2) + c2t*a2)/(-2*r + m::pow(r,2) + a2) + 
     ((-2*(-12880 - 1736*r - 1422*m::pow(r,2) + 2351*m::pow(r,3) + 183*m::pow(r,4) + 568*m::pow(r,5)))/
        (1155*m::pow((-2 + r),2)*m::pow(r,5)) + 
       ((4136932800*c2t + m::pow(r,3)*(5926276 - 276302056*c2t) + 
          35*m::pow(r,8)*(-160553 + 257957*c2t) - 600*m::pow(r,2)*(-34660 + 263379*c2t) - 
          2400*r*(46973 + 1057452*c2t) + 7*m::pow(r,6)*(1818260 + 5960689*c2t) - 
          7*m::pow(r,7)*(-2342017 + 7033666*c2t) + m::pow(r,5)*(-23745574 + 52450598*c2t) + 
          m::pow(r,4)*(22254558 + 213964066*c2t))*a2)/
        (7882875*m::pow((-2 + r),3)*m::pow(r,7)) + 
       ((1498497960960000*c4t - 15375360*r*c2t*(7561039 + 117640156*c2t) - 
          105*m::pow(r,11)*(-2788981959 + 1897271591*c2t) + 
          45*m::pow(r,10)*(-23443211277 + 22665489779*c2t) + 
          32*m::pow(r,3)*(-48726660415 + 12893807943*c2t + 35526074322*c4t) - 
          5*m::pow(r,9)*(-260432795909 + 362226361787*c2t + 52996871148*c4t) - 
          8*m::pow(r,7)*(-17641288689 + 212032217384*c2t + 93024858153*c4t) + 
          2240*m::pow(r,2)*(1367198769 + 35419031615*c2t + 237678689446*c4t) - 
          8*m::pow(r,6)*(-141794251563 - 314511311653*c2t + 306064626342*c4t) + 
          2*m::pow(r,8)*(-692054762131 + 903366440001*c2t + 690571902624*c4t) - 
          8*m::pow(r,5)*(71402844007 + 1108725427549*c2t + 2839916501006*c4t) + 
          16*m::pow(r,4)*(-38744321503 + 312954478079*c2t + 2896117766322*c4t))*
         a4)/(285170886000*m::pow((-2 + r),4)*m::pow(r,9)))*zeta;

            gcov[2][2] = (m::pow(r,2) + c2t*a2)/(1 - c2t) + 
     ((12880 + 8176*r + 5510*m::pow(r,2) + 404*m::pow(r,3) + 19*m::pow(r,4))/(1155*m::pow(r,4)*(-1 + c2t)) + 
       ((-2068466400*c2t - 2400*r*(13141 + 333135*c2t) - 
          35*m::pow(r,6)*(30417 + 515914*c2t) - 300*m::pow(r,2)*(179282 + 677979*c2t) + 
          28*m::pow(r,5)*(-444286 + 937263*c2t) + 14*m::pow(r,4)*(-2871703 + 5607480*c2t) + 
          8*m::pow(r,3)*(-7321036 + 16829791*c2t))*a2)/
        (15765750*m::pow(r,6)*(-1 + c2t)) + 
       ((187312245120000*c4t + 3843840*r*c2t*(1086283 + 14275672*c2t) + 
          105*m::pow(r,8)*(382476643 + 1897271591*c2t) + 
          15*m::pow(r,7)*(-2407188019 + 11688937485*c2t) + 
          15*m::pow(r,6)*(-16192726085 - 5105535869*c2t + 17665623716*c4t) + 
          8*m::pow(r,5)*(-44602451309 - 13151710900*c2t + 26095291149*c4t) + 
          280*m::pow(r,2)*(-453474711 + 16800931631*c2t + 29914099270*c4t) - 
          4*m::pow(r,4)*(98112524011 - 315706822671*c2t + 295759857126*c4t) - 
          8*m::pow(r,3)*(36415813535 - 407717884869*c2t + 629374083084*c4t))*
         a4)/(285170886000*m::pow(r,8)*(-1 + c2t)))*zeta;

            gcov[3][0] = gcov[0][3];

            gcov[3][3] = (1 - c2t)*(m::pow(r,2) + a2 + (2*r*(1 - c2t)*a2)/
        (m::pow(r,2) + c2t*a2)) + 
     (((12880 + 8176*r + 5510*m::pow(r,2) + 404*m::pow(r,3) + 19*m::pow(r,4))*(-1 + c2t))/
        (1155*m::pow(r,4)) - ((-1 + c2t)*(19121585*m::pow(r,7) + 
          m::pow(r,6)*(48553988 - 62357344*c2t) - 596232000*(-1 + c2t) + 
          1528800*r*(-59 + 1412*c2t) + 300*m::pow(r,3)*(223283 + 633978*c2t) + 
          600*m::pow(r,2)*(225321 + 1159783*c2t) - 14*m::pow(r,5)*(-7658151 + 10393928*c2t) - 
          4*m::pow(r,4)*(-44289389 + 63306899*c2t))*a2)/(15765750*m::pow(r,7)) + 
       ((-1 + c2t)*(239373564570*m::pow(r,9) + m::pow(r,8)*(362319213825 - 
            223092971835*c2t) - 150376657267200*c2t*(-1 + c2t) - 
          5*m::pow(r,7)*(-81200765027 + 92098679741*c2t) + 
          m::pow(r,5)*(15681624292 + 7220713321468*c2t - 7549057179624*c4t) + 
          9139200*r*(1079257 + 6525056*c2t + 12891162*c4t) - 
          11200*m::pow(r,2)*(-508763114 - 6368322995*c2t + 1604863153*c4t) - 
          8*m::pow(r,6)*(-57318630343 - 240063048882*c2t + 329040550285*c4t) - 
          56*m::pow(r,3)*(-56838174731 - 632301492673*c2t + 457831886454*c4t) - 
          8*m::pow(r,4)*(-58298717539 - 2527793068239*c2t + 2844163797528*c4t))*
         a4)/(285170886000*m::pow(r,9)))*zeta;

        }
};


// ____________________________________________________________________________________________________________________________
// ____________________________________________________________________________________________________________________________

/**
 * COORDINATE TRANSFORMS:
 * These are transformations which can be applied to base coordinates
 * Each class must define enough functions to apply the transform to coordinates and vectors,
 * both forward and in reverse.
 * That comes out to 4 functions: coord_to_embed, coord_to_native, dXdx, dxdX
 */

/**
 * This class represents a null transformation from the embedding cooridnates, i.e. just using them directly
 */
class NullTransform {
    public:
        static constexpr char name[] = "NullTransform";
        static constexpr GReal startx[3] = {-1, -1, -1};
        static constexpr GReal stopx[3] = {-1, -1, -1};
        // Coordinate transformations
        // Any coordinate value protections (th < 0, th > pi, phi > 2pi) should be in the base system
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            DLOOP1 Xembed[mu] = Xnative[mu];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
        {
            DLOOP1 Xnative[mu] = Xembed[mu];
        }
        // Tangent space transformation matrices
        KOKKOS_INLINE_FUNCTION void dxdX(const GReal X[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
        {
            DLOOP2 dxdX[mu][nu] = (mu == nu);
        }
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal X[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
        {
            DLOOP2 dXdx[mu][nu] = (mu == nu);
        }
};
// This only exists separately to define startx & stopx. Could fall back on base coords for these?
class SphNullTransform {
    public:
        static constexpr char name[] = "SphNullTransform";
        static constexpr GReal startx[3] = {-1, 0., 0.};
        static constexpr GReal stopx[3] = {-1, M_PI, 2*M_PI};
        // Coordinate transformations
        // Any coordinate value protections (th < 0, th > pi, phi > 2pi) should be in the base system
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            DLOOP1 Xembed[mu] = Xnative[mu];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
        {
            DLOOP1 Xnative[mu] = Xembed[mu];
        }
        // Tangent space transformation matrices
        KOKKOS_INLINE_FUNCTION void dxdX(const GReal X[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
        {
            DLOOP2 dxdX[mu][nu] = (mu == nu);
        }
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal X[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
        {
            DLOOP2 dXdx[mu][nu] = (mu == nu);
        }
};

/**
 * Just exponentiate the radial coordinate
 * Makes sense only for spherical base systems!
 */
class ExponentialTransform {
    public:
        static constexpr char name[] = "ExponentialTransform";
        static constexpr GReal startx[3] = {-1, 0., 0.};
        static constexpr GReal stopx[3] = {-1, M_PI, 2*M_PI};

        // Coordinate transformations
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            Xembed[0] = Xnative[0];
            Xembed[1] = m::exp(Xnative[1]);
#if LEGACY_TH
            Xembed[2] = excise(excise(Xnative[2], 0.0, SMALL), M_PI, SMALL);
#else
            Xembed[2] = Xnative[2];
#endif
            Xembed[3] = Xnative[3];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
        {
            Xnative[0] = Xembed[0];
            Xnative[1] = m::log(Xembed[1]);
            Xnative[2] = Xembed[2];
            Xnative[3] = Xembed[3];
        }
        /**
         * Transformation matrix for contravariant vectors to embedding, or covariant vectors to native
         */
        KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
        {
            gzero2(dxdX);
            dxdX[0][0] = 1.;
            dxdX[1][1] = m::exp(Xnative[1]);
            dxdX[2][2] = 1.;
            dxdX[3][3] = 1.;
        }
        /**
         * Transformation matrix for contravariant vectors to native, or covariant vectors to embedding
         */
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
        {
            gzero2(dXdx);
            dXdx[0][0] = 1.;
            dXdx[1][1] = 1 / m::exp(Xnative[1]);
            dXdx[2][2] = 1.;
            dXdx[3][3] = 1.;
        }
};

/**
 * SuperExponential coordinates, for super simulations
 * Implementation follows HARMPI described in Tchekhovskoy+
 */
class SuperExponentialTransform {
    public:
        static constexpr char name[] = "SuperExponentialTransform";
        static constexpr GReal startx[3] = {-1, 0., 0.};
        static constexpr GReal stopx[3] = {-1, M_PI, 2*M_PI};

        const GReal xe1br, xn1br;
        const double npow2, cpow2;

        // Constructor
        KOKKOS_FUNCTION SuperExponentialTransform(GReal xe1br_in, double npow2_in, double cpow2_in):
            xe1br(xe1br_in), npow2(npow2_in), cpow2(cpow2_in), xn1br(m::log(xe1br_in)) {}

        // Coordinate transformations
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            Xembed[0] = Xnative[0];
            const GReal super_dist = Xnative[1] - xn1br;
            Xembed[1] = m::exp(Xnative[1] + (super_dist > 0) * cpow2 * m::pow(super_dist, npow2));
#if LEGACY_TH
            Xembed[2] = excise(excise(Xnative[2], 0.0, SMALL), M_PI, SMALL);
#else
            Xembed[2] = Xnative[2];
#endif
            Xembed[3] = Xnative[3];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
        {
            Xnative[0] = Xembed[0];
            Xnative[2] = Xembed[2];
            Xnative[3] = Xembed[3];
            // TODO can just take log for x1 < xe1br
            ROOT_FIND_1
        }
        /**
         * Transformation matrix for contravariant vectors to embedding, or covariant vectors to native
         */
        KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
        {
            gzero2(dxdX);
            dxdX[0][0] = 1.;
            const GReal super_dist = Xnative[1] - xn1br;
            dxdX[1][1] = m::exp(Xnative[1] + (super_dist > 0) * cpow2 * m::pow(super_dist, npow2))
                            * (1 + (super_dist > 0) * cpow2 * npow2 * m::pow(super_dist, npow2-1));
            dxdX[2][2] = 1.;
            dxdX[3][3] = 1.;
        }
        /**
         * Transformation matrix for contravariant vectors to native, or covariant vectors to embedding
         */
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
        {
            gzero2(dXdx);
            dXdx[0][0] = 1.;
            const GReal super_dist = Xnative[1] - xn1br;
            dXdx[1][1] = 1 / (m::exp(Xnative[1] + (super_dist > 0) * cpow2 * m::pow(super_dist, npow2))
                              * (1 + (super_dist > 0) * cpow2 * npow2 * m::pow(super_dist, npow2-1)));
            dXdx[2][2] = 1.;
            dXdx[3][3] = 1.;
        }
};

/**
 * Modified Kerr-Schild coordinates "MKS"
 * Makes sense only for spherical base systems!
 */
class ModifyTransform {
    public:
        static constexpr char name[] = "ModifyTransform";
        static constexpr GReal startx[3] = {-1, 0., 0.};
        static constexpr GReal stopx[3] = {-1, 1., 2*M_PI};

        const GReal hslope;

        // Constructor
        KOKKOS_FUNCTION ModifyTransform(GReal hslope_in): hslope(hslope_in) {}

        // Coordinate transformations
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            Xembed[0] = Xnative[0];
            Xembed[1] = m::exp(Xnative[1]);
#if LEGACY_TH
            const GReal th = M_PI*Xnative[2] + ((1. - hslope)/2.)*m::sin(2.*M_PI*Xnative[2]);
            Xembed[2] = excise(excise(th, 0.0, SMALL), M_PI, SMALL);
#else
            Xembed[2] = M_PI*Xnative[2] + ((1. - hslope)/2.)*m::sin(2.*M_PI*Xnative[2]);
#endif
            Xembed[3] = Xnative[3];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
        {
            Xnative[0] = Xembed[0];
            Xnative[1] = m::log(Xembed[1]);
            Xnative[3] = Xembed[3];
            // Treat the special case with a large macro
            ROOT_FIND
        }
        /**
         * Transformation matrix for contravariant vectors to embedding, or covariant vectors to native
         */
        KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
        {
            gzero2(dxdX);
            dxdX[0][0] = 1.;
            dxdX[1][1] = m::exp(Xnative[1]);
            dxdX[2][2] = M_PI - (hslope - 1.)*M_PI*m::cos(2.*M_PI*Xnative[2]);
            dxdX[3][3] = 1.;
        }
        /**
         * Transformation matrix for contravariant vectors to native, or covariant vectors to embedding
         */
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
        {
            gzero2(dXdx);
            dXdx[0][0] = 1.;
            dXdx[1][1] = 1 / m::exp(Xnative[1]);
            dXdx[2][2] = 1 / (M_PI - (hslope - 1.)*M_PI*m::cos(2.*M_PI*Xnative[2]));
            dXdx[3][3] = 1.;
        }
};

/**
 * "Funky" Modified Kerr-Schild coordinates
 * Make sense only for spherical base systems!
 */
class FunkyTransform {
    public:
        static constexpr char name[] = "FunkyTransform";
        static constexpr GReal startx[3] = {-1, 0., 0.};
        static constexpr GReal stopx[3] = {-1, 1., 2*M_PI};

        const GReal startx1;
        const GReal hslope, poly_xt, poly_alpha, mks_smooth;
        // Must be *defined* afterward to use constructor below
        const GReal poly_norm;

        // Constructor
        KOKKOS_FUNCTION FunkyTransform(GReal startx1_in, GReal hslope_in, GReal mks_smooth_in, GReal poly_xt_in, GReal poly_alpha_in):
            startx1(startx1_in), hslope(hslope_in), mks_smooth(mks_smooth_in), poly_xt(poly_xt_in), poly_alpha(poly_alpha_in),
            poly_norm(0.5 * M_PI * 1./(1. + 1./(poly_alpha + 1.) * 1./m::pow(poly_xt, poly_alpha))) {}

        // Coordinate transformations
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            Xembed[0] = Xnative[0];
            Xembed[1] = m::exp(Xnative[1]);

            const GReal thG = M_PI*Xnative[2] + ((1. - hslope)/2.)*m::sin(2.*M_PI*Xnative[2]);
            const GReal y = 2*Xnative[2] - 1.;
            const GReal thJ = poly_norm * y * (1. + m::pow(y/poly_xt,poly_alpha) / (poly_alpha + 1.)) + 0.5 * M_PI;
#if LEGACY_TH
            const GReal th = thG + m::exp(mks_smooth * (startx1 - Xnative[1])) * (thJ - thG);
            Xembed[2] = excise(excise(th, 0.0, SMALL), M_PI, SMALL);
#else
            Xembed[2] = thG + m::exp(mks_smooth * (startx1 - Xnative[1])) * (thJ - thG);
#endif
            Xembed[3] = Xnative[3];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
        {
            Xnative[0] = Xembed[0];
            Xnative[1] = m::log(Xembed[1]);
            Xnative[3] = Xembed[3];
            // Treat the special case with a macro
            ROOT_FIND
        }
        /**
         * Transformation matrix for contravariant vectors to embedding, or covariant vectors to native
         */
        KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
        {
            gzero2(dxdX);
            dxdX[0][0] = 1.;
            dxdX[1][1] = m::exp(Xnative[1]);
            dxdX[2][1] = -exp(mks_smooth * (startx1 - Xnative[1])) * mks_smooth
                * (
                M_PI / 2. -
                M_PI * Xnative[2]
                    + poly_norm * (2. * Xnative[2] - 1.)
                        * (1
                            + (m::pow((-1. + 2 * Xnative[2]) / poly_xt, poly_alpha))
                                / (1 + poly_alpha))
                    - 1. / 2. * (1. - hslope) * m::sin(2. * M_PI * Xnative[2]));
            dxdX[2][2] = M_PI + (1. - hslope) * M_PI * m::cos(2. * M_PI * Xnative[2])
                + m::exp(mks_smooth * (startx1 - Xnative[1]))
                    * (-M_PI
                        + 2. * poly_norm
                            * (1.
                                + m::pow((2. * Xnative[2] - 1.) / poly_xt, poly_alpha)
                                    / (poly_alpha + 1.))
                        + (2. * poly_alpha * poly_norm * (2. * Xnative[2] - 1.)
                            * m::pow((2. * Xnative[2] - 1.) / poly_xt, poly_alpha - 1.))
                            / ((1. + poly_alpha) * poly_xt)
                        - (1. - hslope) * M_PI * m::cos(2. * M_PI * Xnative[2]));
            dxdX[3][3] = 1.;
        }
        /**
         * Transformation matrix for contravariant vectors to native, or covariant vectors to embedding
         */
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
        {
            // Okay this one should probably stay numerical
            Real dxdX_tmp[GR_DIM][GR_DIM];
            dxdX(Xnative, dxdX_tmp);
            invert(&dxdX_tmp[0][0],&dXdx[0][0]);
        }
};

/**
 * Wide-pole Kerr-Schild coordinates
 * Make sense only for spherical base systems!
 */
class WidepoleTransform {
    public:
        static constexpr char name[] = "WidepoleTransform";
        static constexpr GReal startx[3] = {-1, 0., 0.};
        static constexpr GReal stopx[3] = {-1, 1., 2*M_PI};

        const GReal lin_frac, n2, n3;
        GReal smoothness;

        // Constructor
        KOKKOS_FUNCTION WidepoleTransform(GReal lin_frac_in, GReal smoothness_in, GReal n2_in, GReal n3_in): lin_frac(lin_frac_in), smoothness(smoothness_in), n2(n2_in), n3(n3_in) 
        {
            GReal n3_temp = n3;
            if (n3 < M_PI) n3_temp = n2;
            GReal temp;
            if (smoothness <= 0) {
                if (lin_frac == 1) temp = 1.;
                else temp = lin_frac / (1. - lin_frac) * (1. / M_PI - 1. / n3_temp) * n3_temp / n2;
                if (abs(temp) < 1) smoothness = 1. / (n2 * log((1. + temp) / (1. - temp)));
                else {
                    printf("WARNING: It is harder to have del phi ~ del th. Try using lin_frac < %g \n",  1./ ((1. / M_PI - 1./ n3_temp) * n3_temp / n2 + 1.));
                    smoothness = 0.8 / n2;
                }
                smoothness = 0.02; //m::max(0.01, smoothness); // fix it for now for test
            }
        }

        // Coordinate transformations
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            Xembed[0] = Xnative[0];
            Xembed[1] = exp(Xnative[1]);
            GReal th;
            //th = M_PI / 2. * (1. + 2. * lin_frac * (Xnative[2] - 0.5) + (1. - lin_frac) * exp((Xnative[2] - 1.) / smoothness) - (1. - lin_frac) * exp(-Xnative[2] / smoothness));
            th = M_PI / 2. * (1. + 2. * lin_frac * (Xnative[2] - 0.5) + (1. - lin_frac) * (tanh((Xnative[2] - 1.) / smoothness) + 1.) - (1. - lin_frac) * (tanh(-Xnative[2] / smoothness) + 1.));
            Xembed[2] = excise(excise(th, 0.0, SMALL), M_PI, SMALL);
            Xembed[3] = Xnative[3];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
        {
            Xnative[0] = Xembed[0];
            Xnative[1] = log(Xembed[1]);
            Xnative[3] = Xembed[3];
            // Treat the special case with a macro
            ROOT_FIND
        }
        /**
         * Transformation matrix for contravariant vectors to embedding, or covariant vectors to native
         */
        KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
        {
            gzero2(dxdX);
            dxdX[0][0] = 1.;
            dxdX[1][1] = exp(Xnative[1]);
            //dxdX[2][2] = M_PI / 2. * (2. * lin_frac + (1. - lin_frac) / smoothness * exp((Xnative[2] - 1.) / smoothness) + (1. - lin_frac) / smoothness * exp(-Xnative[2] / smoothness));
            dxdX[2][2] = M_PI / 2. * (2. * lin_frac + (1. - lin_frac) / (smoothness * m::pow(cosh((Xnative[2] - 1.) / smoothness), 2.)) + (1. - lin_frac) / (smoothness * m::pow(cosh(Xnative[2] / smoothness),2.)));
            dxdX[3][3] = 1.;
        }
        /**
         * Transformation matrix for contravariant vectors to native, or covariant vectors to embedding
         */
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
        {
            // Okay this one should probably stay numerical
            Real dxdX_tmp[GR_DIM][GR_DIM];
            dxdX(Xnative, dxdX_tmp);
            invert(&dxdX_tmp[0][0],&dXdx[0][0]);
        }
};

// Bundle coordinates and transforms into umbrella variant types
// These act as a wannabe "interface" or "parent class" with the exception that access requires "mpark::visit"
// See coordinate_embedding.hpp

using SomeBaseCoords = mpark::variant<SphMinkowskiCoords, CartMinkowskiCoords, SphBLCoords, SphKSCoords, SphBLExtG, SphKSExtG, DCSKSCoords, DCSBLCoords, EDGBKSCoords, EDGBBLCoords>; // Changes Made. 
using SomeTransform = mpark::variant<NullTransform, ExponentialTransform, SuperExponentialTransform, ModifyTransform, FunkyTransform, SphNullTransform,WidepoleTransform>;

// added SphNullTransform
// using SomeBaseCoords = mpark::variant<SphMinkowskiCoords, CartMinkowskiCoords, SphBLCoords, SphKSCoords, SphBLExtG, SphKSExtG>;
// using SomeTransform = mpark::variant<NullTransform, SphNullTransform, ExponentialTransform, SuperExponentialTransform, ModifyTransform, FunkyTransform, WidepoleTransform>;
