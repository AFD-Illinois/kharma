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

// Question : Could I potentially replace all the places that SphKSExtG is used with DCSKSCoords? 
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

        static constexpr GReal A = 1.46797639e-8;
        static constexpr GReal B = 1.29411117;

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

        static constexpr GReal A = 1.46797639e-8;
        static constexpr GReal B = 1.29411117;

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
// Write a class DCSKSCoords with a new gcov definition. 
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

        KOKKOS_FUNCTION DCSKSCoords(GReal spin, GReal z): a(spin), zeta(z) {} //semicolon here ?

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            
            // Assign gcov matrix to zero. 

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal s2t = sth*sth;
            const GReal c2t = cth*cth ; 
            const GReal c4t = c2t*c2t ;
            // const GReal rho2 = r*r + a*a*cth*cth;
            const GReal ep2 = a*a ;
            const GReal ep3 = a * ep2 ;
            const GReal ep4 = ep2*ep2 ;
            const GReal ep5 = a * ep4;

            gzero2(gcov);
            gcov[0][0] = -1.+ (2.*r)/(pow(r,2.)+ ep2*c2t) + zeta*((ep2*(-338688.*(-1.+ 3.*c2t) + 4221.*pow(r,6.)*(-1.+ 3.*c2t) + 4221.*pow(r,7.)*(-1.+ 3.*c2t) - 420.*pow(r,2.)*(263.+ 321.*c2t) + 
            20.*pow(r,3.)*(-5428.+ 2025.*c2t) - 20.*pow(r,4.)*(1523.+ 2781.*c2t) + 168.*r*(-275.+ 3471.*c2t) + 2.*pow(r,5.)*(-2482.+ 6711.*c2t)))/(37632.*pow(r,10.)) + 
       (ep4*(-436560.*pow(r,10.)*(-1.+ 3.*c2t) - 
          436560.*pow(r,11.)*(-1.+ 3.*c2t) + 
          pow(r,5.)*(-27367138.+ 338395440.*c2t- 
            79842960.*c4t) + pow(r,7.)*(5994779.+ 
            9180120.*c2t- 23096700.*c4t) + 
          pow(r,8.)*(1835988.+ 14588424.*c2t- 20269020.*
             c4t) + pow(r,9.)*(609036.+ 6695880.*c2t- 
            14028580.*c4t) - 3048192.*(299.- 1440.*c2t+ 
            720.*c4t) + 1512.*r*(200377.- 262200.*c2t+ 
            546900.*c4t) + 540.*pow(r,2.)*(209695.- 
            1641704.*c2t+ 3850652.*c4t) + 
          36.*pow(r,4.)*(-1309617.+ 14023212.*c2t+ 
            5328620.*c4t) + pow(r,6.)*(1493875.+ 
            66961128.*c2t+ 66934880.*c4t) - 
          12.*pow(r,3.)*(-7413845.- 9229392.*c2t+ 156818740.*
             c4t)))/(13547520.*pow(r,14.))) ; 
            
            gcov[0][1] = (2.*r)/(pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/301056.*(ep2*(38430.*pow(r,7.)+ 19215.*pow(r,8.)+ 44667.*pow(r,9.)+ 
           2709504.*(-1.+ 3.*c2t) + 4032.*pow(r,2.)*
            (97.+ 193.*c2t) - 1344.*r*(733.+ 447.*c2t) + 
           84.*pow(r,6.)*(1529.+ 994.*c2t) + 168.*pow(r,5.)*
            (1127.+ 2200.*c2t) + 32.*pow(r,3.)*(25313.+ 
             9972.*c2t) + 16.*pow(r,4.)*(30463.+ 47862.*c2t)))/
         pow(r,10.)- (ep4*(16616754.*pow(r,11.)+ 40468617.*pow(r,12.)+ 
          34100640.*pow(r,13.)+ pow(r,10.)*(7900548.- 58216320.*c2t) + 
          97542144.*(299.- 1440.*c2t+ 720.*c4t) + 
          241920.*r*(20203.- 237864.*c2t+ 35772.*c4t) - 
          34560.*pow(r,2.)*(34137.+ 11672.*c2t+ 
            1800124.*c4t) + 72.*pow(r,9.)*(1594449.- 
            7358896.*c2t+ 2822080.*c4t) + 
          768.*pow(r,4.)*(-1543157.- 18681696.*c2t+ 
            8492020.*c4t) - 16.*pow(r,8.)*(-5324505.+ 
            66733824.*c2t+ 11040520.*c4t) - 
          256.*pow(r,6.)*(344923.+ 19052946.*c2t+ 
            15954625.*c4t) + 64.*pow(r,5.)*(-1116853.- 
            193194456.*c2t+ 21476040.*c4t) + 
          768.*pow(r,3.)*(-4475005.- 7417476.*c2t+ 
            45527060.*c4t) - 32.*pow(r,7.)*(332559.+ 
            53414256.*c2t+ 51544360.*c4t)))/
        (433520640.*pow(r,14.))); 
            
            gcov[0][3] = (2.*r*a*(-1.+ c2t))/
      (pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/112.*((189.+ 120.*r + 70.*pow(r,2.))*a*
          (-1.+ c2t))/pow(r,6.)+ (ep3*(-1.+ c2t)*
         (-10160640.*(-1.+ 3.*c2t) + 105828.*pow(r,7.)*
           (-1.+ 5.*c2t) + 30.*pow(r,5.)*(2247.+ 95.*c2t) + 
          55440.*r*(-25.+ 453.*c2t) + 
          15.*pow(r,6.)*(-271.+ 24875.*c2t) + 
          360.*pow(r,2.)*(-9205.+ 64257.*c2t) + 
          16.*pow(r,4.)*(-40883.+ 112735.*c2t) + 
          8.*pow(r,3.)*(-382792.+ 1712045.*c2t)))/(1128960.*pow(r,10.)) - 
       (ep5*(-1.+ c2t)*
         (253500126.*pow(r,11.)*(-1.+ 5.*c2t) + 
          825.*pow(r,10.)*(-579283.+ 2007359.*c2t) + 
          1408264704.*(299.- 1440.*c2t+ 720.*c4t) - 
          698544.*r*(200377.- 746040.*c2t+ 
            1030740.*c4t) - 249480.*pow(r,2.)*(209695.- 
            2247176.*c2t+ 4032764.*c4t) + 
          84.*pow(r,8.)*(-17277973.- 20585428.*c2t+ 
            123744773.*c4t) + 5544.*pow(r,3.)*(-7413845.- 
            2922912.*c2t+ 192665380.*c4t) - 
          14.*pow(r,7.)*(237841947.- 817153200.*c2t+ 
            192857740.*c4t) + 42.*pow(r,6.)*(-23970449.- 
            80221656.*c2t+ 201416960.*c4t) + 
          18.*pow(r,9.)*(-42498828.- 73631707.*c2t+ 
            455209643.*c4t) + 616.*pow(r,4.)*(35359659.- 
            382674420.*c2t+ 1128945980.*c4t) + 
          84.*pow(r,5.)*(150519259.- 1600861256.*c2t+ 
            3476016920.*c4t)))/(6258954240.*pow(r,14.)));


            gcov[1][0] = gcov[0][1] ;

            gcov[1][1] = 1.+ (2.*r)/(pow(r,2.)+ ep2*c2t) + 
     zeta*((ep2*(44667.*pow(r,8.)- 677376.*(-1.+ 3.*c2t) - 
          336.*r*(-1741.+ 2577.*c2t) + 
          21.*pow(r,7.)*(3349.+ 2624.*c2t) + 
          42.*pow(r,6.)*(2243.+ 5501.*c2t) + 
          168.*pow(r,4.)*(-1712.+ 15471.*c2t) + 
          36.*pow(r,5.)*(4639.+ 18983.*c2t) + 
          168.*pow(r,2.)*(-13961.+ 41625.*c2t) + 
          8.*pow(r,3.)*(-115844.+ 596277.*c2t)))/(75264.*pow(r,10.)) + 
       (ep4*(34100640.*pow(r,12.)+ 12.*pow(r,10.)*(9425519.+ 
            2153220.*c2t) + 3.*pow(r,11.)*(28800979.+ 
            5094000.*c2t) + pow(r,9.)*(88722252.+ 
            45073920.*c2t- 112228640.*c4t) - 
          24385536.*(299.- 1440.*c2t+ 720.*c4t) - 
          12096.*r*(402407.- 2640840.*c2t+ 
            904620.*c4t) - 8.*pow(r,8.)*(-18530343.+ 
            95285784.*c2t+ 14911000.*c4t) + 
          96.*pow(r,3.)*(74125625.- 728831088.*c2t+ 
            87392300.*c4t) - 192.*pow(r,4.)*(1862003.+ 
            77480766.*c2t+ 115164800.*c4t) - 
          8.*pow(r,7.)*(16908235.+ 280371816.*c2t+ 
            125586100.*c4t) + 864.*pow(r,2.)*(46048633.- 
            215092120.*c2t+ 128516260.*c4t) - 
          8.*pow(r,6.)*(203181015.+ 195320136.*c2t+ 
            760558480.*c4t) - 16.*pow(r,5.)*(320810645.- 
            177099864.*c2t+ 1271442120.*c4t)))/
        (108380160.*pow(r,14.)));  
    
            gcov[1][3] = (a*(-1.+ c2t)*
       (r*(2.+ r) + ep2*c2t))/
      (pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/56.*((-189.- 120.*r - 70.*pow(r,2.)+ (709.*pow(r,6.))/64.)*a*
          (-1.+ c2t))/((-2.+ r)*pow(r,6.)) + 
       (ep3*(-1.+ c2t)*(-468195.*pow(r,8.)- 536655.*pow(r,9.)- 
          40642560.*(-1.+ 3.*c2t) + 20160.*r*
           (733.+ 1959.*c2t) + 8640.*pow(r,2.)*
           (1673.+ 5939.*c2t) + 210.*pow(r,7.)*(-6871.+ 
            7236.*c2t) + 12.*pow(r,6.)*(-269333.+ 
            347590.*c2t) + 24.*pow(r,5.)*(-213628.+ 
            506835.*c2t) + 48.*pow(r,4.)*(-101873.+ 
            584060.*c2t) + 32.*pow(r,3.)*(-124492.+ 
            2059265.*c2t)))/(4515840.*pow(r,10.)) - 
       (ep5*(-1.+ c2t)*(14237034525.*pow(r,12.)+ 
          9822762999.*pow(r,13.)+ 630.*pow(r,11.)*(24351847.+ 15366912.*c2t) + 
          12.*pow(r,10.)*(2055167671.+ 661796080.*c2t) + 
          22532235264.*(299.- 1440.*c2t+ 720.*c4t) - 
          55883520.*r*(-20203.+ 141096.*c2t+ 
            60996.*c4t) - 1596672.*pow(r,2.)*(-1939059.+ 
            7011880.*c2t+ 6069020.*c4t) + 
          88704.*pow(r,3.)*(-2586065.- 69832512.*c2t+ 
            134071420.*c4t) + 896.*pow(r,7.)*(-27911013.- 
            452817432.*c2t+ 982380230.*c4t) + 
          2688.*pow(r,6.)*(-90875083.- 349488852.*c2t+ 
            1114846730.*c4t) + 9856.*pow(r,4.)*(-41602311.- 
            453834864.*c2t+ 1235148500.*c4t) + 
          112.*pow(r,8.)*(417172557.- 2041108464.*c2t+ 
            2487770036.*c4t) + 24.*pow(r,9.)*(1992373203.- 
            2983543860.*c2t+ 4320802640.*c4t) + 
          448.*pow(r,5.)*(-1425858819.- 5438639760.*c2t+ 
            19353172240.*c4t)))/(100143267840.*pow(r,14.))); 

   
            gcov[2][2] = pow(r,2.)+ ep2*c2t- 
     (zeta*(1080.*pow(r,4.)*(338688.+ 80808.*r + 67380.*pow(r,2.)+ 10360.*pow(r,3.)+ 
          18908.*pow(r,4.)+ 9940.*pow(r,5.)+ 4221.*pow(r,6.))*ep2*
         (1.- 3.*c2t) + ep4*
         (3141900.*pow(r,9.)*(-1.+ 3.*c2t) + 1309680.*pow(r,10.)*
           (-1.+ 3.*c2t) - 9144576.*(299.- 1440.*c2t+ 
            720.*c4t) - 4536.*r*(41543.- 947400.*c2t+ 
            420780.*c4t) + 1620.*pow(r,2.)*(-93041.- 
            85688.*c2t+ 1454300.*c4t) + 
          12.*pow(r,8.)*(-535899.- 116700.*c2t+ 
            2367475.*c4t) + 18.*pow(r,5.)*(-7227915.+ 
            2913384.*c2t+ 17254600.*c4t) + 
          36.*pow(r,3.)*(4260605.- 30961368.*c2t+ 
            27684860.*c4t) + 3.*pow(r,6.)*(-20890109.+ 
            18841800.*c2t+ 46796480.*c4t) + 
          pow(r,7.)*(-20549463.+ 6081360.*c2t+ 59683820.*
             c4t) + 4.*pow(r,4.)*(-33335811.- 120639012.*
             c2t+ 227696380.*c4t))))/(40642560.*pow(r,11.)) ; 
            gcov[2][3] = 0.;

            gcov[3][0] = gcov[0][3] ; 
            gcov[3][1] = gcov[1][3] ;
  
            gcov[3][3] = (1.- c2t)*(pow(r,2.)+ ep2- 
       (2.*r*ep2*(-1.+ c2t))/
        (pow(r,2.)+ ep2*c2t)) + 
     zeta*(-1/37632.*((338688.+ 80808.*r + 67380.*pow(r,2.)+ 10360.*pow(r,3.)+ 
           18908.*pow(r,4.)+ 9940.*pow(r,5.)+ 4221.*pow(r,6.))*ep2*
          (-1.+ c2t)*(-1.+ 3.*c2t))/pow(r,7.)+ 
       (ep4*(-1.+ c2t)*
         (3141900.*pow(r,9.)*(-1.+ 3.*c2t) + 1309680.*pow(r,10.)*
           (-1.+ 3.*c2t) - 9144576.*(299.- 1440.*c2t+ 
            720.*c4t) + 4536.*r*(119737.+ 302280.*c2t+ 
            63060.*c4t) + 1620.*pow(r,2.)*(240495.- 
            1419832.*c2t+ 2454908.*c4t) + 
          12.*pow(r,8.)*(-156009.- 1636260.*c2t+ 
            3507145.*c4t) + 18.*pow(r,5.)*(-4337355.- 
            8648856.*c2t+ 25926280.*c4t) + 
          36.*pow(r,3.)*(10727645.- 56829528.*c2t+ 
            47085980.*c4t) + 3.*pow(r,6.)*(-6926429.- 
            37012920.*c2t+ 88687520.*c4t) + 
          pow(r,7.)*(-696903.- 73328880.*c2t+ 119241500.*
             c4t) + 4.*pow(r,4.)*(-9548811.- 215787012.*c2t+ 
            299057380.*c4t)))/(40642560.*pow(r,11.)));

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
            const GReal s2t = sth*sth;
            const GReal c2t = cth*cth ; 
            const GReal c4t = c2t*c2t ;
            const GReal ep2 = a*a ;
            const GReal ep4 = ep2*ep2 ;
            const GReal ep5 = a * ep4;

            trans[0][0] = 1;
            trans[0][1] = -1.+ (pow(r,2.)+ ep2)/((-1.+ r - sqrt(1.- ep2) - 
        ((-915.*ep2)/28672.- (351479.*ep4)/13762560.)*zeta)*
       (-1.+ r + sqrt(1.- ep2) + ((477.*ep2)/4096.+ 
          (731081.*ep4)/13762560.)*zeta)) ;
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
            trans[3][1] = a/((-1.+ r - sqrt(1.- ep2) - 
       ((-915.*ep2)/28672.- (351479.*ep4)/13762560.)*zeta)*
      (-1.+ r + sqrt(1.- ep2) + 
       (709/3584.+ (7477.*ep2)/86016.+ (10982957.*ep4)/
          151388160.)*zeta));
            trans[3][2] = 0;
            trans[3][3] = 1;

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
        }

        KOKKOS_INLINE_FUNCTION void vec_to_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
        {
            GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            GReal rtrans[GR_DIM][GR_DIM], trans[GR_DIM][GR_DIM];
            DLOOP2 rtrans[mu][nu] = (mu == nu);

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal s2t = sth*sth;
            const GReal c2t = cth*cth ; 
            const GReal c4t = c2t*c2t ;
            const GReal ep2 = a*a ;
            const GReal ep4 = ep2*ep2 ;

            trans[0][0] = 1;
            trans[0][1] = -1.+ (pow(r,2.)+ ep2)/((-1.+ r - sqrt(1.- ep2) - 
        ((-915.*ep2)/28672.- (351479.*ep4)/13762560.)*zeta)*
       (-1.+ r + sqrt(1.- ep2) + ((477.*ep2)/4096.+ 
          (731081.*ep4)/13762560.)*zeta)) ;
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
            trans[3][1] = a/((-1.+ r - sqrt(1.- ep2) - 
       ((-915.*ep2)/28672.- (351479.*ep4)/13762560.)*zeta)*
      (-1.+ r + sqrt(1.- ep2) + 
       (709/3584.+ (7477.*ep2)/86016.+ (10982957.*ep4)/
          151388160.)*zeta));
            trans[3][2] = 0;
            trans[3][3] = 1;

            invert(&rtrans[0][0], &trans[0][0]); // INVERTING BECAUSE IT IS NOW FROM KS TO BL!!! 

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
            const GReal ep2 = a*a;
            const GReal ep3 = a * ep2 ;
            const GReal ep4 = ep2*ep2;
            const GReal ep5 = a*ep4 ;
            // const GReal DD = 1. - 2./r + ep2/(r*r);
            // const GReal mu = 1. + ep2*cth*cth/(r*r);


            gcov[0][0] = -1.+ (2.*r)/(pow(r,2.)+ ep2*c2t) + 
     zeta*((ep2*(-338688.*(-1.+ 3.*c2t) + 
          4221.*pow(r,6.)*(-1.+ 3.*c2t) + 4221.*pow(r,7.)*
           (-1.+ 3.*c2t) - 420.*pow(r,2.)*(263.+ 321.*c2t) + 
          20.*pow(r,3.)*(-5428.+ 2025.*c2t) - 
          20.*pow(r,4.)*(1523.+ 2781.*c2t) + 
          168.*r*(-275.+ 3471.*c2t) + 
          2.*pow(r,5.)*(-2482.+ 6711.*c2t)))/(37632.*pow(r,10.)) + 
       (ep4*(-436560.*pow(r,10.)*(-1.+ 3.*c2t) - 
          436560.*pow(r,11.)*(-1.+ 3.*c2t) + 
          pow(r,5.)*(-27367138.+ 338395440.*c2t- 
            79842960.*c4t) + pow(r,7.)*(5994779.+ 
            9180120.*c2t- 23096700.*c4t) + 
          pow(r,8.)*(1835988.+ 14588424.*c2t- 20269020.*
             c4t) + pow(r,9.)*(609036.+ 6695880.*c2t- 
            14028580.*c4t) - 3048192.*(299.- 1440.*c2t+ 
            720.*c4t) + 1512.*r*(200377.- 262200.*c2t+ 
            546900.*c4t) + 540.*pow(r,2.)*(209695.- 
            1641704.*c2t+ 3850652.*c4t) + 
          36.*pow(r,4.)*(-1309617.+ 14023212.*c2t+ 
            5328620.*c4t) + pow(r,6.)*(1493875.+ 
            66961128.*c2t+ 66934880.*c4t) - 
          12.*pow(r,3.)*(-7413845.- 9229392.*c2t+ 156818740.*
             c4t)))/(13547520.*pow(r,14.))) ;

            gcov[0][3] = (-2.*r*a*(1.- c2t))/
      (pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/112.*((189.+ 120.*r + 70.*pow(r,2.))*a*
          (-1.+ c2t))/pow(r,6.)+ (ep3*(-1.+ c2t)*
         (-10160640.*(-1.+ 3.*c2t) + 105828.*pow(r,7.)*
           (-1.+ 5.*c2t) + 30.*pow(r,5.)*(2247.+ 95.*c2t) + 
          55440.*r*(-25.+ 453.*c2t) + 
          15.*pow(r,6.)*(-271.+ 24875.*c2t) + 
          360.*pow(r,2.)*(-9205.+ 64257.*c2t) + 
          16.*pow(r,4.)*(-40883.+ 112735.*c2t) + 
          8.*pow(r,3.)*(-382792.+ 1712045.*c2t)))/(1128960.*pow(r,10.)) - 
       (ep5*(-1.+ c2t)*
         (253500126.*pow(r,11.)*(-1.+ 5.*c2t) + 
          825.*pow(r,10.)*(-579283.+ 2007359.*c2t) + 
          1408264704.*(299.- 1440.*c2t+ 720.*c4t) - 
          698544.*r*(200377.- 746040.*c2t+ 
            1030740.*c4t) - 249480.*pow(r,2.)*(209695.- 
            2247176.*c2t+ 4032764.*c4t) + 
          84.*pow(r,8.)*(-17277973.- 20585428.*c2t+ 
            123744773.*c4t) + 5544.*pow(r,3.)*(-7413845.- 
            2922912.*c2t+ 192665380.*c4t) - 
          14.*pow(r,7.)*(237841947.- 817153200.*c2t+ 
            192857740.*c4t) + 42.*pow(r,6.)*(-23970449.- 
            80221656.*c2t+ 201416960.*c4t) + 
          18.*pow(r,9.)*(-42498828.- 73631707.*c2t+ 
            455209643.*c4t) + 616.*pow(r,4.)*(35359659.- 
            382674420.*c2t+ 1128945980.*c4t) + 
          84.*pow(r,5.)*(150519259.- 1600861256.*c2t+ 
            3476016920.*c4t)))/(6258954240.*pow(r,14.)));

            gcov[1][1] =  (pow(r,2.)+ ep2*c2t)/(-2.*r + pow(r,2.)+ ep2) + 
     zeta*((ep2*(1693440.*(-1.+ 3.*c2t) + 
          1407.*pow(r,7.)*(-1.+ 3.*c2t) - 
          2.*pow(r,5.)*(1154.+ 213.*c2t) + 
          7.*pow(r,6.)*(-201.+ 253.*c2t) - 
          28.*pow(r,2.)*(-787.+ 5499.*c2t) + 
          20.*pow(r,4.)*(-4542.+ 5737.*c2t) - 
          20.*pow(r,3.)*(-6194.+ 23433.*c2t) - 
          56.*r*(-19703.+ 28491.*c2t)))/(12544.*pow((-2.+ r),2.)*pow(r,8.)) + 
       (ep4*(-436560.*pow(r,12.)*(-1.+ 3.*c2t) + 
          240.*pow(r,11.)*(-1819.+ 7662.*c2t) + 
          pow(r,3.)*(3892185204.+ 4105960416.*c2t- 
            6359274480.*c4t) + pow(r,8.)*(-66718519.+ 
            290871480.*c2t- 159391380.*c4t) + 
          pow(r,7.)*(11023875.+ 287437416.*c2t- 
            55986760.*c4t) + pow(r,10.)*(1396596.+ 
            3237000.*c2t- 9469900.*c4t) - 
          140216832.*(299.- 1440.*c2t+ 720.*c4t) + 
          63504.*r*(874921.- 3510840.*c2t+ 
            2171220.*c4t) + 5.*pow(r,9.)*(3065991.- 
            15226368.*c2t+ 11239936.*c4t) + 
          40.*pow(r,6.)*(6235035.- 24204108.*c2t+ 
            13176050.*c4t) - 864.*pow(r,2.)*(22756577.- 
            62589680.*c2t+ 42840440.*c4t) - 
          60.*pow(r,4.)*(78767423.- 112427928.*c2t+ 
            92452100.*c4t) + 4.*pow(r,5.)*(367223585.- 
            977540892.*c2t+ 844072920.*c4t)))/
        (13547520.*pow((-2.+ r),3.)*pow(r,12.)));

            gcov[2][2] = pow(r,2.)+ ep2*c2t- 
     (zeta*(1080.*pow(r,4.)*(338688.+ 80808.*r + 67380.*pow(r,2.)+ 10360.*pow(r,3.)+ 
          18908.*pow(r,4.)+ 9940.*pow(r,5.)+ 4221.*pow(r,6.))*ep2*
         (1.- 3.*c2t) + ep4*
         (3141900.*pow(r,9.)*(-1.+ 3.*c2t) + 1309680.*pow(r,10.)*
           (-1.+ 3.*c2t) - 9144576.*(299.- 1440.*c2t+ 
            720.*c4t) - 4536.*r*(41543.- 947400.*c2t+ 
            420780.*c4t) + 1620.*pow(r,2.)*(-93041.- 
            85688.*c2t+ 1454300.*c4t) + 
          12.*pow(r,8.)*(-535899.- 116700.*c2t+ 
            2367475.*c4t) + 18.*pow(r,5.)*(-7227915.+ 
            2913384.*c2t+ 17254600.*c4t) + 
          36.*pow(r,3.)*(4260605.- 30961368.*c2t+ 
            27684860.*c4t) + 3.*pow(r,6.)*(-20890109.+ 
            18841800.*c2t+ 46796480.*c4t) + 
          pow(r,7.)*(-20549463.+ 6081360.*c2t+ 59683820.*
             c4t) + 4.*pow(r,4.)*(-33335811.- 120639012.*
             c2t+ 227696380.*c4t))))/(40642560.*pow(r,11.));

            gcov[3][0] = gcov[0][3]; 

            gcov[3][3] = (1.- c2t)*(pow(r,2.)+ ep2+ 
       (2.*r*ep2*(1.- c2t))/
        (pow(r,2.)+ ep2*c2t)) + 
     zeta*(-1/37632.*((338688.+ 80808.*r + 67380.*pow(r,2.)+ 10360.*pow(r,3.)+ 
           18908.*pow(r,4.)+ 9940.*pow(r,5.)+ 4221.*pow(r,6.))*ep2*
          (-1.+ c2t)*(-1.+ 3.*c2t))/pow(r,7.)+ 
       (ep4*(-1.+ c2t)*
         (3141900.*pow(r,9.)*(-1.+ 3.*c2t) + 1309680.*pow(r,10.)*
           (-1.+ 3.*c2t) - 9144576.*(299.- 1440.*c2t+ 
            720.*c4t) + 4536.*r*(119737.+ 302280.*c2t+ 
            63060.*c4t) + 1620.*pow(r,2.)*(240495.- 
            1419832.*c2t+ 2454908.*c4t) + 
          12.*pow(r,8.)*(-156009.- 1636260.*c2t+ 
            3507145.*c4t) + 18.*pow(r,5.)*(-4337355.- 
            8648856.*c2t+ 25926280.*c4t) + 
          36.*pow(r,3.)*(10727645.- 56829528.*c2t+ 
            47085980.*c4t) + 3.*pow(r,6.)*(-6926429.- 
            37012920.*c2t+ 88687520.*c4t) + 
          pow(r,7.)*(-696903.- 73328880.*c2t+ 119241500.*
             c4t) + 4.*pow(r,4.)*(-9548811.- 215787012.*c2t+ 
            299057380.*c4t)))/(40642560.*pow(r,11.)));

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

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal s2 = sth*sth;
            const GReal c2t = cth*cth; 
            const GReal c4t = c2t*c2t;
            const GReal rho2 = r*r + a*a*cth*cth;
            const GReal a2 = a*a;
            const GReal ep2 = a2;
            const GReal ep3 = a * ep2 ;
            const GReal ep4 = ep2*ep2;
            const GReal ep5 = a * ep4;
            const GReal a4 = a2*a2;
            // const GReal DD = 1. - 2./r + a2/(r*r);
            // const GReal mu = 1. + a2*cth*cth/(r*r);

            gcov[0][0] = -1.+ (2.*r)/(pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/15.*(-400.+ 96.*r + 66.*pow(r,2.)+ 130.*pow(r,3.)+ 5.*pow(r,4.))/pow(r,7.)+ 
       (ep2*(pow(r,7.)*(444696.- 562338.*c2t) + 
          8820000.*(-1.+ 3.*c2t) - 19600.*r*
           (-467.+ 1251.*c2t) - 63.*pow(r,8.)*(-3267.+ 
            8926.*c2t) + 1050.*pow(r,3.)*(-1465.+ 
            11997.*c2t) - 2100.*pow(r,2.)*(-955.+ 
            22577.*c2t) - 6.*pow(r,6.)*(-59329.+ 82437.*c2t) + 
          15.*pow(r,5.)*(-52533.+ 455029.*c2t) + 
          10.*pow(r,4.)*(-281221.+ 1218513.*c2t)))/(110250.*pow(r,11.)) + 
       (ep4*(675.*pow(r,12.)*(-19717.+ 67726.*c2t) + 
          450.*pow(r,11.)*(43312.+ 101589.*c2t) - 
          164640000.*(-70.+ 585.*c2t+ 156.*c4t) - 
          8232000.*r*(3949.- 11058.*c2t+ 6948.*c4t) - 
          39200.*pow(r,2.)*(189191.- 824825.*c2t+ 
            972045.*c4t) + 60.*pow(r,9.)*(717867.- 
            13885852.*c2t+ 12733507.*c4t) + 
          30.*pow(r,10.)*(209773.- 9090216.*c2t+ 
            16888370.*c4t) + 1400.*pow(r,3.)*(648009.- 
            14691730.*c2t+ 26074500.*c4t) + 
          420.*pow(r,4.)*(553219.- 32471380.*c2t+ 222891320.*
             c4t) - 14.*pow(r,7.)*(-11393603.+ 38599350.*
             c2t+ 359928985.*c4t) + 
          2.*pow(r,8.)*(59964679.- 491173980.*c2t+ 
            452824800.*c4t) - 28.*pow(r,5.)*(21852807.+ 
            12094180.*c2t+ 762315255.*c4t) - 
          14.*pow(r,6.)*(42004007.- 226439060.*c2t+ 
            1041312310.*c4t)))/(30870000.*pow(r,15.)));

            gcov[0][1] = (2.*r)/(pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/60.*(-1600.- 416.*r + 56.*pow(r,2.)+ 548.*pow(r,3.)+ 294.*pow(r,4.)+ 147.*pow(r,5.)+ 
          73.*pow(r,6.))/pow(r,7.)+ (ep2*(2285850.*pow(r,8.)+ 69825.*pow(r,9.)+ 
          617400.*pow(r,10.)+ 141120000.*(-1.+ 3.*c2t) - 
          627200.*r*(-121.+ 288.*c2t) - 11200.*pow(r,3.)*
           (-929.+ 17802.*c2t) - 11200.*pow(r,2.)*
           (-6253.+ 75795.*c2t) + 504.*pow(r,6.)*
           (-8929.+ 114112.*c2t) + 1200.*pow(r,5.)*
           (-17287.+ 122764.*c2t) + 480.*pow(r,4.)*
           (-52522.+ 180841.*c2t) + 
          84.*pow(r,7.)*(12767.+ 228924.*c2t)))/(1764000.*pow(r,11.)) + 
       (ep4*(1392308050.*pow(r,12.)+ 1041898025.*pow(r,13.)- 1799623000.*pow(r,14.)+ 
          500.*pow(r,11.)*(-5124459.+ 2241328.*c2t) - 
          10536960000.*(-70.+ 585.*c2t+ 156.*c4t) - 
          1580544000.*r*(1083.- 1736.*c2t+ 2836.*c4t) + 
          448000.*pow(r,3.)*(-1355339.+ 725916.*c2t+ 
            521022.*c4t) - 2508800.*pow(r,2.)*(530336.- 
            1371665.*c2t+ 1865385.*c4t) + 
          7168.*pow(r,5.)*(-25394838.- 42977570.*c2t+ 
            160747005.*c4t) + 17920.*pow(r,4.)*(-18316909.- 
            25501520.*c2t+ 325649955.*c4t) - 
          80.*pow(r,9.)*(120391011.+ 543979856.*c2t+ 
            348977760.*c4t) + 40.*pow(r,10.)*(-186388779.- 
            511177616.*c2t+ 352826880.*c4t) - 
          896.*pow(r,6.)*(121697859.+ 19970220.*c2t+ 
            463821790.*c4t) - 160.*pow(r,8.)*(87716331.+ 
            307196240.*c2t+ 968666736.*c4t) - 
          448.*pow(r,7.)*(75699653.+ 131599120.*c2t+ 
            1095469160.*c4t)))/(1975680000.*pow(r,15.)));

            gcov[0][3] = (2.*r*a*(-1.+ c2t))/
      (pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/15.*((-400.+ 144.*r + 90.*pow(r,2.)+ 140.*pow(r,3.)+ 9.*pow(r,4.))*a*
          (-1.+ c2t))/pow(r,7.)- (ep3*(-1.+ c2t)*
         (pow(r,4.)*(2736210.- 4410530.*c2t) + 
          pow(r,5.)*(766015.- 3620183.*c2t) - 
          8820000.*(-1.+ 3.*c2t) + 19600.*r*
           (-467.+ 1551.*c2t) - 12.*pow(r,6.)*(26511.+ 
            6310.*c2t) + 750.*pow(r,3.)*(2051.+ 8733.*c2t) + 
          2100.*pow(r,2.)*(-955.+ 21233.*c2t) + 
          3.*pow(r,8.)*(-63529.+ 262520.*c2t) + 
          pow(r,7.)*(-406611.+ 563055.*c2t)))/(110250.*pow(r,11.)) + 
       (ep5*(-1.+ c2t)*
         (1100.*pow(r,11.)*(604577.+ 2765240.*c2t) + 
          275.*pow(r,12.)*(-1565293.+ 9910190.*c2t) + 
          pow(r,7.)*(5564719986.+ 15014000340.*c2t- 
            80476531230.*c4t) - 5433120000.*
           (-70.+ 585.*c2t+ 156.*c4t) - 
          271656000.*r*(3949.- 12858.*c2t+ 5988.*c4t) - 
          1293600.*pow(r,2.)*(189191.- 642545.*c2t+ 
            376485.*c4t) + 138600.*pow(r,3.)*(216003.- 
            6904190.*c2t+ 13490140.*c4t) + 
          13860.*pow(r,4.)*(553219.- 39924380.*c2t+ 
            231663920.*c4t) + 30.*pow(r,10.)*(9643021.- 
            248784875.*c2t+ 723623755.*c4t) + 
          2.*pow(r,8.)*(2147992407.- 8519941500.*c2t+ 
            1468127080.*c4t) + 20.*pow(r,9.)*(80598297.- 
            1173493593.*c2t+ 1518021880.*c4t) - 
          42.*pow(r,6.)*(462044077.- 3947139700.*c2t+ 
            1848679130.*c4t) + 308.*pow(r,5.)*(-65558421.+ 
            118444260.*c2t+ 2022864835.*c4t)))/
        (1018710000.*pow(r,15.)));

            gcov[1][0] = gcov[0][1] ;

            gcov[1][1] = 1.+ (2.*r)/(pow(r,2.)+ ep2*c2t) + 
     zeta*((400.+ 304.*r + 598.*pow(r,2.)+ 380.*pow(r,3.)+ 218.*pow(r,4.)+ 58.*pow(r,5.))/(15.*pow(r,7.)) - 
       (ep2*(396900.*pow(r,9.)- 35280000.*(-1.+ 3.*c2t) - 
          78400.*r*(17.+ 99.*c2t) - 12600.*pow(r,3.)*
           (-4369.+ 633.*c2t) + 980.*pow(r,5.)*(46009.+ 
            7203.*c2t) + 420.*pow(r,7.)*(6684.+ 12533.*c2t) + 
          21.*pow(r,8.)*(39721.+ 41312.*c2t) + 
          2800.*pow(r,2.)*(-60041.+ 235059.*c2t) + 
          40.*pow(r,4.)*(1347041.+ 2150517.*c2t) + 
          12.*pow(r,6.)*(798727.+ 2262494.*c2t)))/(441000.*pow(r,11.)) + 
       (ep4*(1892233000.*pow(r,13.)+ 25.*pow(r,12.)*(55171231.+ 
            85149776.*c2t) + 100.*pow(r,11.)*(-7500645.+ 
            98247164.*c2t) - 2634240000.*
           (-70.+ 585.*c2t+ 156.*c4t) - 
          131712000.*r*(2549.+ 642.*c2t+ 10068.*c4t) + 
          627200.*pow(r,2.)*(-2635481.+ 16660505.*c2t+ 
            1172475.*c4t) + 320.*pow(r,9.)*(37055339.+ 
            177810769.*c2t+ 8023029.*c4t) + 
          22400.*pow(r,3.)*(90871261.- 146475830.*c2t+ 
            254284440.*c4t) + 20.*pow(r,10.)*(11959097.+ 
            907145456.*c2t+ 405320880.*c4t) + 
          80.*pow(r,8.)*(631201573.+ 1861655120.*c2t+ 
            691306860.*c4t) + 6720.*pow(r,4.)*(222514369.- 
            524040980.*c2t+ 729021620.*c4t) + 
          224.*pow(r,7.)*(470763708.+ 1532278990.*c2t+ 
            1060051385.*c4t) + 1344.*pow(r,5.)*(513966826.- 
            1020185960.*c2t+ 1451406515.*c4t) + 
          224.*pow(r,6.)*(1074233009.+ 1240174100.*c2t+ 
            3288968330.*c4t)))/(493920000.*pow(r,15.)));  

            gcov[1][3] = (a*(-1.+ c2t)*(r*(2.+ r) + 
        ep2*c2t))/(pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/60.*((-1600.- 224.*r + 248.*pow(r,2.)+ 684.*pow(r,3.)+ 378.*pow(r,4.)+ 189.*pow(r,5.)+ 
           94.*pow(r,6.))*a*(-1.+ c2t))/pow(r,7.)- 
       (ep3*(-1.+ c2t)*(6464045.*pow(r,9.)+ 2495185.*pow(r,10.)- 
          141120000.*(-1.+ 3.*c2t) + 627200.*r*
           (-121.+ 438.*c2t) + 11200.*pow(r,2.)*
           (47.+ 57063.*c2t) + 4800.*pow(r,3.)*(-10073.+ 
            119472.*c2t) + 14.*pow(r,8.)*(555911.+ 
            642672.*c2t) + 160.*pow(r,4.)*(-68889.+ 
            1863532.*c2t) + 4.*pow(r,7.)*(2593463.+ 
            3116004.*c2t) + 8.*pow(r,6.)*(1229291.+ 
            5668182.*c2t) + 16.*pow(r,5.)*(-278405.+ 
            8480952.*c2t)))/(1764000.*pow(r,11.)) - 
       (ep5*(-1.+ c2t)*(54991814325.*pow(r,13.)+ 
          73888304175.*pow(r,14.)- 150.*pow(r,12.)*(-1137939359.+ 643667904.*
             c2t) - 100.*pow(r,11.)*(-3710302661.+ 3045427264.*
             c2t) + 347719680000.*(-70.+ 585.*c2t+ 
            156.*c4t) + 52157952000.*r*(1083.- 
            2336.*c2t+ 2516.*c4t) - 
          14784000.*pow(r,3.)*(-3265751.+ 2073668.*c2t+ 
            65070.*c4t) + 82790400.*pow(r,2.)*(383336.- 
            149885.*c2t+ 1496625.*c4t) - 
          591360.*pow(r,4.)*(-55440429.+ 31053430.*c2t+ 
            280971705.*c4t) - 19712.*pow(r,5.)*(-879140181.+ 
            309765960.*c2t+ 5879195410.*c4t) - 
          120.*pow(r,10.)*(-5121053291.+ 203299712.*c2t+ 
            8917059360.*c4t) - 48.*pow(r,9.)*(-19536651527.- 
            10408395520.*c2t+ 50410986800.*c4t) - 
          896.*pow(r,6.)*(-9277981317.- 2247068940.*c2t+ 
            73059718520.*c4t) - 192.*pow(r,7.)*(-18001434689.- 
            23312788400.*c2t+ 136575914000.*c4t) - 
          32.*pow(r,8.)*(-57002393073.- 35940222840.*c2t+ 
            249862541260.*c4t)))/(65197440000.*pow(r,15.)));

            gcov[2][2] = pow(r,2.)+ ep2*c2t+ 
     (zeta*(840.*pow(r,4.)*(8820000.- 6213200.*r - 3416700.*pow(r,2.)- 1855650.*pow(r,3.)+ 
          887110.*pow(r,4.)+ 800733.*pow(r,5.)+ 435540.*pow(r,6.)+ 187446.*pow(r,7.))*ep2*
         (1.- 3.*c2t) + ep4*
         (45715050.*pow(r,11.)*(-1.+ 3.*c2t) + 5625.*pow(r,10.)*
           (-20749.+ 58131.*c2t) + 493920000.*
           (-70.+ 585.*c2t+ 156.*c4t) + 
          24696000.*r*(3049.- 10698.*c2t+ 8868.*c4t) + 
          117600.*pow(r,2.)*(280331.- 1711445.*c2t+ 
            1596165.*c4t) + 180.*pow(r,9.)*(-1286466.- 
            846865.*c2t+ 5819941.*c4t) + 
          4200.*pow(r,3.)*(2362411.- 16650910.*c2t+ 
            14489100.*c4t) - 1260.*pow(r,4.)*(-3173281.- 
            5026080.*c2t+ 26477920.*c4t) + 
          42.*pow(r,8.)*(-18071967.- 940590.*c2t+ 
            54146980.*c4t) + 42.*pow(r,6.)*(-19116713.- 
            46592740.*c2t+ 138130070.*c4t) - 
          28.*pow(r,5.)*(11804979.- 261030540.*c2t+ 
            235282135.*c4t) + 6.*pow(r,7.)*(-259078241.- 
            99440670.*c2t+ 857000595.*c4t))))/
      (92610000.*pow(r,12.));

            gcov[3][0] = gcov[0][3];

            gcov[3][1] = gcov[1][3];

            gcov[3][3] = (1.- c2t)*(pow(r,2.)+ ep2- 
       (2.*r*ep2*(-1.+ c2t))/
        (pow(r,2.)+ ep2*c2t)) + 
     zeta*(((8820000.- 6213200.*r - 3416700.*pow(r,2.)- 1855650.*pow(r,3.)+ 
          887110.*pow(r,4.)+ 800733.*pow(r,5.)+ 435540.*pow(r,6.)+ 187446.*pow(r,7.))*ep2*
         (-1.+ c2t)*(-1.+ 3.*c2t))/(110250.*pow(r,8.)) - 
       (ep4*(-1.+ c2t)*
         (45715050.*pow(r,11.)*(-1.+ 3.*c2t) + 5625.*pow(r,10.)*
           (-20749.+ 58131.*c2t) + 493920000.*
           (-70.+ 585.*c2t+ 156.*c4t) + 
          24696000.*r*(3649.- 8958.*c2t+ 6528.*c4t) + 
          352800.*pow(r,2.)*(84857.- 350495.*c2t+ 
            320655.*c4t) + 12600.*pow(r,3.)*(-82303.- 
            1443030.*c2t+ 1592200.*c4t) + 
          180.*pow(r,9.)*(-411718.- 4345857.*c2t+ 
            8444185.*c4t) - 1260.*pow(r,4.)*(1578719.- 
            11450880.*c2t+ 28150720.*c4t) + 
          42.*pow(r,8.)*(-1863327.- 67980150.*c2t+ 
            104977900.*c4t) + 28.*pow(r,5.)*(-14247879.- 
            109560360.*c2t+ 137751665.*c4t) + 
          42.*pow(r,6.)*(30654807.- 316973820.*c2t+ 
            358739630.*c4t) + 6.*pow(r,7.)*(-25024421.- 
            1143700950.*c2t+ 1667207055.*c4t)))/
        (92610000.*pow(r,12.)));

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
            const GReal ep2 = a*a;
            const GReal ep3 = a * ep2 ;
            const GReal ep4 = ep2*ep2;

            trans[0][0] = 1;  
            trans[0][1] = -1.+ (pow(r,2.)+ ep2)/((-1.+ r - sqrt(1.- ep2) - 
        (-49/40.- (277.*ep2)/960.- (145711.*ep4)/3225600.)*
         zeta)*(-1.+ r + sqrt(1.- ep2) + 
        (-1/120.- (613.*ep2)/960.+ (2792449.*ep4)/3225600.)*
         zeta));
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
            trans[3][1] = a/((-1.+ r - sqrt(1.- ep2) - 
       (-49/40.- (277.*ep2)/960.- (145711.*ep4)/3225600.)*
        zeta)*(-1.+ r + sqrt(1.- ep2) + 
       (41/120.+ (113497.*ep2)/100800.+ (38608501.*ep4)/
          35481600.)*zeta)) ; 
            trans[3][2] = 0 ; 
            trans[3][3] = 1 ;

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu]; // CHANGES MADE, (DID NOT INCLUDE THIS BEFORE I THINK)

        }

        KOKKOS_INLINE_FUNCTION void vec_to_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
        {
            GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            GReal rtrans[GR_DIM][GR_DIM], trans[GR_DIM][GR_DIM];
            DLOOP2 rtrans[mu][nu] = (mu == nu);

            const GReal cth = m::cos(th);
            const GReal sth = m::sin(th);
            const GReal s2t = sth*sth;
            const GReal c2t = cth*cth ; 
            const GReal c4t = c2t*c2t ;
            const GReal ep2 = a*a ;
            const GReal ep3 = a * ep2 ;
            const GReal ep4 = ep2*ep2 ;

            trans[0][0] = 1;  
            trans[0][1] = -1.+ (pow(r,2.)+ ep2)/((-1.+ r - sqrt(1.- ep2) - 
        (-49/40.- (277.*ep2)/960.- (145711.*ep4)/3225600.)*
         zeta)*(-1.+ r + sqrt(1.- ep2) + 
        (-1/120.- (613.*ep2)/960.+ (2792449.*ep4)/3225600.)*
         zeta));
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
            trans[3][1] = a/((-1.+ r - sqrt(1.- ep2) - 
       (-49/40.- (277.*ep2)/960.- (145711.*ep4)/3225600.)*
        zeta)*(-1.+ r + sqrt(1.- ep2) + 
       (41/120.+ (113497.*ep2)/100800.+ (38608501.*ep4)/
          35481600.)*zeta)) ; 
            trans[3][2] = 0 ; 
            trans[3][3] = 1 ;

            invert(&rtrans[0][0], &trans[0][0]); // INVERTING BECAUSE IT IS NOW FROM KS TO BL!!! 

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
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
            const GReal ep2 = a*a;
            const GReal ep3 = a * ep2 ;
            const GReal ep4 = ep2*ep2;
            const GReal ep5 = a*ep4 ;
            const GReal r2 = r*r;
            const GReal r4 = r2*r2;
            // const GReal DD = 1. - 2./r + ep2/(r*r);
            // const GReal mu = 1. + ep2*cth*cth/(r*r);

            gcov[0][0] = -1.+ (2.*r)/(pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/15.*(-400.+ 96.*r + 66.*pow(r,2.)+ 130.*pow(r,3.)+ 5.*pow(r,4.))/pow(r,7.)+ 
       (ep2*(pow(r,7.)*(444696.- 562338.*c2t) + 
          8820000.*(-1.+ 3.*c2t) - 19600.*r*
           (-467.+ 1251.*c2t) - 63.*pow(r,8.)*(-3267.+ 
            8926.*c2t) + 1050.*pow(r,3.)*(-1465.+ 
            11997.*c2t) - 2100.*pow(r,2.)*(-955.+ 
            22577.*c2t) - 6.*pow(r,6.)*(-59329.+ 82437.*c2t) + 
          15.*pow(r,5.)*(-52533.+ 455029.*c2t) + 
          10.*pow(r,4.)*(-281221.+ 1218513.*c2t)))/(110250.*pow(r,11.)) + 
       (ep4*(675.*pow(r,12.)*(-19717.+ 67726.*c2t) + 
          450.*pow(r,11.)*(43312.+ 101589.*c2t) - 
          164640000.*(-70.+ 585.*c2t+ 156.*c4t) - 
          8232000.*r*(3949.- 11058.*c2t+ 6948.*c4t) - 
          39200.*pow(r,2.)*(189191.- 824825.*c2t+ 
            972045.*c4t) + 60.*pow(r,9.)*(717867.- 
            13885852.*c2t+ 12733507.*c4t) + 
          30.*pow(r,10.)*(209773.- 9090216.*c2t+ 
            16888370.*c4t) + 1400.*pow(r,3.)*(648009.- 
            14691730.*c2t+ 26074500.*c4t) + 
          420.*pow(r,4.)*(553219.- 32471380.*c2t+ 222891320.*
             c4t) - 14.*pow(r,7.)*(-11393603.+ 38599350.*
             c2t+ 359928985.*c4t) + 
          2.*pow(r,8.)*(59964679.- 491173980.*c2t+ 
            452824800.*c4t) - 28.*pow(r,5.)*(21852807.+ 
            12094180.*c2t+ 762315255.*c4t) - 
          14.*pow(r,6.)*(42004007.- 226439060.*c2t+ 
            1041312310.*c4t)))/(30870000.*pow(r,15.))) ;

            gcov[0][3] =  (-2.*r*a*(1.- c2t))/
      (pow(r,2.)+ ep2*c2t) + 
     zeta*(-1/15.*((-400.+ 144.*r + 90.*pow(r,2.)+ 140.*pow(r,3.)+ 9.*pow(r,4.))*a*
          (-1.+ c2t))/pow(r,7.)- (ep3*(-1.+ c2t)*
         (pow(r,4.)*(2736210.- 4410530.*c2t) + 
          pow(r,5.)*(766015.- 3620183.*c2t) - 
          8820000.*(-1.+ 3.*c2t) + 19600.*r*
           (-467.+ 1551.*c2t) - 12.*pow(r,6.)*(26511.+ 
            6310.*c2t) + 750.*pow(r,3.)*(2051.+ 8733.*c2t) + 
          2100.*pow(r,2.)*(-955.+ 21233.*c2t) + 
          3.*pow(r,8.)*(-63529.+ 262520.*c2t) + 
          pow(r,7.)*(-406611.+ 563055.*c2t)))/(110250.*pow(r,11.)) + 
       (ep5*(-1.+ c2t)*
         (1100.*pow(r,11.)*(604577.+ 2765240.*c2t) + 
          275.*pow(r,12.)*(-1565293.+ 9910190.*c2t) + 
          pow(r,7.)*(5564719986.+ 15014000340.*c2t- 
            80476531230.*c4t) - 5433120000.*
           (-70.+ 585.*c2t+ 156.*c4t) - 
          271656000.*r*(3949.- 12858.*c2t+ 5988.*c4t) - 
          1293600.*pow(r,2.)*(189191.- 642545.*c2t+ 
            376485.*c4t) + 138600.*pow(r,3.)*(216003.- 
            6904190.*c2t+ 13490140.*c4t) + 
          13860.*pow(r,4.)*(553219.- 39924380.*c2t+ 
            231663920.*c4t) + 30.*pow(r,10.)*(9643021.- 
            248784875.*c2t+ 723623755.*c4t) + 
          2.*pow(r,8.)*(2147992407.- 8519941500.*c2t+ 
            1468127080.*c4t) + 20.*pow(r,9.)*(80598297.- 
            1173493593.*c2t+ 1518021880.*c4t) - 
          42.*pow(r,6.)*(462044077.- 3947139700.*c2t+ 
            1848679130.*c4t) + 308.*pow(r,5.)*(-65558421.+ 
            118444260.*c2t+ 2022864835.*c4t)))/
        (1018710000.*pow(r,15.)));

            gcov[3][0] = gcov[0][3];

            gcov[1][1] = (pow(r,2.)+ ep2*c2t)/
      (-2.*r + pow(r,2.)+ ep2) + 
     zeta*(-1/15.*(-1840.+ 48.*r + 30.*pow(r,2.)+ 260.*pow(r,3.)+ 15.*pow(r,4.)+ 15.*pow(r,5.))/
         (pow((-2.+ r),2.)*pow(r,5.)) + (ep2*(55125.*pow(r,10.)+ 
          299880000.*(-1.+ 3.*c2t) - 1176000.*r*
           (-496.+ 1413.*c2t) + 42.*pow(r,8.)*
           (7787.+ 16014.*c2t) - 21.*pow(r,9.)*(-6301.+ 
            26778.*c2t) + 6.*pow(r,7.)*(-169753.+ 
            178509.*c2t) - 1400.*pow(r,3.)*(-39470.+ 
            296889.*c2t) + 1400.*pow(r,2.)*(-222133.+ 
            865731.*c2t) - 80.*pow(r,5.)*(-365951.+ 
            1009653.*c2t) + 90.*pow(r,4.)*(-501057.+ 
            1414481.*c2t) + pow(r,6.)*(-3428093.+ 
            22301529.*c2t)))/(110250.*pow((-2.+ r),3.)*pow(r,9.)) + 
       (ep4*(5788125.*pow(r,15.)+ 225.*pow(r,14.)*(-144901.+ 
            203178.*c2t) - 450.*pow(r,13.)*(-238789.+ 
            339067.*c2t) + 16464000000.*
           (-70.+ 585.*c2t+ 156.*c4t) + 
          98784000.*r*(48009.- 232978.*c2t+ 
            13748.*c4t) - 3292800.*pow(r,2.)*(1462459.- 
            5543725.*c2t+ 1597155.*c4t) + 
          30.*pow(r,12.)*(-9771723.- 3121232.*c2t+ 
            11639882.*c4t) - 30.*pow(r,11.)*(-3716939.- 
            59148640.*c2t+ 90116602.*c4t) - 
          28.*pow(r,6.)*(1537315067.- 11371159180.*c2t+ 
            343277600.*c4t) + 5600.*pow(r,3.)*(272959823.- 
            786639750.*c2t+ 347081820.*c4t) + 
          10.*pow(r,10.)*(-40457041.- 479621928.*c2t+ 
            930989658.*c4t) + 28.*pow(r,7.)*(-396696052.+ 
            2336490700.*c2t+ 1025187015.*c4t) + 
          560.*pow(r,4.)*(9892691.- 127506670.*c2t+ 
            1452290430.*c4t) - 280.*pow(r,5.)*(-78157011.+ 
            2296369510.*c2t+ 1738224390.*c4t) + 
          2.*pow(r,8.)*(7891214729.- 16305148340.*c2t+ 
            7026895110.*c4t) - 2.*pow(r,9.)*(1528530734.- 
            2049109850.*c2t+ 9100746145.*c4t)))/
        (30870000.*pow((-2.+ r),4.)*pow(r,13.)));

            gcov[2][2] = pow(r,2.)+ ep2*c2t+ 
     (zeta*(840.*pow(r,4.)*(8820000.- 6213200.*r - 3416700.*pow(r,2.)- 1855650.*pow(r,3.)+ 
          887110.*pow(r,4.)+ 800733.*pow(r,5.)+ 435540.*pow(r,6.)+ 187446.*pow(r,7.))*ep2*
         (1.- 3.*c2t) + ep4*
         (45715050.*pow(r,11.)*(-1.+ 3.*c2t) + 5625.*pow(r,10.)*
           (-20749.+ 58131.*c2t) + 493920000.*
           (-70.+ 585.*c2t+ 156.*c4t) + 
          24696000.*r*(3049.- 10698.*c2t+ 8868.*c4t) + 
          117600.*pow(r,2.)*(280331.- 1711445.*c2t+ 
            1596165.*c4t) + 180.*pow(r,9.)*(-1286466.- 
            846865.*c2t+ 5819941.*c4t) + 
          4200.*pow(r,3.)*(2362411.- 16650910.*c2t+ 
            14489100.*c4t) - 1260.*pow(r,4.)*(-3173281.- 
            5026080.*c2t+ 26477920.*c4t) + 
          42.*pow(r,8.)*(-18071967.- 940590.*c2t+ 
            54146980.*c4t) + 42.*pow(r,6.)*(-19116713.- 
            46592740.*c2t+ 138130070.*c4t) - 
          28.*pow(r,5.)*(11804979.- 261030540.*c2t+ 
            235282135.*c4t) + 6.*pow(r,7.)*(-259078241.- 
            99440670.*c2t+ 857000595.*c4t))))/
      (92610000.*pow(r,12.));

            gcov[3][3] = (1.- c2t)*
      (pow(r,2.)+ ep2+ (2.*r*ep2*(1.- c2t))/
        (pow(r,2.)+ ep2*c2t)) + 
     zeta*(((8820000.- 6213200.*r - 3416700.*pow(r,2.)- 1855650.*pow(r,3.)+ 
          887110.*pow(r,4.)+ 800733.*pow(r,5.)+ 435540.*pow(r,6.)+ 187446.*pow(r,7.))*ep2*
         (-1.+ c2t)*(-1.+ 3.*c2t))/(110250.*pow(r,8.)) - 
       (ep4*(-1.+ c2t)*
         (45715050.*pow(r,11.)*(-1.+ 3.*c2t) + 5625.*pow(r,10.)*
           (-20749.+ 58131.*c2t) + 493920000.*
           (-70.+ 585.*c2t+ 156.*c4t) + 
          24696000.*r*(3649.- 8958.*c2t+ 6528.*c4t) + 
          352800.*pow(r,2.)*(84857.- 350495.*c2t+ 
            320655.*c4t) + 12600.*pow(r,3.)*(-82303.- 
            1443030.*c2t+ 1592200.*c4t) + 
          180.*pow(r,9.)*(-411718.- 4345857.*c2t+ 
            8444185.*c4t) - 1260.*pow(r,4.)*(1578719.- 
            11450880.*c2t+ 28150720.*c4t) + 
          42.*pow(r,8.)*(-1863327.- 67980150.*c2t+ 
            104977900.*c4t) + 28.*pow(r,5.)*(-14247879.- 
            109560360.*c2t+ 137751665.*c4t) + 
          42.*pow(r,6.)*(30654807.- 316973820.*c2t+ 
            358739630.*c4t) + 6.*pow(r,7.)*(-25024421.- 
            1143700950.*c2t+ 1667207055.*c4t)))/
        (92610000.*pow(r,12.)));

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

// Bundle coordinates and transforms into umbrella variant types
// These act as a wannabe "interface" or "parent class" with the exception that access requires "mpark::visit"
// See coordinate_embedding.hpp

using SomeBaseCoords = mpark::variant<SphMinkowskiCoords, CartMinkowskiCoords, SphBLCoords, SphKSCoords, SphBLExtG, SphKSExtG, DCSKSCoords, DCSBLCoords, EDGBKSCoords, EDGBBLCoords>; // Changes Made. 
using SomeTransform = mpark::variant<NullTransform, ExponentialTransform, SuperExponentialTransform, ModifyTransform, FunkyTransform, SphNullTransform>;

// added SphNullTransform