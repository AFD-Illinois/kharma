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
 * TODO CMKS, MKS3 transforms, proper Cartesian<->Spherical functions stolen from e.g. prob_common.hpp
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

/**
 * Cartesian Coordinates over flat space.
 */
class CartMinkowskiCoords {
    public:
        const bool spherical = false;
        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            DLOOP2 gcov[mu][nu] = (mu == nu) - 2*(mu == 0 && nu == 0);
        }
};

/**
 * Spherical coordinates for flat space
 */
class SphMinkowskiCoords {
    public:
        const bool spherical = true;
        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = m::max(Xembed[1], SMALL);
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            const GReal sth = sin(th);

            gzero2(gcov);
            gcov[0][0] = 1.;
            gcov[1][1] = 1.;
            gcov[2][2] = r*r;
            gcov[3][3] = m::pow(sth*r, 2);
        }
};

/**
 * Spherical Kerr-Schild coordinates
 */
class SphKSCoords {
    public:
        // BH Spin is a property of KS
        const GReal a;
        const bool spherical = true;
        const bool ext_g; // added by Hyerin (02/27/23)

        KOKKOS_FUNCTION SphKSCoords(GReal spin, bool external_gravity): a(spin), ext_g(external_gravity) {};

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);

            const GReal cos2 = m::pow(cos(th), 2);
            const GReal sin2 = m::pow(sin(th), 2);
            const GReal rho2 = r*r + a*a*cos2;
            
            // (Hyerin 11/13/22) test
            const GReal A = 4.24621057 * m::pow(10.,-9); //1.46797639*m::pow(10.,-8);
            const GReal B = 1.35721335; //1.29411117;
            const GReal Phi_g = (A/(B-1.)) * (m::pow(r,B-1.)-m::pow(2,B-1.));

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

            // Hyerin TODO: add an error when spin != 0
            if (ext_g) {
                if (a>0) printf("WARNING: External gravity is not compatible with nonzero spin! \n");
                gcov[0][0] -= 2. * Phi_g;
                gcov[0][1] -= 2. * Phi_g;
                gcov[1][0] -= 2. * Phi_g;
                gcov[1][1] -= 2. * Phi_g;
            }
        }

        // For converting from BL
        // TODO will we ever need a from_ks?
        KOKKOS_INLINE_FUNCTION void vec_from_bl(const GReal Xembed[GR_DIM], const Real vcon_bl[GR_DIM], Real vcon[GR_DIM]) const
        {
            GReal r = Xembed[1];
            Real trans[GR_DIM][GR_DIM];
            DLOOP2 trans[mu][nu] = (mu == nu);
            trans[0][1] = 2.*r/(r*r - 2.*r + a*a);
            trans[3][1] = a/(r*r - 2.*r + a*a);

            // external gravity from GIZMO
            const GReal A = 4.24621057 * m::pow(10.,-9); //1.46797639*m::pow(10.,-8);
            const GReal B = 1.35721335; //1.29411117;
            const GReal Phi_g = (A/(B-1.)) * (m::pow(r,B-1.)-m::pow(2,B-1.));

            if (ext_g) {
                trans[0][1] = (2./r - 2.*Phi_g)/(1. - 2./r + 2.*Phi_g);
            }

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

            // external gravity from GIZMO
            const GReal A = 4.24621057 * m::pow(10.,-9); //1.46797639*m::pow(10.,-8);
            const GReal B = 1.35721335; //1.29411117;
            const GReal Phi_g = (A/(B-1.)) * (m::pow(r,B-1.)-m::pow(2,B-1.));
            
            if (ext_g) {
                rtrans[0][1] = (2./r - 2.*Phi_g)/(1. - 2./r + 2.*Phi_g);
            }
            invert(&rtrans[0][0], &trans[0][0]);

            gzero(vcon);
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
        }

        // TODO more: isco etc?
        KOKKOS_INLINE_FUNCTION GReal rhor() const
        {
            return (1. + m::sqrt(1. - a*a));
        }
};

/**
 * Boyer-Lindquist coordinates as an embedding system
 */
class SphBLCoords {
    public:
        // BH Spin is a property of BL
        const GReal a;
        const bool spherical = true;
        const bool ext_g; // added by Hyerin (11/13/22)

        KOKKOS_FUNCTION SphBLCoords(GReal spin, bool external_gravity): a(spin), ext_g(external_gravity) {}

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            const GReal r = Xembed[1];
            const GReal th = excise(excise(Xembed[2], 0.0, SMALL), M_PI, SMALL);
            const GReal cth = cos(th), sth = sin(th);

            const GReal s2 = sth*sth;
            const GReal a2 = a*a;
            const GReal r2 = r*r;
            // TODO this and gcov_embed for KS should look more similar...
            const GReal mmu = 1. + a2*cth*cth/r2; // mu is taken as an index

            // (Hyerin 11/13/22) test
            const GReal A = 4.24621057 * m::pow(10.,-9); //1.46797639*m::pow(10.,-8);
            const GReal B = 1.35721335; //1.29411117;
            const GReal Phi_g = (A/(B-1.)) * (m::pow(r,B-1.)-m::pow(2,B-1.));

            gzero2(gcov);
            gcov[0][0]  = -(1. - 2./(r*mmu));
            gcov[0][3]  = -2.*a*s2/(r*mmu);
            gcov[1][1]   = mmu/(1. - 2./r + a2/r2);
            gcov[2][2]   = r2*mmu;
            gcov[3][0]  = -2.*a*s2/(r*mmu);
            gcov[3][3]   = s2*(r2 + a2 + 2.*a2*s2/(r*mmu));

            // Hyerin TODO: add an error when spin != 0 
            if (ext_g) {
                if (a>0) printf("WARNING: External gravity is not compatible with nonzero spin! \n");
                gcov[0][0] -= 2. * Phi_g;
                gcov[1][1] *= (1. - 2./r + a2/r2) / (1. - 2./r + 2.*Phi_g);
            }
        }

        // TODO vec to/from ks, put guaranteed ks/bl fns into embedding

        KOKKOS_INLINE_FUNCTION GReal rhor() const
        {
            return (1. + m::sqrt(1. - a*a));
        }
};

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
        // Coordinate transformations
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            Xembed[0] = Xnative[0];
            Xembed[1] = exp(Xnative[1]);
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
            Xnative[1] = log(Xembed[1]);
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
            dxdX[1][1] = exp(Xnative[1]);
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
            dXdx[1][1] = 1 / exp(Xnative[1]);
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
        const GReal hslope;

        // Constructor
        KOKKOS_FUNCTION ModifyTransform(GReal hslope_in): hslope(hslope_in) {}

        // Coordinate transformations
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            Xembed[0] = Xnative[0];
            Xembed[1] = exp(Xnative[1]);
#if LEGACY_TH
            const GReal th = M_PI*Xnative[2] + ((1. - hslope)/2.)*sin(2.*M_PI*Xnative[2]);
            Xembed[2] = excise(excise(th, 0.0, SMALL), M_PI, SMALL);
#else
            Xembed[2] = M_PI*Xnative[2] + ((1. - hslope)/2.)*sin(2.*M_PI*Xnative[2]);
#endif
            Xembed[3] = Xnative[3];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
        {
            Xnative[0] = Xembed[0];
            Xnative[1] = log(Xembed[1]);
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
            dxdX[1][1] = exp(Xnative[1]);
            dxdX[2][2] = M_PI - (hslope - 1.)*M_PI*cos(2.*M_PI*Xnative[2]);
            dxdX[3][3] = 1.;
        }
        /**
         * Transformation matrix for contravariant vectors to native, or covariant vectors to embedding
         */
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
        {
            gzero2(dXdx);
            dXdx[0][0] = 1.;
            dXdx[1][1] = 1 / exp(Xnative[1]);
            dXdx[2][2] = 1 / (M_PI - (hslope - 1.)*M_PI*cos(2.*M_PI*Xnative[2]));
            dXdx[3][3] = 1.;
        }
};

/**
 * "Funky" Modified Kerr-Schild coordinates
 * Make sense only for spherical base systems!
 */
class FunkyTransform {
    public:
        const GReal startx1;
        const GReal hslope, poly_xt, poly_alpha, mks_smooth;
        GReal poly_norm; // TODO make this const and use a wrapper/factory to make these things?

        // Constructor
        KOKKOS_FUNCTION FunkyTransform(GReal startx1_in, GReal hslope_in, GReal mks_smooth_in, GReal poly_xt_in, GReal poly_alpha_in):
            startx1(startx1_in), hslope(hslope_in), mks_smooth(mks_smooth_in), poly_xt(poly_xt_in), poly_alpha(poly_alpha_in)
            {
                poly_norm = 0.5 * M_PI * 1./(1. + 1./(poly_alpha + 1.) * 1./m::pow(poly_xt, poly_alpha));
            }

        // Coordinate transformations
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            Xembed[0] = Xnative[0];
            Xembed[1] = exp(Xnative[1]);

            const GReal thG = M_PI*Xnative[2] + ((1. - hslope)/2.)*sin(2.*M_PI*Xnative[2]);
            const GReal y = 2*Xnative[2] - 1.;
            const GReal thJ = poly_norm * y * (1. + m::pow(y/poly_xt,poly_alpha) / (poly_alpha + 1.)) + 0.5 * M_PI;
#if LEGACY_TH
            const GReal th = thG + exp(mks_smooth * (startx1 - Xnative[1])) * (thJ - thG);
            Xembed[2] = excise(excise(th, 0.0, SMALL), M_PI, SMALL);
#else
            Xembed[2] = thG + exp(mks_smooth * (startx1 - Xnative[1])) * (thJ - thG);
#endif
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
            dxdX[2][1] = -exp(mks_smooth * (startx1 - Xnative[1])) * mks_smooth
                * (
                M_PI / 2. -
                M_PI * Xnative[2]
                    + poly_norm * (2. * Xnative[2] - 1.)
                        * (1
                            + (m::pow((-1. + 2 * Xnative[2]) / poly_xt, poly_alpha))
                                / (1 + poly_alpha))
                    - 1. / 2. * (1. - hslope) * sin(2. * M_PI * Xnative[2]));
            dxdX[2][2] = M_PI + (1. - hslope) * M_PI * cos(2. * M_PI * Xnative[2])
                + exp(mks_smooth * (startx1 - Xnative[1]))
                    * (-M_PI
                        + 2. * poly_norm
                            * (1.
                                + m::pow((2. * Xnative[2] - 1.) / poly_xt, poly_alpha)
                                    / (poly_alpha + 1.))
                        + (2. * poly_alpha * poly_norm * (2. * Xnative[2] - 1.)
                            * m::pow((2. * Xnative[2] - 1.) / poly_xt, poly_alpha - 1.))
                            / ((1. + poly_alpha) * poly_xt)
                        - (1. - hslope) * M_PI * cos(2. * M_PI * Xnative[2]));
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
        const GReal lin_frac, n2, n3;
        GReal smoothness;

        // Constructor
        KOKKOS_FUNCTION WidepoleTransform(GReal lin_frac_in, GReal n2_in, GReal n3_in): lin_frac(lin_frac_in), n2(n2_in), n3(n3_in) 
        {
            GReal n3_temp = n3;
            if (n3 < M_PI) n3_temp = n2;
            GReal temp;
            if (lin_frac == 1) temp = 1.;
            else temp = lin_frac / (1. - lin_frac) * (1. / M_PI - 1. / n3_temp) * n3_temp / n2;
            if (abs(temp) < 1) smoothness = 1. / (n2 * log((1. + temp) / (1. - temp)));
            else {
                printf("WARNING: It is harder to have del phi ~ del th. Try using lin_frac < %g \n",  1./ ((1. / M_PI - 1./ n3_temp) * n3_temp / n2 + 1.));
                smoothness = 0.8 / n2;
            }
            smoothness = m::max(0.01, smoothness); // fix it for now for test
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
// Note nesting isn't allowed -- do it yourself by calling the steps if that's really important...
using SomeBaseCoords = mpark::variant<SphMinkowskiCoords, CartMinkowskiCoords, SphBLCoords, SphKSCoords>;
using SomeTransform = mpark::variant<NullTransform, ExponentialTransform, SuperExponentialTransform, ModifyTransform, FunkyTransform, WidepoleTransform>;
