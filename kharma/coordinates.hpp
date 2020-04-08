/*
 * coordinates.hpp:  A mess^(TM)
 */
#pragma once

#include "decs.hpp"
#include "matrix.hpp"
#include "coordinate_embedding.hpp"

#include <mpark/variant.hpp>

#define COORDSINGFIX 1
#define SINGSMALL 1e-20

// Common function for non-invertible coordinate systems
template<typename Function>
void root_find(const GReal Xembed[NDIM], GReal Xnative[NDIM], Function coord_to_embed);


/**
 * EMBEDDING SYSTEMS:
 * These are the usual systems of coordinates for different spacetimes.
 * Each class must define at least gcov_embed, the metric in terms of their own coordinates Xembed
 * Some extra convenience classes have been defined for some 
 */

/**
 * Cartesian Coordinates over flat space.
 */
class CartMinkowskiCoords {
    public:
        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[NDIM], Real gcov[NDIM][NDIM]) const
            {DLOOP2 gcov[mu][nu] = (mu == nu) - 2*(mu == 0 && nu == 0);}
};

/**
 * Spherical coordinates for flat space
 * See docs common to all embeddings
 */
class SphMinkowskiCoords {
    public:
        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[NDIM], Real gcov[NDIM][NDIM]) const
            {} // TODO TODO TODO
};

/**
 * Spherical Kerr-Schild coordinates
 * See docs common to all embeddings
 */
class SphKSCoords {
    protected:
        // BH Spin is a property of KS
        const GReal a;

    public:
        SphKSCoords(GReal spin): a(spin) {};

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[NDIM], Real gcov[NDIM][NDIM]) const
        {
            GReal r = Xembed[1], th = Xembed[2];
            GReal sth, cth, s2, rho2;

            cth = cos(th);
            sth = sin(th);

            s2 = sth*sth;
            rho2 = r*r + a*a*cth*cth;

            gcov[0][0] = -1. + 2.*r/rho2;
            gcov[0][1] = 2.*r/rho2;
            gcov[0][2] = 0.;
            gcov[0][3] = -2.*a*r*s2/rho2;

            gcov[1][0] = gcov[0][1];
            gcov[1][1] = 1. + 2.*r/rho2;
            gcov[1][2] = 0.;
            gcov[1][3] = -a*s2*(1. + 2.*r/rho2);

            gcov[2][0] = 0.;
            gcov[2][1] = 0.;
            gcov[2][2] = rho2;
            gcov[2][3] = 0.;

            gcov[3][0] = gcov[0][3];
            gcov[3][1] = gcov[1][3];
            gcov[3][2] = 0.;
            gcov[3][3] = s2*(rho2 + a*a*s2*(1. + 2.*r/rho2));
        }

        // For converting from BL
        // TODO will we ever need a from_ks?
        KOKKOS_INLINE_FUNCTION void vec_from_bl(const GReal Xembed[NDIM], const Real vcon_bl[NDIM], Real vcon[NDIM]) const
        {
            GReal r = Xembed[1];
            Real trans[NDIM][NDIM];
            DLOOP2 trans[mu][nu] = (mu == nu);
            trans[0][1] = 2.*r/(r*r - 2.*r + a*a);
            trans[3][1] = a/(r*r - 2.*r + a*a);

            DLOOP1 vcon[mu] = 0.;
            DLOOP2 vcon[mu] += trans[mu][nu]*vcon_bl[nu];
        }

        // TODO more: isco etc.
        KOKKOS_INLINE_FUNCTION GReal rhor() const
        {
            return (1. + sqrt(1. - a*a));
        }
};

/**
 * Boyer-Lindquist coordinates as an embedding system
 */
class SphBLCoords {
    protected:
        // BH Spin is a property of BL
        const GReal a;

    public:
        SphBLCoords(GReal spin): a(spin) {}

        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[NDIM], Real gcov[NDIM][NDIM]) const
        {
            GReal r = Xembed[1], th = Xembed[2];
            Real sth, cth, s2, a2, r2, DD, mu;

            cth = cos(th);
            sth = sin(th);

            sth = fabs(sin(th));
            s2 = sth*sth;
            cth = cos(th);
            a2 = a*a;
            r2 = r*r;
            DD = 1. - 2./r + a2/r2;
            mu = 1. + a2*cth*cth/r2;

            DLOOP2 gcov[mu][nu] = 0.; // TODO spread for fewer ops?
            gcov[0][0]  = -(1. - 2./(r*mu));
            gcov[0][3]  = -2.*a*s2/(r*mu);
            gcov[3][0]  = gcov[0][3];
            gcov[1][1]   = mu/DD;
            gcov[2][2]   = r2*mu;
            gcov[3][3]   = r2*sth*sth*(1. + a2/r2 + 2.*a2*s2/(r2*r*mu));
        }

        KOKKOS_INLINE_FUNCTION void vec_from_bl(const GReal Xembed[NDIM], const Real vcon_bl[NDIM], Real vcon[NDIM]) const
        {
            DLOOP1 vcon[mu] = vcon_bl[mu];
        }

        KOKKOS_INLINE_FUNCTION GReal rhor() const
        {
            return (1. + sqrt(1. - a*a));
        }
};

/**
 * This class represents a null transformation from the embedding cooridnates, i.e. just using them directly
 */
class SphNullTransform {
    public:
        // Coordinate transformations
        // Protect embedding theta from ever hitting 0 or Pi
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[NDIM], GReal Xembed[NDIM]) const
        {
            Xembed[0] = Xnative[0];
            Xembed[1] = Xnative[1];
            GReal th = Xnative[2];
#if COORDSINGFIX
            if (fabs(th) < SINGSMALL) {
                if (th >= 0)
                    th = SINGSMALL;
                if (th < 0)
                    th = -SINGSMALL;
            }
            if (fabs(M_PI - th) < SINGSMALL) {
                if (th >= M_PI)
                    th = M_PI + SINGSMALL;
                if (th < M_PI)
                    th = M_PI - SINGSMALL;
            }
#endif
            Xembed[2] = th;
            Xembed[3] = Xnative[3];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[NDIM], GReal Xnative[NDIM]) const
        {
            DLOOP1 Xnative[mu] = Xembed[mu];
        }

        // Tangent space transformation matrices
        // TODO actual vec_to_embed, tensor_to_embed?
        KOKKOS_INLINE_FUNCTION void dxdX_to_embed(const GReal X[NDIM], Real dxdX[NDIM][NDIM]) const
        {
            DLOOP2 dxdX[mu][nu] = (mu == nu);
        }
        KOKKOS_INLINE_FUNCTION void dxdX_to_native(const GReal X[NDIM], Real dxdX[NDIM][NDIM]) const
        {
            DLOOP2 dxdX[mu][nu] = (mu == nu);
        }
};

class CartNullTransform {
    public:
        // This is an even simpler form of the above
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[NDIM], GReal Xembed[NDIM]) const
        {
            DLOOP1 Xembed[mu] = Xnative[mu];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[NDIM], GReal Xnative[NDIM]) const
        {
            DLOOP1 Xnative[mu] = Xembed[mu];
        }
        KOKKOS_INLINE_FUNCTION void dxdX_to_embed(const GReal X[NDIM], Real dxdX[NDIM][NDIM]) const
        {
            DLOOP2 dxdX[mu][nu] = (mu == nu);
        }
        KOKKOS_INLINE_FUNCTION void dxdX_to_native(const GReal X[NDIM], Real dxdX[NDIM][NDIM]) const
        {
            DLOOP2 dxdX[mu][nu] = (mu == nu);
        }
};

/**
 * "Funky" Modified Kerr-Schild coordinates
 * Make sense only for spherical base systems!
 */
class FunkyTransform {
    protected:
        const GReal startx1;
        const GReal hslope, poly_xt, poly_alpha, mks_smooth;
        GReal poly_norm; // TODO make this const and use a wrapper/factory to make these things?
    public:
        // Constructor
        FunkyTransform(GReal startx1_in, GReal hslope_in, GReal mks_smooth_in, GReal poly_xt_in, GReal poly_alpha_in):
            startx1(startx1_in), hslope(hslope_in), mks_smooth(mks_smooth_in), poly_xt(poly_xt_in), poly_alpha(poly_alpha_in)
            {
                poly_norm = 0.5 * M_PI * 1./(1. + 1./(poly_alpha + 1.) * 1./pow(poly_xt, poly_alpha));
            }

        // Coordinate transformations
        // Protect embedding theta from ever hitting 0 or Pi
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[NDIM], GReal Xembed[NDIM]) const
        {
            Xembed[0] = Xnative[0];
            Xembed[1] = exp(Xnative[1]);

            GReal thG = M_PI*Xnative[2] + ((1. - hslope)/2.)*sin(2.*M_PI*Xnative[2]);
            GReal y = 2*Xnative[2] - 1.;
            GReal thJ = poly_norm * y * (1. + pow(y/poly_xt,poly_alpha) / (poly_alpha + 1.)) + 0.5 * M_PI;

            GReal th = thG + exp(mks_smooth * (startx1 - Xnative[1])) * (thJ - thG);
#if COORDSINGFIX
            if (fabs(th) < SINGSMALL) {
                if (th >= 0)
                    th = SINGSMALL;
                if (th < 0)
                    th = -SINGSMALL;
            }
            if (fabs(M_PI - th) < SINGSMALL) {
                if (th >= M_PI)
                    th = M_PI + SINGSMALL;
                if (th < M_PI)
                    th = M_PI - SINGSMALL;
            }
#endif
            Xembed[2] = th;
            Xembed[3] = Xnative[3];
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[NDIM], GReal Xnative[NDIM]) const
        {
            Xnative[0] = Xembed[0];
            Xnative[1] = log(Xembed[1]);
            Xnative[3] = Xembed[3];
            // Treat the special case
            root_find(Xembed, Xnative, &FunkyTransform::coord_to_embed);
        }

        /**
         * Transformation matrix for vectors (i.e. Jacobian) from the embedding to native coordinate system
         */
        KOKKOS_INLINE_FUNCTION void dxdX_to_native(const GReal Xnative[NDIM], Real dxdX[NDIM][NDIM]) const
        {
            DLOOP2 dxdX[mu][nu] = 0;
            dxdX[0][0] = 1.;
            dxdX[1][1] = exp(Xnative[1]);
            dxdX[2][1] = -exp(mks_smooth * (startx1 - Xnative[1])) * mks_smooth
                * (
                M_PI / 2. -
                M_PI * Xnative[2]
                    + poly_norm * (2. * Xnative[2] - 1.)
                        * (1
                            + (pow((-1. + 2 * Xnative[2]) / poly_xt, poly_alpha))
                                / (1 + poly_alpha))
                    - 1. / 2. * (1. - hslope) * sin(2. * M_PI * Xnative[2]));
            dxdX[2][2] = M_PI + (1. - hslope) * M_PI * cos(2. * M_PI * Xnative[2])
                + exp(mks_smooth * (startx1 - Xnative[1]))
                    * (-M_PI
                        + 2. * poly_norm
                            * (1.
                                + pow((2. * Xnative[2] - 1.) / poly_xt, poly_alpha)
                                    / (poly_alpha + 1.))
                        + (2. * poly_alpha * poly_norm * (2. * Xnative[2] - 1.)
                            * pow((2. * Xnative[2] - 1.) / poly_xt, poly_alpha - 1.))
                            / ((1. + poly_alpha) * poly_xt)
                        - (1. - hslope) * M_PI * cos(2. * M_PI * Xnative[2]));
            dxdX[3][3] = 1.;
        }
        /**
         * Transformation matrix for vectors from the native back to the embedding coordinate system
         * Note the given coordinates should still be native
         */
        KOKKOS_INLINE_FUNCTION void dxdX_to_embed(const GReal Xnative[NDIM], Real dxdX[NDIM][NDIM]) const
        {
            // Lazy way.  Surely there's an analytic inverse to be had somewhere...
            Real dxdX_inv[NDIM][NDIM];
            dxdX_to_native(Xnative, dxdX_inv);
            invert(&dxdX_inv[0][0],&dxdX[0][0]);
        }
};

// Bundle coordinates and transforms into umbrella variant types
// I am 99% certain there's a way to add CoordinateEmbedding to this list.
using SomeBaseCoords = mpark::variant<SphMinkowskiCoords, CartMinkowskiCoords, SphBLCoords, SphKSCoords>;
using SomeTransform = mpark::variant<SphNullTransform, CartNullTransform, FunkyTransform>;

/**
 * Root finder for X[2] since it is sometimes not analytically invertible
 * Written so it can be extended if people have very crazy coordinate ideas that require 2D solves
 * @param Xembed the vector of embedding coordinates to convert
 * @param Xnative output; but should have all native coordinates except X[2] already
 * @param coord_to_embed function taking the vector X to embedding coordinates
 */
template<typename Function>
void root_find(const GReal Xembed[NDIM], GReal Xnative[NDIM], Function& coord_to_embed)
{
  double th = Xembed[2];
  double tha, thb, thc;

  // Currently only solves in X[2] but could be multi-dimensional
  double Xa[NDIM], Xb[NDIM], Xc[NDIM], Xtmp[NDIM];
  Xa[1] = Xnative[1];
  Xa[3] = Xnative[3];

  Xb[1] = Xa[1];
  Xb[3] = Xa[3];
  Xc[1] = Xa[1];
  Xc[3] = Xa[3];

  if (Xembed[2] < M_PI / 2.) {
    Xa[2] = 0.;
    Xb[2] = 0.5 + SMALL;
  } else {
    Xa[2] = 0.5 - SMALL;
    Xb[2] = 1.;
  }

  double tol = 1.e-9;
  coord_to_embed(Xa, Xtmp);
  tha = Xtmp[2];
  coord_to_embed(Xb, Xtmp);
  thb = Xtmp[2];

  // check limits first
  if (fabs(tha-th) < tol) {
    Xnative[2] = Xa[2];
    return;
  } else if (fabs(thb-th) < tol) {
    Xnative[2] = Xb[2];
    return;
  }

  // bisect for a bit
  for (int i = 0; i < 1000; i++) {
    Xc[2] = 0.5 * (Xa[2] + Xb[2]);
    coord_to_embed(Xc, Xtmp);
    thc = Xtmp[2];

    if ((thc - th) * (thb - th) < 0.)
      Xa[2] = Xc[2];
    else
      Xb[2] = Xc[2];
    
    if (fabs(thc - th) < tol)
      break;
  }

  Xnative[2] = Xc[2];
}