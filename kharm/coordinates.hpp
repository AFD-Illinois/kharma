/*
 * coordinates.hpp:  A mess^(TM)
 */
#pragma once

#include "decs.hpp"
#include "matrix.hpp"

#include <mpark/variant.hpp>

#define COORDSINGFIX 1
#define SINGSMALL 1e-20

/**
 * Class defining properties mapping a native coordinate system to minkowski space
 */
class MinkowskiCoords {
    public:
        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[NDIM], Real gcov[NDIM][NDIM]) const
            {DLOOP2 gcov[mu][nu] = (mu == nu) - 2*(mu == 0 && nu == 0);}
        // TODO from_bl, rhor, etc that just error?
};

/**
 * Kerr-Schild coordinates as an embedding system
 */
class KSCoords {
    protected:
        // BH Spin is a property of KS
        const GReal a;

    public:
        KSCoords(GReal spin): a(spin) {};

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
class BLCoords {
    protected:
        // BH Spin is a property of BL
        const GReal a;

    public:
        BLCoords(GReal spin): a(spin) {};

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
class NullSphTransform {
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

class NullCartTransform {
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

// Lay out the coordinate systems and transforms we'll use
// Note some allowed combinations don't make sense, e.g. Minkowski+non-Null
class BLCoords;
class KSCoords;
class MinkowskiCoords;
using SomeBaseCoords = mpark::variant<BLCoords, KSCoords, MinkowskiCoords>;

class NullTransform;
// class ModifyTransform;
// class MoreModifyTransform;
// class FunkyModifyTransform;
// class MKS3Transform;
// typedef mpark::variant<NullTransform, ModifyTransform, MoreModifyTransform, FunkyModifyTransform, MKS3Transform> SomeTransform;
using SomeTransform = mpark::variant<NullSphTransform, NullCartTransform>;
