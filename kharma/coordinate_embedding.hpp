/*
 * coordinate_embedding.hpp:  An object to handle transformations to/from native coordinates in GR
 */
#pragma once

#include "coordinate_systems.hpp"
#include "decs.hpp"
#include "matrix.hpp"

/**
 * Coordinates in HARM are logically Cartesian -- that is, in some coordinate system they are evenly spaced
 * However, working in GR allows us to define that "native" or "transformed" coordinate system arbitrarily in relation to the
 * "base" or "embedding" coordinates, usually Spherical Kerr-Schild coordinates
 *
 * That is, as long as we have a bijective map of base<->transformed coordinates, we can define the latter arbitrarily, which is
 * great for putting resolution where we need and not where we don't.
 *
 * This class keeps track of the base coordinates and the map, which must be classes which are members of the SomeXX std::variant containers.
 * The BaseCoords class must implement:
 * * gcov_embed
 * The Transform class must implement:
 * * coord_to_embed
 * * coord_to_native
 * * dxdX_to_embed
 * * dxdX_to_native
 *
 * TODO put lower/raise in here?
 */
class CoordinateEmbedding {
    public:
        SomeBaseCoords base;
        SomeTransform transform;

        CoordinateEmbedding(SomeBaseCoords& base_in, SomeTransform& transform_in): base(base_in), transform(transform_in) {}
        // Alternate versions if some types have default constructors
        CoordinateEmbedding(SomeBaseCoords& base_in): base(base_in)
        {
            // TODO detect Sph/Cart based on base type
            // TODO eliminate these calls when the transform is null?  Maybe useful for kipole
            transform = SphNullTransform();
        }

        // Spell out the interface we take from BaseCoords
        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[NDIM], Real gcov[NDIM][NDIM]) const
        {
            mpark::visit( [&Xembed, &gcov](const auto& self) {
                self.gcov_embed(Xembed, gcov);
            }, base);
        }
        // and from the Transform
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[NDIM], GReal Xembed[NDIM]) const
        {
            mpark::visit( [&Xnative, &Xembed](const auto& self) {
                self.coord_to_embed(Xnative, Xembed);
            }, transform);
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[NDIM], GReal Xnative[NDIM]) const
        {
            mpark::visit( [&Xnative, &Xembed](const auto& self) {
                self.coord_to_native(Xembed, Xnative);
            }, transform);
        }
        KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[NDIM], Real dxdX[NDIM][NDIM]) const
        {
            mpark::visit( [&Xnative, &dxdX](const auto& self) {
                self.dxdX(Xnative, dxdX);
            }, transform);
        }
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[NDIM], Real dXdx[NDIM][NDIM]) const
        {
            mpark::visit( [&Xnative, &dXdx](const auto& self) {
                self.dXdx(Xnative, dXdx);
            }, transform);
        }

        // VECTOR TRANSFORMS
        // Contravariant vectors:
        KOKKOS_INLINE_FUNCTION void con_vec_to_embed(const GReal Xnative[NDIM], const GReal vcon_native[NDIM], GReal vcon_embed[NDIM]) const
        {
            Real dxdX_temp[NDIM][NDIM];
            dxdX(Xnative, dxdX_temp);
            DLOOP1 {
                vcon_embed[mu] = 0;
                for(int nu=0; nu < NDIM; ++nu)
                    vcon_embed[mu] += dxdX_temp[mu][nu] * vcon_native[nu];
            }
        }
        KOKKOS_INLINE_FUNCTION void con_vec_to_native(const GReal Xnative[NDIM], const GReal vcon_embed[NDIM], GReal vcon_native[NDIM]) const
        {
            Real dXdx_temp[NDIM][NDIM];
            dXdx(Xnative, dXdx_temp);
            DLOOP1 { // TODO is this faster, or DLOOP1/2?
                vcon_native[mu] = 0;
                for(int nu=0; nu < NDIM; ++nu)
                    vcon_native[mu] += dXdx_temp[mu][nu] * vcon_embed[nu];
            }
        }
        // Covariant are opposite
        KOKKOS_INLINE_FUNCTION void cov_vec_to_native(const GReal Xnative[NDIM], const GReal vcov_embed[NDIM], GReal vcov_native[NDIM]) const
            {con_vec_to_embed(Xnative, vcov_embed, vcov_native);}
        KOKKOS_INLINE_FUNCTION void cov_vec_to_embed(const GReal Xnative[NDIM], const GReal vcov_native[NDIM], GReal vcov_embed[NDIM]) const
            {con_vec_to_native(Xnative, vcov_native, vcov_embed);}

        // TENSOR TRANSFORMS
        // Covariant first
        KOKKOS_INLINE_FUNCTION void cov_tensor_to_embed(const GReal Xnative[NDIM], const GReal tcov_native[NDIM][NDIM], GReal tcov_embed[NDIM][NDIM]) const
        {
            Real dXdx_temp[NDIM][NDIM];
            dXdx(Xnative, dXdx_temp);

            DLOOP2 {
                tcov_embed[mu][nu] = 0;
                for (int lam = 0; lam < NDIM; ++lam) {
                    for (int kap = 0; kap < NDIM; ++kap) {
                        tcov_embed[mu][nu] += tcov_native[lam][kap]*dXdx_temp[lam][mu]*dXdx_temp[kap][nu];
                    }
                }
            }
        }
        KOKKOS_INLINE_FUNCTION void cov_tensor_to_native(const GReal Xnative[NDIM], const GReal tcov_embed[NDIM][NDIM], GReal tcov_native[NDIM][NDIM]) const
        {
            Real dxdX_temp[NDIM][NDIM];
            dxdX(Xnative, dxdX_temp);

            DLOOP2 {
                tcov_native[mu][nu] = 0;
                for (int lam = 0; lam < NDIM; lam++) {
                    for (int kap = 0; kap < NDIM; kap++) {
                        tcov_native[mu][nu] += tcov_embed[lam][kap]*dxdX_temp[lam][mu]*dxdX_temp[kap][nu];
                    }
                }
            }
        }
        // Con are opposite
        KOKKOS_INLINE_FUNCTION void con_tensor_to_embed(const GReal Xnative[NDIM], const GReal tcon_native[NDIM][NDIM], GReal tcon_embed[NDIM][NDIM]) const
            {cov_tensor_to_native(Xnative, tcon_native, tcon_embed);}
        KOKKOS_INLINE_FUNCTION void con_tensor_to_native(const GReal Xnative[NDIM], const GReal tcon_embed[NDIM][NDIM], GReal tcon_native[NDIM][NDIM]) const
            {cov_tensor_to_embed(Xnative, tcon_embed, tcon_native);}

        // And then some metric properties
        // TODO gcon_embed, gdet_embed
        KOKKOS_INLINE_FUNCTION void gcov_native(const GReal Xnative[NDIM], Real gcov[NDIM][NDIM]) const
        {
            Real gcov_em[NDIM][NDIM];
            GReal Xembed[NDIM];
            // Get coordinates in embedding system
            coord_to_embed(Xnative, Xembed);

            // Get metric in embedding coordinates
            gcov_embed(Xembed, gcov_em);

            // Transform to native coordinates
            cov_tensor_to_native(Xnative, gcov_em, gcov);
        }
        KOKKOS_INLINE_FUNCTION Real gcon_native(const GReal X[NDIM], Real gcon[NDIM][NDIM]) const
        {
            Real gcov[NDIM][NDIM];
            gcov_native(X, gcov);
            return gcon_native(gcov, gcon);
        }
        KOKKOS_INLINE_FUNCTION Real gcon_native(const Real gcov[NDIM][NDIM], Real gcon[NDIM][NDIM]) const
        {
            Real gdet = invert(&gcov[0][0], &gcon[0][0]);
            return sqrt(fabs(gdet));
        }



        KOKKOS_INLINE_FUNCTION void conn_native(const GReal X[NDIM], Real conn[NDIM][NDIM][NDIM]) const
        {
            Real tmp[NDIM][NDIM][NDIM];
            Real gcon[NDIM][NDIM];
            GReal Xh[NDIM], Xl[NDIM];
            Real gh[NDIM][NDIM];
            Real gl[NDIM][NDIM];

            for (int nu = 0; nu < NDIM; nu++) {
                DLOOP1 Xh[mu] = Xl[mu] = X[mu];
                Xh[nu] += DELTA;
                Xl[nu] -= DELTA;
                gcov_native(Xh, gh);
                gcov_native(Xl, gl);

                for (int lam = 0; lam < NDIM; lam++) {
                    for (int kap = 0; kap < NDIM; kap++) {
                        conn[lam][kap][nu] = (gh[lam][kap] - gl[lam][kap])/
                                                        (Xh[nu] - Xl[nu]);
                    }
                }
            }

            // Rearrange to find \Gamma_{lam nu mu}
            for (int lam = 0; lam < NDIM; lam++) {
                for (int nu = 0; nu < NDIM; nu++) {
                    for (int mu = 0; mu < NDIM; mu++) {
                        tmp[lam][nu][mu] = 0.5 * (conn[nu][lam][mu] +
                                                  conn[mu][lam][nu] -
                                                  conn[mu][nu][lam]);
                    }
                }
            }

            // Need gcon for raising index
            gcon_native(X, gcon);

            // Raise index to get \Gamma^lam_{nu mu}
            for (int lam = 0; lam < NDIM; lam++) {
                for (int nu = 0; nu < NDIM; nu++) {
                    for (int mu = 0; mu < NDIM; mu++) {
                        conn[lam][nu][mu] = 0.;

                        for (int kap = 0; kap < NDIM; kap++)
                            conn[lam][nu][mu] += gcon[lam][kap] * tmp[kap][nu][mu];
                    }
                }
            }
        }
};