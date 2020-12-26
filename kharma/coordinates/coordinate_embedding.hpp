/*
 * coordinate_embedding.hpp:  An object to handle transformations to/from native coordinates in GR
 */
#pragma once

#include "decs.hpp"

#include "coordinate_systems.hpp"
#include "matrix.hpp"

using namespace std;

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
 * TODO keep notion of whether base coords are spherical, use to return guaranteed r,th,phi or x,y,z
 * TODO implement an rhor() that calls through or returns 0? Or make required callthrough
 */
class CoordinateEmbedding {
    public:
        SomeBaseCoords base;
        SomeTransform transform;

        // Common code for constructors
#pragma hd_warning_disable
        KOKKOS_FUNCTION void EmplaceSystems(const SomeBaseCoords& base_in, const SomeTransform& transform_in) {
            // Isn't there some more elegant way to say "yeah the types are fine just copy da bits"?
            if (mpark::holds_alternative<SphMinkowskiCoords>(base_in)) {
                base.emplace<SphMinkowskiCoords>(mpark::get<SphMinkowskiCoords>(base_in));
            } else if (mpark::holds_alternative<CartMinkowskiCoords>(base_in)) {
                base.emplace<CartMinkowskiCoords>(mpark::get<CartMinkowskiCoords>(base_in));
            } else if (mpark::holds_alternative<SphBLCoords>(base_in)) {
                base.emplace<SphBLCoords>(mpark::get<SphBLCoords>(base_in));
            } else if (mpark::holds_alternative<SphKSCoords>(base_in)) {
                base.emplace<SphKSCoords>(mpark::get<SphKSCoords>(base_in));
            } else {
                printf("Tried to copy invalid base coordinates!");
                //throw std::invalid_argument("Tried to copy invalid base coordinates!");
            }

            if (mpark::holds_alternative<SphNullTransform>(transform_in)) {
                transform.emplace<SphNullTransform>(mpark::get<SphNullTransform>(transform_in));
            } else if (mpark::holds_alternative<CartNullTransform>(transform_in)) {
                transform.emplace<CartNullTransform>(mpark::get<CartNullTransform>(transform_in));
            } else if (mpark::holds_alternative<ModifyTransform>(transform_in)) {
                transform.emplace<ModifyTransform>(mpark::get<ModifyTransform>(transform_in));
            } else if (mpark::holds_alternative<FunkyTransform>(transform_in)) {
                transform.emplace<FunkyTransform>(mpark::get<FunkyTransform>(transform_in));
            } else {
                printf("Tried to copy invalid coordinate transform!");
                //throw std::invalid_argument("Tried to copy invalid coordinate transform!");
            }
        }

        // Constructors
#pragma hd_warning_disable
        CoordinateEmbedding() = default;
#pragma hd_warning_disable
        KOKKOS_FUNCTION CoordinateEmbedding(SomeBaseCoords& base_in, SomeTransform& transform_in): base(base_in), transform(transform_in) {}
#pragma hd_warning_disable
        KOKKOS_FUNCTION CoordinateEmbedding(const CoordinateEmbedding& src)
        {
            EmplaceSystems(src.base, src.transform);
        }
#pragma hd_warning_disable
        KOKKOS_FUNCTION const CoordinateEmbedding& operator=(const CoordinateEmbedding& src)
        {
            EmplaceSystems(src.base, src.transform);
            return *this;
        }

        // Spell out the interface we take from BaseCoords
        KOKKOS_INLINE_FUNCTION bool spherical() const
        {
            return mpark::visit( [&](const auto& self) {
                return self.spherical;
            }, base);
        }
        KOKKOS_INLINE_FUNCTION void gcov_embed(const GReal Xembed[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            mpark::visit( [&Xembed, &gcov](const auto& self) {
                self.gcov_embed(Xembed, gcov);
            }, base);
        }
        // and from the Transform
        KOKKOS_INLINE_FUNCTION void coord_to_embed(const GReal Xnative[GR_DIM], GReal Xembed[GR_DIM]) const
        {
            mpark::visit( [&Xnative, &Xembed](const auto& self) {
                self.coord_to_embed(Xnative, Xembed);
            }, transform);
        }
        KOKKOS_INLINE_FUNCTION void coord_to_native(const GReal Xembed[GR_DIM], GReal Xnative[GR_DIM]) const
        {
            mpark::visit( [&Xnative, &Xembed](const auto& self) {
                self.coord_to_native(Xembed, Xnative);
            }, transform);
        }
        KOKKOS_INLINE_FUNCTION void dxdX(const GReal Xnative[GR_DIM], Real dxdX[GR_DIM][GR_DIM]) const
        {
            mpark::visit( [&Xnative, &dxdX](const auto& self) {
                self.dxdX(Xnative, dxdX);
            }, transform);
        }
        KOKKOS_INLINE_FUNCTION void dXdx(const GReal Xnative[GR_DIM], Real dXdx[GR_DIM][GR_DIM]) const
        {
            mpark::visit( [&Xnative, &dXdx](const auto& self) {
                self.dXdx(Xnative, dXdx);
            }, transform);
        }

        // VECTOR TRANSFORMS
        // Contravariant vectors:
        KOKKOS_INLINE_FUNCTION void con_vec_to_embed(const GReal Xnative[GR_DIM], const GReal vcon_native[GR_DIM], GReal vcon_embed[GR_DIM]) const
        {
            Real dxdX_temp[GR_DIM][GR_DIM];
            dxdX(Xnative, dxdX_temp);
            DLOOP1 {
                vcon_embed[mu] = 0;
                for(int nu=0; nu < GR_DIM; ++nu)
                    vcon_embed[mu] += dxdX_temp[mu][nu] * vcon_native[nu];
            }
        }
        KOKKOS_INLINE_FUNCTION void con_vec_to_native(const GReal Xnative[GR_DIM], const GReal vcon_embed[GR_DIM], GReal vcon_native[GR_DIM]) const
        {
            Real dXdx_temp[GR_DIM][GR_DIM];
            dXdx(Xnative, dXdx_temp);
            DLOOP1 { // TODO is this faster, or DLOOP1/2?
                vcon_native[mu] = 0;
                for(int nu=0; nu < GR_DIM; ++nu)
                    vcon_native[mu] += dXdx_temp[mu][nu] * vcon_embed[nu];
            }
        }
        // Covariant are opposite
        KOKKOS_INLINE_FUNCTION void cov_vec_to_native(const GReal Xnative[GR_DIM], const GReal vcov_embed[GR_DIM], GReal vcov_native[GR_DIM]) const
            {con_vec_to_embed(Xnative, vcov_embed, vcov_native);}
        KOKKOS_INLINE_FUNCTION void cov_vec_to_embed(const GReal Xnative[GR_DIM], const GReal vcov_native[GR_DIM], GReal vcov_embed[GR_DIM]) const
            {con_vec_to_native(Xnative, vcov_native, vcov_embed);}

        // TENSOR TRANSFORMS
        // Covariant first
        KOKKOS_INLINE_FUNCTION void cov_tensor_to_embed(const GReal Xnative[GR_DIM], const GReal tcov_native[GR_DIM][GR_DIM], GReal tcov_embed[GR_DIM][GR_DIM]) const
        {
            Real dXdx_temp[GR_DIM][GR_DIM];
            dXdx(Xnative, dXdx_temp);

            DLOOP2 {
                tcov_embed[mu][nu] = 0;
                for (int lam = 0; lam < GR_DIM; ++lam) {
                    for (int kap = 0; kap < GR_DIM; ++kap) {
                        tcov_embed[mu][nu] += tcov_native[lam][kap]*dXdx_temp[lam][mu]*dXdx_temp[kap][nu];
                    }
                }
            }
        }
        KOKKOS_INLINE_FUNCTION void cov_tensor_to_native(const GReal Xnative[GR_DIM], const GReal tcov_embed[GR_DIM][GR_DIM], GReal tcov_native[GR_DIM][GR_DIM]) const
        {
            Real dxdX_temp[GR_DIM][GR_DIM];
            dxdX(Xnative, dxdX_temp);

            DLOOP2 {
                tcov_native[mu][nu] = 0;
                for (int lam = 0; lam < GR_DIM; lam++) {
                    for (int kap = 0; kap < GR_DIM; kap++) {
                        tcov_native[mu][nu] += tcov_embed[lam][kap]*dxdX_temp[lam][mu]*dxdX_temp[kap][nu];
                    }
                }
            }
        }
        // Con are opposite
        KOKKOS_INLINE_FUNCTION void con_tensor_to_embed(const GReal Xnative[GR_DIM], const GReal tcon_native[GR_DIM][GR_DIM], GReal tcon_embed[GR_DIM][GR_DIM]) const
            {cov_tensor_to_native(Xnative, tcon_native, tcon_embed);}
        KOKKOS_INLINE_FUNCTION void con_tensor_to_native(const GReal Xnative[GR_DIM], const GReal tcon_embed[GR_DIM][GR_DIM], GReal tcon_native[GR_DIM][GR_DIM]) const
            {cov_tensor_to_embed(Xnative, tcon_embed, tcon_native);}

        // And then some metric properties
        // TODO gcon_embed, gdet_embed
        KOKKOS_INLINE_FUNCTION void gcov_native(const GReal Xnative[GR_DIM], Real gcov[GR_DIM][GR_DIM]) const
        {
            Real gcov_em[GR_DIM][GR_DIM];
            GReal Xembed[GR_DIM];
            // Get coordinates in embedding system
            coord_to_embed(Xnative, Xembed);

            // Get metric in embedding coordinates
            gcov_embed(Xembed, gcov_em);

            // Transform to native coordinates
            cov_tensor_to_native(Xnative, gcov_em, gcov);
        }
        KOKKOS_INLINE_FUNCTION Real gcon_native(const GReal X[GR_DIM], Real gcon[GR_DIM][GR_DIM]) const
        {
            Real gcov[GR_DIM][GR_DIM];
            gcov_native(X, gcov);
            return gcon_native(gcov, gcon);
        }
        KOKKOS_INLINE_FUNCTION Real gcon_native(const Real gcov[GR_DIM][GR_DIM], Real gcon[GR_DIM][GR_DIM]) const
        {
            Real gdet = invert(&gcov[0][0], &gcon[0][0]);
            return sqrt(fabs(gdet));
        }

        KOKKOS_INLINE_FUNCTION void conn_native(const GReal X[GR_DIM], Real conn[GR_DIM][GR_DIM][GR_DIM]) const
        {
            Real tmp[GR_DIM][GR_DIM][GR_DIM];
            Real gcon[GR_DIM][GR_DIM];
            GReal Xh[GR_DIM], Xl[GR_DIM];
            Real gh[GR_DIM][GR_DIM];
            Real gl[GR_DIM][GR_DIM];

            for (int nu = 0; nu < GR_DIM; nu++) {
                DLOOP1 Xh[mu] = Xl[mu] = X[mu];
                Xh[nu] += DELTA;
                Xl[nu] -= DELTA;
                gcov_native(Xh, gh);
                gcov_native(Xl, gl);

                for (int lam = 0; lam < GR_DIM; lam++) {
                    for (int kap = 0; kap < GR_DIM; kap++) {
                        conn[lam][kap][nu] = (gh[lam][kap] - gl[lam][kap])/
                                                        (Xh[nu] - Xl[nu]);
                    }
                }
            }

            // Rearrange to find \Gamma_{lam nu mu}
            for (int lam = 0; lam < GR_DIM; lam++) {
                for (int nu = 0; nu < GR_DIM; nu++) {
                    for (int mu = 0; mu < GR_DIM; mu++) {
                        tmp[lam][nu][mu] = 0.5 * (conn[nu][lam][mu] +
                                                  conn[mu][lam][nu] -
                                                  conn[mu][nu][lam]);
                    }
                }
            }

            // Need gcon for raising index
            gcon_native(X, gcon);

            // Raise index to get \Gamma^lam_{nu mu}
            for (int lam = 0; lam < GR_DIM; lam++) {
                for (int nu = 0; nu < GR_DIM; nu++) {
                    for (int mu = 0; mu < GR_DIM; mu++) {
                        conn[lam][nu][mu] = 0.;

                        for (int kap = 0; kap < GR_DIM; kap++)
                            conn[lam][nu][mu] += gcon[lam][kap] * tmp[kap][nu][mu];
                    }
                }
            }
        }
};
