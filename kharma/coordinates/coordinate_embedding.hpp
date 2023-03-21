/* 
 *  File: coordinate_embedding.hpp
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

// std::variant requires C++ exceptions,
// so it will never be SYCL-ready.
// Instead we use mpark's reimplementation,
// patched to never throw exceptions.
// Because who needs those?
#include <mpark/variant.hpp>
//#include <variant>
//namespace mpark = std;

#include "decs.hpp"

#include "coordinate_systems.hpp"
#include "matrix.hpp"

/**
 * Coordinates in HARM are logically Cartesian -- that is, in some coordinate system, here dubbed "native"
 * coordinates, each cell is a rectangular prism of exactly the same shape as all the others.
 * However, working in GR allows us to define that "native" coordinate system arbitrarily
 * in relation to the "base" or "embedding" coordinates, which are usually Spherical Kerr-Schild coordinates.
 *
 * That is, as long as we have a bijective map of base<->transformed coordinates, we can define the latter
 * arbitrarily, which is great for putting resolution where we need and not where we don't.
 *
 * This class keeps track of the base coordinates and the map, defining generic functions guaranteed to return
 * something in the native or embedding coordinate system, given something else -- e.g. covariant metric "gcov"
 * at a native coordinate location Xnative.
 * 
 * Each system or transform must be a class defining a basic interface:
 * see coordinate_systems.hpp for the current examples.
 * 
 * Specifically, the BaseCoords class must implement:
 * * gcov_embed
 * And the Transform class must implement:
 * * coord_to_embed
 * * coord_to_native
 * * dxdX_to_embed
 * * dxdX_to_native
 * 
 * Each possible class is added to a couple of mpark::variant containers, and then to the chains of if statements below.
 *
 * TODO convenience functions.  Intelligent r/th/phi, x/y/z, KS and BL, a, rhor, etc by auto-translating contents
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
            }

            if (mpark::holds_alternative<NullTransform>(transform_in)) {
                transform.emplace<NullTransform>(mpark::get<NullTransform>(transform_in));
            } else if (mpark::holds_alternative<ExponentialTransform>(transform_in)) {
                transform.emplace<ExponentialTransform>(mpark::get<ExponentialTransform>(transform_in));
            } else if (mpark::holds_alternative<ModifyTransform>(transform_in)) {
                transform.emplace<ModifyTransform>(mpark::get<ModifyTransform>(transform_in));
            } else if (mpark::holds_alternative<FunkyTransform>(transform_in)) {
                transform.emplace<FunkyTransform>(mpark::get<FunkyTransform>(transform_in));
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

        // Convenience functions to get common things
        KOKKOS_INLINE_FUNCTION bool spherical() const
        {
            return mpark::visit( [&](const auto& self) {
                return self.spherical;
            }, base);
        }
        // KOKKOS_INLINE_FUNCTION GReal rhor() const
        // {
        //     return mpark::visit( [&](const auto& self) {
        //         self.rhor();
        //     }, base);
        // }
        KOKKOS_INLINE_FUNCTION GReal get_a() const
        {
            if (mpark::holds_alternative<SphKSCoords>(base)) {
                return mpark::get<SphKSCoords>(base).a;
            } else if (mpark::holds_alternative<SphBLCoords>(base)) {
                return mpark::get<SphBLCoords>(base).a;
            } else {
                return 0.0; //throw std::invalid_argument("BH Spin is not defined for selected coordinate system!");
            }
        }
        KOKKOS_INLINE_FUNCTION bool is_ks() const
        {
            if (mpark::holds_alternative<SphKSCoords>(base)) {
                return true;
            } else {
                return false;
            }
        }
        KOKKOS_INLINE_FUNCTION bool is_cart_minkowski() const
        {
            if (mpark::holds_alternative<CartMinkowskiCoords>(base) && mpark::holds_alternative<NullTransform>(transform)) {
                return true;
            } else {
                return false;
            }
        }

        // Spell out the interface we take from BaseCoords
        // TODO add a gcon_embed, gdet_embed
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

        // And then derived metric properties
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
            return m::sqrt(m::abs(gdet));
        }
        KOKKOS_INLINE_FUNCTION Real gdet_native(const GReal X[GR_DIM]) const
        {
            Real gcov[GR_DIM][GR_DIM], gcon[GR_DIM][GR_DIM];
            gcov_native(X, gcov);
            return gcon_native(gcov, gcon);
        }

        KOKKOS_INLINE_FUNCTION void conn_native(const GReal X[GR_DIM], const GReal delta, Real conn[GR_DIM][GR_DIM][GR_DIM]) const
        {
            GReal tmp[GR_DIM][GR_DIM][GR_DIM];
            GReal gcon[GR_DIM][GR_DIM];
            GReal Xh[GR_DIM], Xl[GR_DIM];
            GReal gh[GR_DIM][GR_DIM];
            GReal gl[GR_DIM][GR_DIM];

            for (int nu = 0; nu < GR_DIM; nu++) {
                DLOOP1 Xl[mu] = X[mu] - delta*(mu == nu);
                DLOOP1 Xh[mu] = X[mu] + delta*(mu == nu);
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
