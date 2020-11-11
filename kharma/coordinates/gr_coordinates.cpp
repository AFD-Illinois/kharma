/* 
 *  File: gr_coordinates.cpp
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

/*
 * Coordinate functions for GR 
 */

#include "gr_coordinates.hpp"

// This file doesn't have MeshBlock access, so it uses raw Kokkos calls
using namespace Kokkos;

#if FAST_CARTESIAN
/**
 * Fast Cartesian GRCoordinates just use the underlying UniformCartesian object for everything
 */
GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin) {}
GRCoordinates::GRCoordinates(const GRCoordinates &src, int coarsen): UniformCartesian(src, coarsen) {}
#else
// Internal function for initializing cache
void init_GRCoordinates(GRCoordinates& G, int n1, int n2, int n3);

/**
 * Construct a GRCoordinates object with a transformation according to preferences set in the package
 */
GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin)
{
    // This is effectively a constructor for the CoordinateEmbedding object,
    // but in KHARMA, that object is only used through this one.
    // And I want the option to use that code elsewhere as it's quite general & nice
    std::string base_str = pin->GetString("coordinates", "base"); // Require every problem to specify very basic geometry
    std::string transform_str = pin->GetString("coordinates", "transform");

    SomeBaseCoords base;
    if (base_str == "spherical_minkowski") {
        base.emplace<SphMinkowskiCoords>(SphMinkowskiCoords());
    } else if (base_str == "cartesian_minkowski" || base_str == "minkowski") {
        base.emplace<CartMinkowskiCoords>(CartMinkowskiCoords());
    } else if (base_str == "spherical_ks" || base_str == "ks") {
        GReal a = pin->GetReal("coordinates", "a");
        base.emplace<SphKSCoords>(SphKSCoords(a));
    } else if (base_str == "spherical_bl" || base_str == "bl") {
        GReal a = pin->GetReal("coordinates", "a");
        base.emplace<SphBLCoords>(SphBLCoords(a));
    } else {
        throw std::invalid_argument("Unsupported base coordinates!");
    }

    bool spherical = mpark::visit( [&](const auto& self) {
                return self.spherical;
            }, base);

    SomeTransform transform;
    if (transform_str == "null") {
        if (spherical) {
            transform.emplace<SphNullTransform>(SphNullTransform());
        } else {
            transform.emplace<CartNullTransform>(CartNullTransform());
        }
    } else if (transform_str == "modified" || transform_str == "mks") {
        if (!spherical) throw std::invalid_argument("Transform is for spherical coordinates!");
        GReal hslope = pin->GetOrAddReal("coordinates", "hslope", 0.3);
        transform.emplace<ModifyTransform>(ModifyTransform(hslope));
    } else if (transform_str == "funky" || transform_str == "fmks") {
        if (!spherical) throw std::invalid_argument("Transform is for spherical coordinates!");
        GReal hslope = pin->GetOrAddReal("coordinates", "hslope", 0.3);
        GReal startx1 = pin->GetReal("parthenon/mesh", "x1min");
        GReal mks_smooth = pin->GetOrAddReal("coordinates", "mks_smooth", 0.5);
        GReal poly_xt = pin->GetOrAddReal("coordinates", "poly_xt", 0.82);
        GReal poly_alpha = pin->GetOrAddReal("coordinates", "poly_alpha", 14.0);
        transform.emplace<FunkyTransform>(FunkyTransform(startx1, hslope, mks_smooth, poly_xt, poly_alpha));
    } else {
        throw std::invalid_argument("Unsupported coordinate transform!");
    }

    coords = CoordinateEmbedding(base, transform);

    n1 = rs.nx1 + 2*NGHOST;
    n2 = rs.nx2 > 1 ? rs.nx2 + 2*NGHOST : 1;
    n3 = rs.nx3 > 1 ? rs.nx3 + 2*NGHOST : 1;

    init_GRCoordinates(*this, n1, n2, n3);
}


GRCoordinates::GRCoordinates(const GRCoordinates &src, int coarsen): UniformCartesian(src, coarsen)
{
    //std::cerr << "Calling coarsen constructor" << std::endl;
    coords = src.coords;
    n1 = src.n1/coarsen;
    n2 = src.n2/coarsen;
    n3 = src.n3/coarsen;
    init_GRCoordinates(*this, n1, n2, n3);
}

/**
 * Initialize any cached geometry that GRCoordinates will need to return.
 *
 * This needs to be defined *outside* of the GRCoordinates object, because of some
 * fun issues with C++ Lambda capture, which Kokkos brings to the fore
 */
void init_GRCoordinates(GRCoordinates& G, int n1, int n2, int n3) {
    cerr << "Creating GRCoordinate cache size " << n1 << " " << n2 << endl;
    // Cache geometry.  May be faster than re-computing. May not be.
    G.gcon_direct = GeomTensor2("gcon", NLOC, n2, n1, GR_DIM, GR_DIM);
    G.gcov_direct = GeomTensor2("gcov", NLOC, n2, n1, GR_DIM, GR_DIM);
    G.gdet_direct = GeomScalar("gdet", NLOC, n2, n1);
    G.conn_direct = GeomTensor3("conn", n2, n1, GR_DIM, GR_DIM, GR_DIM);

    // Member variables have an implicit this->
    // C++ Lambdas (and therefore Kokkos Lambdas) capture pointers to objects, not full objects
    // Hence, you *CANNOT* use this->, or members, from inside kernels
    auto gcon_local = G.gcon_direct;
    auto gcov_local = G.gcov_direct;
    auto gdet_local = G.gdet_direct;
    auto conn_local = G.conn_direct;

    Kokkos::parallel_for("init_geom", MDRangePolicy<Rank<2>>({0,0}, {n2, n1}),
        KOKKOS_LAMBDA_2D {
            GReal X[GR_DIM];
            Real gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
            for (int loc=0; loc < NLOC; ++loc) {
                G.coord(0, j, i, (Loci)loc, X);
                G.coords.gcov_native(X, gcov_loc);
                gdet_local(loc, j, i) = G.coords.gcon_native(gcov_loc, gcon_loc);
                DLOOP2 {
                    gcov_local(loc, j, i, mu, nu) = gcov_loc[mu][nu];
                    gcon_local(loc, j, i, mu, nu) = gcon_loc[mu][nu];
                }
            }
        }
    );
    Kokkos::parallel_for("init_geom", MDRangePolicy<Rank<2>>({0,0}, {n2, n1}),
        KOKKOS_LAMBDA_2D {
            GReal X[GR_DIM];
            G.coord(0, j, i, Loci::center, X);
            Real conn_loc[GR_DIM][GR_DIM][GR_DIM];
            G.coords.conn_native(X, conn_loc);
            DLOOP3 conn_local(j, i, mu, nu, lam) = conn_loc[mu][nu][lam];
        }
    );

    FLAG("GRCoordinates metric init");
}
#endif // FAST_CARTESIAN