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

// This needs to be included only here -- it requires full-formed Parthenon
// types, which are not available when importing this file's header
#include "types.hpp"

using Kokkos::MDRangePolicy;
using Kokkos::Rank;

// Stepsize for numerical derivatives of the metric
#define DELTA 1.e-8

// Points to average (one side of a square, odd) when calculating the connections,
// and metric determinants on faces
#define CONN_AVG_POINTS 1
// Whether to make corrections to some metric quantities to match
// metric determinant derivatives
#define CONN_CORRECTIONS 0

#if FAST_CARTESIAN
/**
 * Fast Cartesian GRCoordinates objects just use the underlying UniformCartesian object for everything
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
    // TODO This is effectively a constructor for the CoordinateEmbedding object
    // We should move it there so we can handle system names, synonyms & categories in one place
    std::string base_str = pin->GetString("coordinates", "base"); // Require every problem to specify very basic geometry
    std::string transform_str = pin->GetString("coordinates", "transform"); // This is guessed in kharma.cpp

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
        transform.emplace<NullTransform>(NullTransform());
    } else if (transform_str == "exponential" || transform_str == "exp" || transform_str == "eks") {
        if (!spherical) throw std::invalid_argument("Transform is for spherical coordinates!");
        transform.emplace<ExponentialTransform>(ExponentialTransform());
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

    n1 = rs.nx1 + 2*Globals::nghost;
    n2 = rs.nx2 > 1 ? rs.nx2 + 2*Globals::nghost : 1;
    n3 = rs.nx3 > 1 ? rs.nx3 + 2*Globals::nghost : 1;
    //cout << "Initialized coordinates with nghost " << Globals::nghost << std::endl;

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
 * Initialize any cached geometry that GRCoordinates will need to return. While
 * GRCoordinates objects will be moved device-side, this can be run only on the
 * host.
 *
 * This needs to be defined *outside* of the GRCoordinates object, because of some
 * fun issues with C++ Lambda capture, which Kokkos brings to the fore
 */
void init_GRCoordinates(GRCoordinates& G, int n1, int n2, int n3) {
    //cerr << "Creating GRCoordinate cache size " << n1 << " " << n2 << std::endl;
    // Cache geometry.  May be faster than re-computing. May not be.
    G.gcon_direct = GeomTensor2("gcon", NLOC, n2+1, n1+1, GR_DIM, GR_DIM);
    G.gcov_direct = GeomTensor2("gcov", NLOC, n2+1, n1+1, GR_DIM, GR_DIM);
    G.gdet_direct = GeomScalar("gdet", NLOC, n2+1, n1+1);
    G.conn_direct = GeomTensor3("conn", n2, n1, GR_DIM, GR_DIM, GR_DIM);
    G.gdet_conn_direct = GeomTensor3("conn", n2, n1, GR_DIM, GR_DIM, GR_DIM);

    // Member variables have an implicit this->
    // C++ Lambdas (and therefore Kokkos Lambdas) capture pointers to objects, not full objects
    // Hence, you *CANNOT* use this->, or members, from inside kernels
    auto gcon_local = G.gcon_direct;
    auto gcov_local = G.gcov_direct;
    auto gdet_local = G.gdet_direct;
    auto conn_local = G.conn_direct;
    auto gdet_conn_local = G.gdet_conn_direct;

    Kokkos::parallel_for("init_geom", MDRangePolicy<Rank<2>>({0,0}, {n2+1, n1+1}),
        KOKKOS_LAMBDA (const int& j, const int& i) {
            // Iterate through locations. This could be done in fancy ways, but
            // this highlights what's actually going on.
            for (int iloc =0; iloc < NLOC; iloc++) {
                Loci loc = (Loci) iloc;
                // radius of points to sample, floor(npoints/2)
                const int radius = CONN_AVG_POINTS / 2;
                const int diameter = CONN_AVG_POINTS;
                const int square = CONN_AVG_POINTS*CONN_AVG_POINTS;
                if (loc == Loci::center || loc == Loci::face3) {
                    // This prevents overstepping conn's bounds by halting in the last zone
                    if (i >= n1 || j >= n2) continue;
                    // Get a square of points evenly across each cell,
                    // over both nontrivial geometry directions 1,2
                    // Note this never hits/passes the pole
                    for (int k=-radius; k <= radius; k++) {
                        for (int l=-radius; l <= radius; l++) {
                            GReal X[GR_DIM];
                            G.coord(0, j, i, loc, X);
                            GReal Xn1[GR_DIM], Xn2[GR_DIM];
                            G.coord(0, j, i+1, loc, Xn1);
                            G.coord(0, j+1, i, loc, Xn2);
                            X[1] += (Xn1[1] - X[1])/CONN_AVG_POINTS * k;
                            X[2] += (Xn2[2] - X[2])/CONN_AVG_POINTS * l;
                            // Get geometry at points
                            GReal gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
                            G.coords.gcov_native(X, gcov_loc);
                            const GReal gdet = G.coords.gcon_native(gcov_loc, gcon_loc);
                            // Add to running averages
                            gdet_local(loc, j, i) += gdet / square;
                            DLOOP2 {
                                gcov_local(loc, j, i, mu, nu) += gcov_loc[mu][nu] / square;
                                gcon_local(loc, j, i, mu, nu) += gcon_loc[mu][nu] / square;
                            }
                            if (loc == Loci::center) {
                                // In the center, get the connection and gdet*connection
                                Real conn_loc[GR_DIM][GR_DIM][GR_DIM];
                                G.coords.conn_native(X, DELTA, conn_loc);
                                DLOOP3 {
                                    conn_local(j, i, mu, nu, lam) += conn_loc[mu][nu][lam] / square;
                                    gdet_conn_local(j, i, mu, nu, lam) += gdet*conn_loc[mu][nu][lam] / square;
                                }
                            }
                        }
                    }
                } else if (loc == Loci::face1 || loc == Loci::face2) {
                    for (int k=-radius; k <= radius; k++) {
                        // Like the above, but only average over a particular face (line for 2D geometry)
                        GReal X[GR_DIM];
                        G.coord(0, j, i, loc, X);
                        GReal Xn1[GR_DIM];
                        // Step in the nontrivial direction perpendicular to the normal
                        const int avg_dir = (loc == Loci::face1) ? X2DIR : X1DIR;
                        // Get the direction/distance
                        G.coord(0, j + (avg_dir == X2DIR), i + (avg_dir == X1DIR), loc, Xn1);
                        X[avg_dir] += (Xn1[avg_dir] - X[avg_dir])/diameter * k;
                        // Get geometry at the point
                        GReal gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
                        G.coords.gcov_native(X, gcov_loc);
                        const GReal gdet = G.coords.gcon_native(gcov_loc, gcon_loc);
                        // Add to running averages
                        gdet_local(loc, j, i) += gdet / diameter;
                        DLOOP2 {
                            gcov_local(loc, j, i, mu, nu) += gcov_loc[mu][nu] / diameter;
                            gcon_local(loc, j, i, mu, nu) += gcon_loc[mu][nu] / diameter;
                        }
                    }
                } else {
                    // Just one point
                    GReal X[GR_DIM];
                    G.coord(0, j, i, loc, X);
                    // Get geometry
                    GReal gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
                    G.coords.gcov_native(X, gcov_loc);
                    const GReal gdet = G.coords.gcon_native(gcov_loc, gcon_loc);
                    // Set geometry
                    gdet_local(loc, j, i) = gdet;
                    DLOOP2 {
                        gcov_local(loc, j, i, mu, nu) = gcov_loc[mu][nu];
                        gcon_local(loc, j, i, mu, nu) = gcon_loc[mu][nu];
                    }
                }
            }
        }
    );
    if (CONN_CORRECTIONS) {
        Kokkos::parallel_for("geom_corrections", MDRangePolicy<Rank<2>>({0,0}, {n2, n1}),
            KOKKOS_LAMBDA (const int& j, const int& i) {
                // In the two directions the grid changes, make sure that we *exactly*
                // satisfy the req't gdet*conn^mu_mu_nu = d_nu gdet, when evaluated on faces
                // This will make the source term exactly balance the flux differences,
                // crucial near the poles
                GReal X[GR_DIM];
                G.coord(0, j, i, Loci::center, X);
                if (1) { //(m::abs(X[2] - 0) < 0.08 || m::abs(X[2] - 1.0) < 0.08)) {
                    for (int lam=1; lam < GR_DIM; lam++) {
                        const Loci loc = loc_of(lam);
                        // Get gdet values at faces we calculated above
                        GReal Xfm[GR_DIM], Xfp[GR_DIM];
                        G.coord(0, j, i, loc, Xfm);
                        G.coord(0, j + (lam == X2DIR), i + (lam == X1DIR), loc, Xfp);
                        double gdetfm = gdet_local(loc, j, i);
                        double gdetfp = gdet_local(loc, j + (lam == X2DIR), i + (lam == X1DIR));
                        GReal target = (gdetfp - gdetfm) / (Xfp[lam] - Xfm[lam] + SMALL);

                        // Then sum the coefficients and record nonzero ones for modification
                        GReal test_sum = 0;
                        GReal sum_portions, portions[GR_DIM] = {0};
                        DLOOP1 {
                            test_sum += gdet_conn_local(j, i, mu, mu, lam);
                            portions[mu] = m::abs(gdet_conn_local(j, i, mu, mu, lam));
                            sum_portions += portions[mu];
                        }
                        DLOOP1 portions[mu] /= sum_portions;
                        //printf("Zone %d %d target: %.3g test_sum: %.3g correction: %.3g\n", i, j, target, test_sum, diff);

                        // Add the difference among components equally
                        const GReal diff = test_sum - target;
                        DLOOP1 gdet_conn_local(j, i, mu, mu, lam) = gdet_conn_local(j, i, mu, mu, lam) - diff*portions[mu];

                        // This is separated and set equal, as there will be one self-assignment
                        DLOOP1 gdet_conn_local(j, i, mu, lam, mu) = gdet_conn_local(j, i, mu, mu, lam);
                    }
                }
            }
        );
    }

    Flag("GRCoordinates metric init");
}
#endif // FAST_CARTESIAN
