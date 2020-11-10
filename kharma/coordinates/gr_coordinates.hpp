/* 
 *  File: gr_coordinates.hpp
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

#pragma hd_warning_disable

#include "decs.hpp"

//#include "debug.hpp"
#include "coordinate_embedding.hpp"
#include "utils.hpp"

// Everywhere else we can just import <parthenon/parthenon.hpp>
// Here we have to be careful of circular dependencies
#include "coordinates/uniform_cartesian.hpp"
#include "parameter_input.hpp"
//#include "mesh/domain.hpp"
//#include "mesh/mesh.hpp"

#include "Kokkos_Core.hpp"

// TODO standardize namespaces, maybe fewer?
// Also parthenon may not always like this namespace import...
using namespace parthenon;
using namespace std;
using namespace Kokkos;

// Option to ignore coordinates entirely,
// and only use flat-space SR in Cartesian coordinates
#define FAST_CARTESIAN 0
// Don't cache values of the metric, etc, just call into CoordinateEmbedding directly
#define NO_CACHE 0

/**
 * Replacement/extension coordinate class for Parthenon
 * 
 * Parthenon's UniformCartesian coordinate class keeps, for each meshblock,
 * cell locations on a global X,Y,Z grid.
 * We wish to map these "logically-Cartesian" coordinates to volumes of interest,
 * with differential geometry and lots of caching
 */
class GRCoordinates : public UniformCartesian
{
public:
    // Host-side coordinates object pointer
    // Note we keep the actual object in GRCoordinates.  This is a royal pain to implement,
    // but ensures it will get copied device-side by C++14 Lambdas, circumventing *so many* bugs
    CoordinateEmbedding coords;

    // UniformCartesian doesn't need domain size.  We do.
    int n1, n2, n3;
    // And optionally some caches
#if !FAST_CARTESIAN && !NO_CACHE
    GeomTensor2 gcon_direct, gcov_direct;
    GeomScalar gdet_direct;
    GeomTensor3 conn_direct;
#endif

    // Constructors.  Must implement all parent constructors as Parthenon uses them all
    // ... and some operator=
    KOKKOS_INLINE_FUNCTION GRCoordinates(): UniformCartesian() {};
    KOKKOS_INLINE_FUNCTION GRCoordinates(const RegionSize &rs, ParameterInput *pin);
    KOKKOS_INLINE_FUNCTION GRCoordinates(const GRCoordinates &src, int coarsen);

    KOKKOS_INLINE_FUNCTION GRCoordinates(const GRCoordinates& src);
    KOKKOS_INLINE_FUNCTION GRCoordinates operator=(const GRCoordinates& src);

    // TODO Test these vs going all-in on full-matrix versions and computing on the fly
    KOKKOS_INLINE_FUNCTION Real gcon(const Loci loc, const int& j, const int& i, const int mu, const int nu) const;
    KOKKOS_INLINE_FUNCTION Real gcov(const Loci loc, const int& j, const int& i, const int mu, const int nu) const;
    KOKKOS_INLINE_FUNCTION Real gdet(const Loci loc, const int& j, const int& i) const;
    KOKKOS_INLINE_FUNCTION Real conn(const int& j, const int& i, const int mu, const int nu, const int lam) const;

    KOKKOS_INLINE_FUNCTION void gcon(const Loci loc, const int& j, const int& i, Real gcon[GR_DIM][GR_DIM]) const;
    KOKKOS_INLINE_FUNCTION void gcov(const Loci loc, const int& j, const int& i, Real gcov[GR_DIM][GR_DIM]) const;
    KOKKOS_INLINE_FUNCTION void conn(const int& j, const int& i, Real conn[GR_DIM][GR_DIM][GR_DIM]) const;

    // Coordinates of the GRCoordinates, i.e. "native"
    KOKKOS_INLINE_FUNCTION void coord(const int& k, const int& j, const int& i, const Loci& loc, GReal X[GR_DIM]) const;
    // Coordinates of the embedding system, usually r,th,phi[KS] or x1,x2,x3
    KOKKOS_INLINE_FUNCTION void coord_embed(const int& k, const int& j, const int& i, const Loci& loc, GReal Xembed[GR_DIM]) const;
    // Coordinates in a specific 

    // Transformations using the cached geometry
    KOKKOS_INLINE_FUNCTION void lower(const Real vcon[GR_DIM], Real vcov[GR_DIM],
                                        const int& k, const int& j, const int& i, const Loci loc) const;
    KOKKOS_INLINE_FUNCTION void raise(const Real vcov[GR_DIM], Real vcon[GR_DIM],
                                        const int& k, const int& j, const int& i, const Loci loc) const;

    // TODO Indexing functions and named slices to make it comfy
};

// Declare internal function for initializing cache
KOKKOS_INLINE_FUNCTION void init_GRCoordinates(GRCoordinates& G, int n1, int n2, int n3);

// Define the inline functions in header since they don't cross translation units

#if FAST_CARTESIAN
/**
 * Fast Cartesian GRCoordinates just use the underlying UniformCartesian object for everything
 */
KOKKOS_INLINE_FUNCTION GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin) {}
#else
/**
 * Construct a GRCoordinates object with a transformation according to preferences set in the package
 */
KOKKOS_INLINE_FUNCTION GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin)
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
        printf("Unsupported base coordinates!");
        //throw std::invalid_argument("Unsupported base coordinates!");
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
        if (!spherical) {
            printf("Transform is for spherical coordinates!");
            //throw std::invalid_argument("Transform is for spherical coordinates!");
        }
        GReal hslope = pin->GetOrAddReal("coordinates", "hslope", 0.3);
        transform.emplace<ModifyTransform>(ModifyTransform(hslope));
    } else if (transform_str == "funky" || transform_str == "fmks") {
        if (!spherical) {
            printf("Transform is for spherical coordinates!");
            //throw std::invalid_argument("Transform is for spherical coordinates!");
        }
        GReal hslope = pin->GetOrAddReal("coordinates", "hslope", 0.3);
        GReal startx1 = pin->GetReal("parthenon/mesh", "x1min");
        GReal mks_smooth = pin->GetOrAddReal("coordinates", "mks_smooth", 0.5);
        GReal poly_xt = pin->GetOrAddReal("coordinates", "poly_xt", 0.82);
        GReal poly_alpha = pin->GetOrAddReal("coordinates", "poly_alpha", 14.0);
        transform.emplace<FunkyTransform>(FunkyTransform(startx1, hslope, mks_smooth, poly_xt, poly_alpha));
    } else {
        printf("Unsupported coordinate transform!");
        //throw std::invalid_argument("Unsupported coordinate transform!");
    }

    coords = CoordinateEmbedding(base, transform);

    n1 = rs.nx1 + 2*NGHOST;
    n2 = rs.nx2 > 1 ? rs.nx2 + 2*NGHOST : 1;
    n3 = rs.nx3 > 1 ? rs.nx3 + 2*NGHOST : 1;

    init_GRCoordinates(*this, n1, n2, n3);
}
#endif

// OTHER CONSTRUCTORS: Same between implementations

KOKKOS_INLINE_FUNCTION GRCoordinates::GRCoordinates(const GRCoordinates &src, int coarsen): UniformCartesian(src, coarsen)
{
    //std::cerr << "Calling coarsen constructor" << std::endl;
    coords = src.coords;
    n1 = src.n1/coarsen;
    n2 = src.n2/coarsen;
    n3 = src.n3/coarsen;
    init_GRCoordinates(*this, n1, n2, n3);
}

KOKKOS_INLINE_FUNCTION GRCoordinates::GRCoordinates(const GRCoordinates &src): UniformCartesian(src)
{
    //std::cerr << "Calling copy constructor size " << src.n1 << " " << src.n2 << std::endl;
    coords = src.coords;
    n1 = src.n1;
    n2 = src.n2;
    n3 = src.n3;
#if !FAST_CARTESIAN && !NO_CACHE
    gcon_direct = src.gcon_direct;
    gcov_direct = src.gcov_direct;
    gdet_direct = src.gdet_direct;
    conn_direct = src.conn_direct;
#endif
}

KOKKOS_INLINE_FUNCTION GRCoordinates GRCoordinates::operator=(const GRCoordinates& src)
{
    //std::cerr << "Calling assignment operator size " << src.n1 << " " << src.n2 << std::endl;
    UniformCartesian::operator=(src);
    coords = src.coords;
    n1 = src.n1;
    n2 = src.n2;
    n3 = src.n3;
#if !FAST_CARTESIAN && !NO_CACHE
    gcon_direct = src.gcon_direct;
    gcov_direct = src.gcov_direct;
    gdet_direct = src.gdet_direct;
    conn_direct = src.conn_direct;
#endif
    return *this;
}

/**
 * Initialize any cached geometry that GRCoordinates will need to return.
 *
 * This needs to be defined *outside* of the GRCoordinates object, because of some
 * fun issues with C++ Lambda capture, which Kokkos brings to the fore
 */
#if FAST_CARTESIAN || NO_CACHE
KOKKOS_INLINE_FUNCTION void init_GRCoordinates(GRCoordinates& G, int n1, int n2, int n3) {}
#else
KOKKOS_INLINE_FUNCTION void init_GRCoordinates(GRCoordinates& G, int n1, int n2, int n3) {
    printf("Creating GRCoordinate cache: j,i size %d,%d\n", n2, n1);
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
#endif


/**
 * Function to return native coordinates on the GRCoordinates
 */
KOKKOS_INLINE_FUNCTION void GRCoordinates::coord(const int& k, const int& j, const int& i, const Loci& loc, Real X[GR_DIM]) const
{
    X[0] = 0;
    switch(loc)
    {
    case Loci::face1:
        X[1] = x1f(i);
        X[2] = x2v(j);
        X[3] = x3v(k);
        break;
    case Loci::face2:
        X[1] = x1v(i);
        X[2] = x2f(j);
        X[3] = x3v(k);
        break;
    case Loci::face3:
        X[1] = x1v(i);
        X[2] = x2v(j);
        X[3] = x3f(k);
        break;
    case Loci::center:
        X[1] = x1v(i);
        X[2] = x2v(j);
        X[3] = x3v(k);
        break;
    case Loci::corner:
        X[1] = x1f(i);
        X[2] = x2f(j);
        X[3] = x3f(k);
        break;
    }
}

#if FAST_CARTESIAN
KOKKOS_INLINE_FUNCTION void GRCoordinates::coord_embed(const int& k, const int& j, const int& i, const Loci& loc, GReal Xembed[GR_DIM]) const
{
    // Only supports null transform
    coord(k, j, i, loc, Xembed);
}
#else
/**
 * TODO Currently CANNOT be called from device-side code
 */
KOKKOS_INLINE_FUNCTION void GRCoordinates::coord_embed(const int& k, const int& j, const int& i, const Loci& loc, GReal Xembed[GR_DIM]) const
{
    GReal Xnative[GR_DIM];
    coord(k, j, i, loc, Xnative);
    coords.coord_to_embed(Xnative, Xembed);
}
#endif

KOKKOS_INLINE_FUNCTION void GRCoordinates::lower(const Real vcon[GR_DIM], Real vcov[GR_DIM],
                                        const int& k, const int& j, const int& i, const Loci loc) const
{
    DLOOP1 vcov[mu] = 0;
    DLOOP2 vcov[mu] += gcov(loc, j, i, mu, nu) * vcon[nu];
}
KOKKOS_INLINE_FUNCTION void GRCoordinates::raise(const Real vcov[GR_DIM], Real vcon[GR_DIM],
                                        const int& k, const int& j, const int& i, const Loci loc) const
{
    DLOOP1 vcon[mu] = 0;
    DLOOP2 vcon[mu] += gcon(loc, j, i, mu, nu) * vcov[nu];
}

// Three different implementations of the metric functions:
// FAST_CARTESIAN: Minkowski space constant values
// NO_CACHE: Re-calculate from coordinates object on every access
// NORMAL: Cache each zone center and return cached value thereafter
#if FAST_CARTESIAN
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcon(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
    {return -2*(mu == 0 && nu == 0) + (mu == nu);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcov(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
    {return -2*(mu == 0 && nu == 0) + (mu == nu);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gdet(const Loci loc, const int& j, const int& i) const
    {return 1;}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::conn(const int& j, const int& i, const int mu, const int nu, const int lam) const
    {return 0;}

KOKKOS_INLINE_FUNCTION void GRCoordinates::gcon(const Loci loc, const int& j, const int& i, Real gcon[GR_DIM][GR_DIM]) const
    {DLOOP2 gcon[mu][nu] = -2*(mu == 0 && nu == 0) + (mu == nu);}
KOKKOS_INLINE_FUNCTION void GRCoordinates::gcov(const Loci loc, const int& j, const int& i, Real gcov[GR_DIM][GR_DIM]) const
    {DLOOP2 gcov[mu][nu] = -2*(mu == 0 && nu == 0) + (mu == nu);}
KOKKOS_INLINE_FUNCTION void GRCoordinates::conn(const int& j, const int& i, Real conn[GR_DIM][GR_DIM][GR_DIM]) const
    {DLOOP3 conn[mu][nu][lam] = 0;}
#elif NO_CACHE
// TODO these are currently VERY SLOW.  Rework them to generate just the desired component. (TODO gdet?...)
// Except conn.  We never need conn fast.
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcon(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
{
    GReal X[GR_DIM], gcon[GR_DIM][GR_DIM];
    coord(0, j, i, loc, X);
    coords.gcon_native(X, gcon);
    return gcon[mu][nu];
}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcov(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
{
    GReal X[GR_DIM], gcov[GR_DIM][GR_DIM];
    coord(0, j, i, loc, X);
    coords.gcov_native(X, gcov);
    return gcov[mu][nu];
}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gdet(const Loci loc, const int& j, const int& i) const
{
    GReal X[GR_DIM], gcon[GR_DIM][GR_DIM];
    coord(0, j, i, loc, X);
    return coords.gcon_native(X, gcon);
}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::conn(const int& j, const int& i, const int mu, const int nu, const int lam) const
{
    GReal X[GR_DIM], conn[GR_DIM][GR_DIM][GR_DIM];
    coord(0, j, i, Loci::center, X);
    coords.conn_native(X, conn);
    return conn[mu][nu][lam];
}

KOKKOS_INLINE_FUNCTION void GRCoordinates::gcon(const Loci loc, const int& j, const int& i, Real gcon[GR_DIM][GR_DIM]) const
{
    GReal X[GR_DIM];
    coord(0, j, i, loc, X);
    coords.gcon_native(X, gcon);
}
KOKKOS_INLINE_FUNCTION void GRCoordinates::gcov(const Loci loc, const int& j, const int& i, Real gcov[GR_DIM][GR_DIM]) const
{
    GReal X[GR_DIM];
    coord(0, j, i, loc, X);
    coords.gcov_native(X, gcov);
}
KOKKOS_INLINE_FUNCTION void GRCoordinates::conn(const int& j, const int& i, Real conn[GR_DIM][GR_DIM][GR_DIM]) const
{
    GReal X[GR_DIM];
    coord(0, j, i, Loci::center, X);
    coords.conn_native(X, conn);
}
#else
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcon(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
    {return gcon_direct(loc, j, i, mu, nu);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gcov(const Loci loc, const int& j, const int& i, const int mu, const int nu) const
    {return gcov_direct(loc, j, i, mu, nu);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::gdet(const Loci loc, const int& j, const int& i) const
    {return gdet_direct(loc, j, i);}
KOKKOS_INLINE_FUNCTION Real GRCoordinates::conn(const int& j, const int& i, const int mu, const int nu, const int lam) const
    {return conn_direct(j, i, mu, nu, lam);}

KOKKOS_INLINE_FUNCTION void GRCoordinates::gcon(const Loci loc, const int& j, const int& i, Real gcon[GR_DIM][GR_DIM]) const
    {DLOOP2 gcon[mu][nu] = gcon_direct(loc, j, i, mu, nu);}
KOKKOS_INLINE_FUNCTION void GRCoordinates::gcov(const Loci loc, const int& j, const int& i, Real gcov[GR_DIM][GR_DIM]) const
    {DLOOP2 gcov[mu][nu] = gcov_direct(loc, j, i, mu, nu);}
KOKKOS_INLINE_FUNCTION void GRCoordinates::conn(const int& j, const int& i, Real conn[GR_DIM][GR_DIM][GR_DIM]) const
    {DLOOP3 conn[mu][nu][lam] = conn_direct(j, i, mu, nu, lam);}
#endif