/**
 * GRCoordinates class
 */
#pragma once

#include "coordinates/uniform_cartesian.hpp"

#include "decs.hpp"

#include "coordinate_embedding.hpp"
#include "utils.hpp"

// TODO standardize namespaces
using namespace parthenon;
using namespace std;

// Option to ignore coordinates entirely,
// and only use flat-space SR in Cartesian coordinates
#define FAST_CARTESIAN 0
// Don't cache values of the metric, etc, just call into CoordinateEmbedding directly
#define NO_CACHE 0

/**
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

    // Simple test for spherical coordinates or not
    // Needed for better errors, and switching up floor definitions
    bool spherical;

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
    GRCoordinates(): UniformCartesian() {};
    GRCoordinates(const RegionSize &rs, ParameterInput *pin);
    GRCoordinates(const GRCoordinates &src, int coarsen);

    GRCoordinates(const GRCoordinates& src);
    GRCoordinates operator=(const GRCoordinates& src);

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

    // Transformations using the cached geometry
    KOKKOS_INLINE_FUNCTION void lower(const Real vcon[GR_DIM], Real vcov[GR_DIM],
                                        const int& k, const int& j, const int& i, const Loci loc) const;
    KOKKOS_INLINE_FUNCTION void raise(const Real vcov[GR_DIM], Real vcon[GR_DIM],
                                        const int& k, const int& j, const int& i, const Loci loc) const;

    // TODO Indexing functions and named slices to make it comfy
};

// Define the inline functions in header since they don't cross translation units

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