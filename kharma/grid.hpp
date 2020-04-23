/**
 * Grid class
 */
#pragma once

#include "mesh/mesh.hpp"

#include "decs.hpp"
#include "coordinate_embedding.hpp"

// TODO standardize namespaces
using namespace parthenon;
using namespace Kokkos;
using namespace std;

class Grid;
void init_grids(Grid& G);

// Option to ignore coordinates entirely,
// and basically just use flat-space SR
#define FAST_CARTESIAN 0
// Don't cache grids, just call into CoordinateEmbedding directly
#define NO_CACHE 0

/**
 * Mesh objects exist to answer the following questions in a structured-mesh code:
 * 1. Where is this grid zone? (i,j,k -> X and back)
 * 2. What are the local metric properties?
 * (3. What subset of zones should I run over to do X?)
 *
 * This version of Grid is primarily a wrapper for Parthenon's implementations of (1) and (3), adding:
 * * i,j,k,loc calling convention to (1)
 * * Cache or transparent index calls to CoordinateEmbedding for (2)
 * And eventually:
 * * Named slices for (3)
 */
class Grid
{
public:
    // This will be a pointer to Device memory
    // It will not be dereference-able in host code
    CoordinateEmbedding* coords;

    // Store a MeshBlock pointer for host-side operations
    MeshBlock *pmy_block;

    // But also store directly what we'll need for coord() which is device-side
    ParArrayND<Real> x1f, x2f, x3f;
    ParArrayND<Real> x1v, x2v, x3v;


#if FAST_CARTESIAN || NO_CACHE
#else
    GeomTensor2 gcon_direct, gcov_direct;
    GeomScalar gdet_direct;
    GeomTensor3 conn_direct;
#endif

    // Constructors
    Grid(MeshBlock* pmb);
    Grid(MeshBlock* pmb, CoordinateEmbedding* coordinates);


    // TODO I'm not sure if these can be faster
    // I should definitely add full-matrix versions though
#if FAST_CARTESIAN
    KOKKOS_INLINE_FUNCTION Real gcon(const Loci loc, const int i, const int j, const int mu, const int nu) const
        {return -2*(mu == 0 && nu == 0) + (mu == nu);}
    KOKKOS_INLINE_FUNCTION Real gcov(const Loci loc, const int i, const int j, const int mu, const int nu) const
        {return -2*(mu == 0 && nu == 0) + (mu == nu);}
    KOKKOS_INLINE_FUNCTION Real gdet(const Loci loc, const int i, const int j) const
        {return 1;}
    KOKKOS_INLINE_FUNCTION Real conn(const int i, const int j, const int mu, const int nu, const int lam) const
        {return 0;}

    KOKKOS_INLINE_FUNCTION void gcon(const Loci loc, const int i, const int j, Real gcon[NDIM][NDIM]) const
        {DLOOP2 gcon[mu][nu] = -2*(mu == 0 && nu == 0) + (mu == nu);}
    KOKKOS_INLINE_FUNCTION void gcov(const Loci loc, const int i, const int j, Real gcov[NDIM][NDIM]) const
        {DLOOP2 gcov[mu][nu] = -2*(mu == 0 && nu == 0) + (mu == nu);}
    KOKKOS_INLINE_FUNCTION void conn(const int i, const int j, Real conn[NDIM][NDIM][NDIM]) const
        {DLOOP3 conn[mu][nu][lam] = 0;}
#elif NO_CACHE
    // TODO Finish
    // These will be VERY SLOW.  Better to switch to the below and cache locally if we go this route
    KOKKOS_INLINE_FUNCTION Real gcon(const Loci loc, const int i, const int j, const int mu, const int nu) const
    {
        GReal X[NDIM], gcon[NDIM][NDIM];
        coord(i, j, 0, loc, X);
        coords->gcon_native(X, gcon);
        return gcon[mu][nu];
    }
    KOKKOS_INLINE_FUNCTION Real gcov(const Loci loc, const int i, const int j, const int mu, const int nu) const
    {
        GReal X[NDIM], gcov[NDIM][NDIM];
        coord(i, j, 0, loc, X);
        coords->gcov_native(X, gcov);
        return gcov[mu][nu];
    }
    KOKKOS_INLINE_FUNCTION Real gdet(const Loci loc, const int i, const int j) const
    {
        GReal X[NDIM], gcon[NDIM][NDIM];
        coord(i, j, 0, loc, X);
        return coords->gcon_native(X, gcon);
    }
    KOKKOS_INLINE_FUNCTION Real conn(const int i, const int j, const int mu, const int nu, const int lam) const
    {
        GReal X[NDIM], conn[NDIM][NDIM][NDIM];
        coord(i, j, 0, Loci::center, X);
        coords->conn_native(X, conn);
        return conn[mu][nu][lam];
    }

    KOKKOS_INLINE_FUNCTION void gcon(const Loci loc, const int i, const int j, Real gcon[NDIM][NDIM]) const
    {
        GReal X[NDIM];
        coord(i, j, 0, loc, X);
        coords->gcon_native(X, gcon);
    }
    KOKKOS_INLINE_FUNCTION void gcov(const Loci loc, const int i, const int j, Real gcov[NDIM][NDIM]) const
    {
        GReal X[NDIM];
        coord(i, j, 0, loc, X);
        coords->gcov_native(X, gcov);
    }
    KOKKOS_INLINE_FUNCTION void conn(const int i, const int j, Real conn[NDIM][NDIM][NDIM]) const
    {
        GReal X[NDIM];
        coord(i, j, 0, Loci::center, X);
        coords->conn_native(X, conn);
    }
#else
    KOKKOS_INLINE_FUNCTION Real gcon(const Loci loc, const int i, const int j, const int mu, const int nu) const
        {return gcon_direct(loc, i, j, mu, nu);}
    KOKKOS_INLINE_FUNCTION Real gcov(const Loci loc, const int i, const int j, const int mu, const int nu) const
        {return gcov_direct(loc, i, j, mu, nu);}
    KOKKOS_INLINE_FUNCTION Real gdet(const Loci loc, const int i, const int j) const
        {return gdet_direct(loc, i, j);}
    KOKKOS_INLINE_FUNCTION Real conn(const int i, const int j, const int mu, const int nu, const int lam) const
        {return conn_direct(i, j, mu, nu, lam);}

    KOKKOS_INLINE_FUNCTION void gcon(const Loci loc, const int i, const int j, Real gcon[NDIM][NDIM]) const
        {DLOOP2 gcon[mu][nu] = gcon_direct(loc, i, j, mu, nu);}
    KOKKOS_INLINE_FUNCTION void gcov(const Loci loc, const int i, const int j, Real gcov[NDIM][NDIM]) const
        {DLOOP2 gcov[mu][nu] = gcov_direct(loc, i, j, mu, nu);}
    KOKKOS_INLINE_FUNCTION void conn(const int i, const int j, Real conn[NDIM][NDIM][NDIM]) const
        {DLOOP3 conn[mu][nu][lam] = conn_direct(i, j, mu, nu, lam);}
#endif

    // Coordinates of the grid, i.e. "native"
    KOKKOS_INLINE_FUNCTION void coord(const int& i, const int& j, const int& k, const Loci& loc, GReal X[NDIM]) const;
    KOKKOS_INLINE_FUNCTION void coord_embed(const int& i, const int& j, const int& k, const Loci& loc, GReal Xembed[NDIM]) const
    {
        GReal Xnative[NDIM];
        coord(i, j, k, loc, Xnative);
        coords->coord_to_embed(Xnative, Xembed);
    }

    // Transformations using the cached geometry
    // TODO could dramatically expand usage with enough loop fission
    // Local versions on arrays
    KOKKOS_INLINE_FUNCTION void lower(const Real vcon[NDIM], Real vcov[NDIM],
                                        const int i, const int j, const int k, const Loci loc) const;
    KOKKOS_INLINE_FUNCTION void raise(const Real vcov[NDIM], Real vcon[NDIM],
                                        const int i, const int j, const int k, const Loci loc) const;

    // TODO Indexing i.e. flattening?  Seems like Parthenon will handle this
    // TODO Return common slices as whatever objects Parthenon wants
};

// Define the inline functions in header since they don't cross translation units

/**
 * Function to return native coordinates on the grid
 */
KOKKOS_INLINE_FUNCTION void Grid::coord(const int& i, const int& j, const int& k, const Loci& loc, Real X[NDIM]) const
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

KOKKOS_INLINE_FUNCTION void Grid::lower(const Real vcon[NDIM], Real vcov[NDIM],
                                        const int i, const int j, const int k, const Loci loc) const
{
    DLOOP1 vcov[mu] = 0;
    DLOOP2 vcov[mu] += gcov(loc, i, j, mu, nu) * vcon[nu];
}
KOKKOS_INLINE_FUNCTION void Grid::raise(const Real vcov[NDIM], Real vcon[NDIM],
                                        const int i, const int j, const int k, const Loci loc) const
{
    DLOOP1 vcon[mu] = 0;
    DLOOP2 vcon[mu] += gcon(loc, i, j, mu, nu) * vcov[nu];
}