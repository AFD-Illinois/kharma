/*
 * Class representing a logically Cartesian grid of points in a CoordinateEmbedding, including:
 * * Grid zone locations, start & end points
 * * Ghost or "halo" zones, iterators over grids with/without halos
 */
#pragma once

// Option to ignore coordinates entirely,
// and basically just use flat-space SR
#define FAST_CARTESIAN 0

#include "decs.hpp"
#include "coordinate_embedding.hpp"

#include <vector>
#include <memory>

// TODO standardize namespaces
using namespace Kokkos;
using namespace std;

// TODO list
// More individual grid operations (like raise/lower)?
// MPI/grid less than full support. Do Domain Decomp here?
// Passive variables
// Asymmetric in x3?

class Grid;
void init_grids(Grid& G);

/**
 * Class holding all parameters related to a logically Cartesian grid a.k.a. mesh.
 * Intended to be minimal. 400 lines later...
 */
class Grid
{
public:
    // TODO if we ditch AMR we can probably scrape up time by making these global/static to all grids
    // In light of all this state (~500b?)
    int n1, n2, n3, nvar;
    int ng, gn1, gn2, gn3;
    int n1tot, n2tot, n3tot;
    int n1start, n2start, n3start;
    GReal startx1, startx2, startx3;
    GReal dx1, dx2, dx3;

    GeomTensor gcon_direct, gcov_direct;
    GeomScalar gdet_direct;
    GeomConn conn_direct;

    // This will be a pointer to Device memory
    // It will not be dereference-able in host code
    CoordinateEmbedding* coords;

    // Constructors
    Grid(CoordinateEmbedding* coordinates, std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng_in=3, int nvar_in=8);
    Grid(CoordinateEmbedding* coordinates, std::vector<int> fullshape, std::vector<int> startn, std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng_in=3, int nvar_in=8);


    // TODO does abstracting to functions slow the memory-access versions?
#if FAST
    KOKKOS_INLINE_FUNCTION Real gcon(const Loci loc, const int i, const int j, const int mu, const int nu) const {return coords->gcon_native(X, );}
    KOKKOS_INLINE_FUNCTION Real gcov(const Loci loc, const int i, const int j, const int mu, const int nu) const {return -2*(mu == 0 && nu == 0) + (mu == nu);}
    KOKKOS_INLINE_FUNCTION Real gdet(const Loci loc, const int i, const int j) const {return 1;}
    KOKKOS_INLINE_FUNCTION Real conn(const int i, const int j, const int mu, const int nu, const int lam) const {return 0;}
#else
    KOKKOS_INLINE_FUNCTION Real gcon(const Loci loc, const int i, const int j, const int mu, const int nu) const {return gcon_direct(loc, i, j, mu, nu);}
    KOKKOS_INLINE_FUNCTION Real gcov(const Loci loc, const int i, const int j, const int mu, const int nu) const {return gcov_direct(loc, i, j, mu, nu);}
    KOKKOS_INLINE_FUNCTION Real gdet(const Loci loc, const int i, const int j) const {return gdet_direct(loc, i, j);}
    KOKKOS_INLINE_FUNCTION Real conn(const int i, const int j, const int mu, const int nu, const int lam) const {return conn_direct(i, j, mu, nu, lam);}
#endif

    // Coordinates of the grid, i.e. "native"
    KOKKOS_INLINE_FUNCTION void coord(const int i, const int j, const int k, const Loci loc, GReal X[NDIM], bool use_ghosts=true) const;
    // Convenience function.  TODO more of these passthroughs?  Fewer?
    KOKKOS_INLINE_FUNCTION void coord_embed(const int i, const int j, const int k, const Loci loc, GReal Xembed[NDIM]) const
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
    // Versions on an element of a GridVector
    KOKKOS_INLINE_FUNCTION void lower(const GridVector vcon, GridVector vcov,
                                        const int i, const int j, const int k, const Loci loc) const;
    KOKKOS_INLINE_FUNCTION void raise(const GridVector vcov, GridVector vcon,
                                        const int i, const int j, const int k, const Loci loc) const;

    // Indexing. TODO assumes forward indexing, auto-switch Right and Left
    KOKKOS_INLINE_FUNCTION int i4(const int i, const int j, const int k, const int p, const bool use_ghosts=true) const
    {
        if (use_ghosts) {
            return i*gn2*gn3*nvar + j*gn3*nvar + k*nvar + p;
        } else {
            return i*n2*n3*nvar + j*n3*nvar + k*nvar + p;
        }
    }
    KOKKOS_INLINE_FUNCTION int i3(const int i, const int j, const int k, const bool use_ghosts=true) const
    {
        if (use_ghosts) {
            return i*gn2*gn3 + j*gn3 + k;
        } else {
            return i*n2*n3 + j*n3 + k;
        }
    }
    // TODO MPI buffer index

    // RangePolicies over grid.  Construct them on the fly to stay slim
    // TODO can I consolidate these?  Types are fun
    MDRangePolicy<Rank<3>> bulk_0() const {return MDRangePolicy<Rank<3>>({0, 0, 0}, {n1, n2, n3});}
    MDRangePolicy<Rank<3>> bulk_ng() const {return MDRangePolicy<Rank<3>>({ng, ng, ng}, {n1+ng, n2+ng, n3+ng});}
    MDRangePolicy<Rank<3>> all_0() const {return MDRangePolicy<Rank<3>>({0, 0, 0}, {n1+2*ng, n2+2*ng, n3+2*ng});}
    MDRangePolicy<OpenMP,Rank<3>> h_bulk_0() const {return MDRangePolicy<OpenMP,Rank<3>>({0, 0, 0}, {n1, n2, n3});}
    MDRangePolicy<OpenMP,Rank<3>> h_bulk_ng() const {return MDRangePolicy<OpenMP,Rank<3>>({ng, ng, ng}, {n1+ng, n2+ng, n3+ng});}
    MDRangePolicy<OpenMP,Rank<3>> h_all_0() const {return MDRangePolicy<OpenMP,Rank<3>>({0, 0, 0}, {n1+2*ng, n2+2*ng, n3+2*ng});}

    MDRangePolicy<Rank<3>> bulk_plus(const int i) const {return MDRangePolicy<Rank<3>>({ng-i, ng-i, ng-i}, {n1+ng+i, n2+ng+i, n3+ng+i});}

    // Versions with 4th index for fluid variables
    MDRangePolicy<Rank<4>> bulk_0_p() const {return MDRangePolicy<Rank<4>>({0, 0, 0, 0}, {n1, n2, n3, nvar});}
    MDRangePolicy<Rank<4>> bulk_ng_p() const {return MDRangePolicy<Rank<4>>({ng, ng, ng, 0}, {n1+ng, n2+ng, n3+ng, nvar});}
    MDRangePolicy<Rank<4>> all_0_p() const {return MDRangePolicy<Rank<4>>({0, 0, 0, 0}, {n1+2*ng, n2+2*ng, n3+2*ng, nvar});}
    MDRangePolicy<OpenMP,Rank<4>> h_bulk_0_p() const {return MDRangePolicy<OpenMP,Rank<4>>({0, 0, 0, 0}, {n1, n2, n3, nvar});}
    MDRangePolicy<OpenMP,Rank<4>> h_bulk_ng_p() const {return MDRangePolicy<OpenMP,Rank<4>>({ng, ng, ng, 0}, {n1+ng, n2+ng, n3+ng, nvar});}
    MDRangePolicy<OpenMP,Rank<4>> h_all_0_p() const {return MDRangePolicy<OpenMP,Rank<4>>({0, 0, 0, 0}, {n1+2*ng, n2+2*ng, n3+2*ng, nvar});}

    MDRangePolicy<Rank<4>> bulk_plus_p(const int i) const {return MDRangePolicy<Rank<4>>({ng-i, ng-i, ng-i, 0}, {n1+ng+i, n2+ng+i, n3+ng+i, nvar});}

    // Boundaries
    // TODO reverse these for speed?
    MDRangePolicy<Rank<3>> bound_x1_l() const {return MDRangePolicy<Rank<3>>({0, ng, ng}, {ng, n2+ng, n3+ng});}
    MDRangePolicy<Rank<3>> bound_x1_r() const {return MDRangePolicy<Rank<3>>({n1+ng, ng, ng}, {n1+2*ng, n2+ng, n3+ng});}

    MDRangePolicy<Rank<3>> bound_x2_l() const {return MDRangePolicy<Rank<3>>({0, 0, ng}, {n1+2*ng, ng, n3+ng});}
    MDRangePolicy<Rank<3>> bound_x2_r() const {return MDRangePolicy<Rank<3>>({0, n2+ng, ng}, {n1+2*ng, n2+2*ng, n3+ng});}

    MDRangePolicy<Rank<3>> bound_x3_l() const {return MDRangePolicy<Rank<3>>({0, 0, 0}, {n1+2*ng, n2+2*ng, ng});}
    MDRangePolicy<Rank<3>> bound_x3_r() const {return MDRangePolicy<Rank<3>>({0, 0, n3+ng}, {n1+2*ng, n2+2*ng, n3+2*ng});}

    // TODO MPI boundary stuff: packing and unpacking


    // TODO Versions of above in 2D for geometry stuff?

};

/**
 * Construct a grid which starts at 0 and covers the entire global space
 */
Grid::Grid(CoordinateEmbedding* coordinates, std::vector<int> shape, std::vector<GReal> startx,
            std::vector<GReal> endx, const int ng_in, const int nvar_in)
{
    nvar = nvar_in;
    ng = ng_in;

    n1tot = n1 = shape[0];
    n2tot = n2 = shape[1];
    n3tot = n3 = shape[2];

    n1start = 0;
    n2start = 0;
    n3start = 0;

    gn1 = n1 + 2*ng;
    gn2 = n2 + 2*ng;
    gn3 = n3 + 2*ng;

    startx1 = startx[0];
    startx2 = startx[1];
    startx3 = startx[2];

    dx1 = (endx[0] - startx1) / n1;
    dx2 = (endx[1] - startx2) / n2;
    dx3 = (endx[2] - startx3) / n3;

    coords = coordinates;
    init_grids(*this);
}

/**
 * Construct a sub-grid starting at some point in a global space
 */
Grid::Grid(CoordinateEmbedding* coordinates, std::vector<int> fullshape, std::vector<int> startn,
            std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx,
            const int ng_in, const int nvar_in)
{
    nvar = nvar_in;
    ng = ng_in;

    n1tot = fullshape[0];
    n2tot = fullshape[1];
    n3tot = fullshape[2];

    n1start = startn[0];
    n2start = startn[1];
    n3start = startn[2];

    n1 = shape[0];
    n2 = shape[1];
    n3 = shape[2];

    gn1 = n1 + 2*ng;
    gn2 = n2 + 2*ng;
    gn3 = n3 + 2*ng;

    startx1 = startx[0];
    startx2 = startx[1];
    startx3 = startx[2];

    dx1 = (endx[0] - startx1) / n1;
    dx2 = (endx[1] - startx2) / n2;
    dx3 = (endx[2] - startx3) / n3;

    coords = coordinates;
    init_grids(*this);
}

/**
 * Function to return native coordinates of a grid
 * TODO is it more instruction-efficient to split this per location or have a separate one for centers?
 */
KOKKOS_INLINE_FUNCTION void Grid::coord(const int i, const int j, const int k, Loci loc, GReal *X, bool use_ghosts) const
{
    X[0] = 0;
    switch (loc)
    {
    case face1:
        X[1] = startx1 + (n1start + i - (use_ghosts)*ng) * dx1;
        X[2] = startx2 + (n2start + j + 0.5 - (use_ghosts)*ng) * dx2;
        X[3] = startx3 + (n3start + k + 0.5 - (use_ghosts)*ng) * dx3;
        break;
    case face2:
        X[1] = startx1 + (n1start + i + 0.5 - (use_ghosts)*ng) * dx1;
        X[2] = startx2 + (n2start + j - (use_ghosts)*ng) * dx2;
        X[3] = startx3 + (k + 0.5 - (use_ghosts)*ng) * dx3;
        break;
    case face3:
        X[1] = startx1 + (n1start + i + 0.5 - (use_ghosts)*ng) * dx1;
        X[2] = startx2 + (n2start + j + 0.5 - (use_ghosts)*ng) * dx2;
        X[3] = startx3 + (n3start + k - (use_ghosts)*ng) * dx3;
        break;
    case center:
        X[1] = startx1 + (n1start + i + 0.5 - (use_ghosts)*ng) * dx1;
        X[2] = startx2 + (n2start + j + 0.5 - (use_ghosts)*ng) * dx2;
        X[3] = startx3 + (n3start + k + 0.5 - (use_ghosts)*ng) * dx3;
        break;
    case corner:
        X[1] = startx1 + (n1start + i - (use_ghosts)*ng) * dx1;
        X[2] = startx2 + (n2start + j - (use_ghosts)*ng) * dx2;
        X[3] = startx3 + (n3start + k - (use_ghosts)*ng) * dx3;
        break;
    }
}

#if FAST_CARTESIAN
KOKKOS_INLINE_FUNCTION void Grid::lower(const Real vcon[NDIM], Real vcov[NDIM],
                                        const int i, const int j, const int k, const Loci loc) const
{ DLOOP1 vcov[mu] = (1 - 2*(mu == 0)) * vcon[mu]; }
KOKKOS_INLINE_FUNCTION void Grid::raise(const Real vcov[NDIM], Real vcon[NDIM],
                                        const int i, const int j, const int k, const Loci loc) const
{ DLOOP1 vcon[mu] =(1 - 2*(mu == 0)) * vcov[mu]; }
KOKKOS_INLINE_FUNCTION void Grid::lower(const GridVector vcon, GridVector vcov,
                                        const int i, const int j, const int k, const Loci loc) const
{ DLOOP1 vcov(i, j, k, mu) = (1 - 2*(mu == 0)) * vcon(i, j, k, mu); }
KOKKOS_INLINE_FUNCTION void Grid::raise(const GridVector vcov, GridVector vcon,
                                        const int i, const int j, const int k, const Loci loc) const
{ DLOOP1 vcon(i, j, k, mu) = (1 - 2*(mu == 0)) * vcov(i, j, k, mu); }

KOKKOS_INLINE_FUNCTION void lower(const Real vcon[NDIM], const GeomTensor gcov, Real vcov[NDIM],
                                        const int i, const int j, const int k, const Loci loc)
{ DLOOP1 vcov[mu] = (1 - 2*(mu == 0)) * vcon[mu]; }
KOKKOS_INLINE_FUNCTION void lower(const GridVector vcon, const GeomTensor gcov, GridVector vcov,
                                        const int i, const int j, const int k, const Loci loc)
{ DLOOP1 vcov(i, j, k, mu) = (1 - 2*(mu == 0)) * vcon(i, j, k, mu); }
#else
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
KOKKOS_INLINE_FUNCTION void Grid::lower(const GridVector vcon, GridVector vcov,
                                        const int i, const int j, const int k, const Loci loc) const
{
    DLOOP1 vcov(i, j, k, mu) = 0;
    DLOOP2 vcov(i, j, k, mu) += gcov(loc, i, j, mu, nu) * vcon(i, j, k, nu);
}
KOKKOS_INLINE_FUNCTION void Grid::raise(const GridVector vcov, GridVector vcon,
                                        const int i, const int j, const int k, const Loci loc) const
{
    DLOOP1 vcon(i, j, k, mu) = 0;
    DLOOP2 vcon(i, j, k, mu) += gcon(loc, i, j, mu, nu) * vcov(i, j, k, nu);
}

// Versions not reliant on a grid object. TODO extend or extinguish
KOKKOS_INLINE_FUNCTION void lower(const Real vcon[NDIM], const GeomTensor gcov, Real vcov[NDIM],
                                        const int i, const int j, const int k, const Loci loc)
{
    DLOOP1 vcov[mu] = 0;
    DLOOP2 vcov[mu] += gcov(loc, i, j, mu, nu) * vcon[nu];
}
KOKKOS_INLINE_FUNCTION void lower(const GridVector vcon, const GeomTensor gcov, GridVector vcov,
                                        const int i, const int j, const int k, const Loci loc)
{
    DLOOP1 vcov(i, j, k, mu) = 0;
    DLOOP2 vcov(i, j, k, mu) += gcov(loc, i, j, mu, nu) * vcon(i, j, k, nu);
}
#endif

// TODO move this back into Grid by capturing only what it needs
// Currently it is separate because of a neat bug:
// Capturing this->, even if not used, messes up calling a virtual function
#if FAST_CARTESIAN
void init_grids(Grid& G) {}
#else
void init_grids(Grid& G) {
    // Cache geometry.  Probably faster in most cases than re-computing due to amortization of reads
    G.gcon_direct = GeomTensor("gcon", NLOC, G.gn1, G.gn2);
    G.gcov_direct = GeomTensor("gcov", NLOC, G.gn1, G.gn2);
    G.gdet_direct = GeomScalar("gdet", NLOC, G.gn1, G.gn2);
    G.conn_direct = GeomConn("conn", G.gn1, G.gn2);

    // Member variables have an implicit this->
    // Kokkos captures pointers to objects, not full objects
    // Hence, you *CANNOT* use this->, or members, from inside kernels
    auto gcon_local = G.gcon_direct;
    auto gcov_local = G.gcov_direct;
    auto gdet_local = G.gdet_direct;
    auto conn_local = G.conn_direct;
    CoordinateEmbedding cs = *(G.coords);

    Kokkos::parallel_for("init_geom", MDRangePolicy<Rank<2>>({0, 0}, {G.gn1, G.gn2}),
        KOKKOS_LAMBDA (const int& i, const int& j) {
            GReal X[NDIM];
            Real gcov_loc[NDIM][NDIM], gcon_loc[NDIM][NDIM];
            for (int loc=0; loc < NLOC; ++loc) {
                G.coord(i, j, 0, (Loci)loc, X);
                cs.gcov_native(X, gcov_loc);
                gdet_local(loc, i, j) = cs.gcon_native(gcov_loc, gcon_loc);
                DLOOP2 {
                    gcov_local(loc, i, j, mu, nu) = gcov_loc[mu][nu];
                    gcon_local(loc, i, j, mu, nu) = gcon_loc[mu][nu];
                }
            }
        }
    );
    Kokkos::parallel_for("init_conn", MDRangePolicy<Rank<2>>({0, 0}, {G.gn1, G.gn2}),
        KOKKOS_LAMBDA (const int& i, const int& j) {
            GReal X[NDIM];
            G.coord(i, j, 0, Loci::center, X);
            Real conn_loc[NDIM][NDIM][NDIM];
            cs.conn_func(X, conn_loc);
            DLOOP2 for(int kap=0; kap<NDIM; ++kap)
                conn_local(i, j, mu, nu, kap) = conn_loc[mu][nu][kap];
        }
    );

    FLAG("Grid metric init");
}
#endif