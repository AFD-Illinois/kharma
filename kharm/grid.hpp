/*
 * Class representing a logically Cartesian grid of points in a CoordinateSystem, including:
 * * Grid zone locations, start & end points
 * * Ghost or "halo" zones, iterators over grids with/without halos
 */
#pragma once

#include "decs.hpp"
#include "coordinates.hpp"

#include <vector>

using namespace Kokkos;

// TODO list
// Carrying geometry on grid loci
// Grid/geom/linalg operations? (considering spatially-symmetric geom?)
// MPI/grid less than full support
// Passive variables


/**
 * Class representing a logically Cartesian grid of points embedded in a CoordinateSystem, including:
 * * Grid zone locations, start & end points
 * * Ghost or "halo" zones, iterators over grids with/without halos
 * 
 * Grids have a notion of "native" or Cartesian coordinates, and "embedding" or physical coordinates.
 * Most operations in HARM are done in native coordinates, thus most grid functions (e.g. metric, etc)
 * return values in native coordinates. 
 */
class Grid
{
public:
    int n1, n2, n3, nvar;
    int ng, gn1, gn2, gn3;
    int n1tot, n2tot, n3tot;
    int n1start, n2start, n3start;
    GReal startx1, startx2, startx3;
    GReal dx1, dx2, dx3;

    CoordinateSystem coords;

    // Kokkos policies for iterating over this grid
    MDRangePolicy<Rank<3>> *bulk_0, *bulk_ng, *all_0;    
    MDRangePolicy<OpenMP, Rank<3>> *h_bulk_0, *h_bulk_ng, *h_all_0;


    Grid(CoordinateSystem coords_in, std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng_in=3, int nvar_in=8);

#if USE_MPI
    Grid(std::vector<int> fullshape, std::vector<int> starti, std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng_in=3, int nvar_in=8);
#endif

    template<typename T>
    KOKKOS_INLINE_FUNCTION void coord(int i, int j, int k, Loci loc, T X) const;

};

/**
 * Non-MPI, non-AMR grid constructor. Lots of defaults.
 * TODO more defaults?
 */
Grid::Grid(CoordinateSystem coords_in, std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng_in, int nvar_in)
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

    bulk_0 = new MDRangePolicy<Rank<3>>({0, 0, 0}, {n1, n2, n3});
    h_bulk_0 = new MDRangePolicy<OpenMP, Rank<3>>({0, 0, 0}, {n1, n2, n3});
    bulk_ng = new MDRangePolicy<Rank<3>>({ng, ng, ng}, {n1+ng, n2+ng, n3+ng});
    all_0 = new MDRangePolicy<Rank<3>>({0, 0, 0}, {n1+2*ng, n2+2*ng, n3+2*ng});

    coords = coords_in;
}

#if USE_MPI
Grid::Grid(std::vector<int> starti, std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng) :
{

}
#endif

/**
 * Function to return native coordinates of a grid
 */
template<typename T>
KOKKOS_INLINE_FUNCTION void Grid::coord(int i, int j, int k, Loci loc, T X) const
{
#if USE_MPI
    i += n1start;
    j += n2start;
    k += n3start;
#endif

    X[0] = 0;
    switch (loc)
    {
    case face1:
        X[1] = startx1 + (i - ng) * dx1;
        X[2] = startx2 + (j + 0.5 - ng) * dx2;
        X[3] = startx3 + (k + 0.5 - ng) * dx3;
        break;
    case face2:
        X[1] = startx1 + (i + 0.5 - ng) * dx1;
        X[2] = startx2 + (j - ng) * dx2;
        X[3] = startx3 + (k + 0.5 - ng) * dx3;
        break;
    case face3:
        X[1] = startx1 + (i + 0.5 - ng) * dx1;
        X[2] = startx2 + (j + 0.5 - ng) * dx2;
        X[3] = startx3 + (k - ng) * dx3;
        break;
    case center:
        X[1] = startx1 + (i + 0.5 - ng) * dx1;
        X[2] = startx2 + (j + 0.5 - ng) * dx2;
        X[3] = startx3 + (k + 0.5 - ng) * dx3;
        break;
    case corner:
        X[1] = startx1 + (i - ng) * dx1;
        X[2] = startx2 + (j - ng) * dx2;
        X[3] = startx3 + (k - ng) * dx3;
        break;
    }
}
