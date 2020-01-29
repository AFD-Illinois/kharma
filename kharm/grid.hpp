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
 * Struct holding all parameters related to the logically Cartesian grid.  Purposefully minimal 
 */
class Grid
{
public:
    // TODO if we ditch AMR we can probably scrape up time making grids static
    // That or *very* disciplined pass-by-ref
    int n1, n2, n3, nvar;
    int ng, gn1, gn2, gn3;
    int n1tot, n2tot, n3tot;
    int n1start, n2start, n3start;
    GReal startx1, startx2, startx3;
    GReal dx1, dx2, dx3;

    // Constructors
    Grid(std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng_in=3, int nvar_in=8);
    Grid(std::vector<int> fullshape, std::vector<int> startn, std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng_in=3, int nvar_in=8);

    // Coordinates.  Make as generic as we can
    template<typename T>
    KOKKOS_INLINE_FUNCTION void coord(int i, int j, int k, Loci loc, T X) const;

    // Indexing
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

    // RangePolicies over grid.  Construct them on the fly to stay slim
    // TODO can I consolidate these?  Types are fun
    MDRangePolicy<Rank<3>> bulk_0() const {return MDRangePolicy<Rank<3>>({0, 0, 0}, {n1, n2, n3});};
    MDRangePolicy<Rank<3>> bulk_ng() const {return MDRangePolicy<Rank<3>>({ng, ng, ng}, {n1+ng, n2+ng, n3+ng});};
    MDRangePolicy<Rank<3>> all_0() const {return MDRangePolicy<Rank<3>>({0, 0, 0}, {n1+2*ng, n2+2*ng, n3+2*ng});};
    MDRangePolicy<OpenMP,Rank<3>> h_bulk_0() const {return MDRangePolicy<OpenMP,Rank<3>>({0, 0, 0}, {n1, n2, n3});};
    MDRangePolicy<OpenMP,Rank<3>> h_bulk_ng() const {return MDRangePolicy<OpenMP,Rank<3>>({ng, ng, ng}, {n1+ng, n2+ng, n3+ng});};
    MDRangePolicy<OpenMP,Rank<3>> h_all_0() const {return MDRangePolicy<OpenMP,Rank<3>>({0, 0, 0}, {n1+2*ng, n2+2*ng, n3+2*ng});};

    MDRangePolicy<Rank<3>> bulk_plus(const int i) const {return MDRangePolicy<Rank<3>>({ng-i, ng-i, ng-i}, {n1+ng+i, n2+ng+i, n3+ng+i});};

    MDRangePolicy<Rank<4>> bulk_0_p() const {return MDRangePolicy<Rank<3>>({0, 0, 0, 0}, {n1, n2, n3, nvar});};
    MDRangePolicy<Rank<4>> bulk_ng_p() const {return MDRangePolicy<Rank<3>>({ng, ng, ng, 0}, {n1+ng, n2+ng, n3+ng, nvar});};
    MDRangePolicy<Rank<4>> all_0_p() const {return MDRangePolicy<Rank<3>>({0, 0, 0, 0}, {n1+2*ng, n2+2*ng, n3+2*ng, nvar});};

    MDRangePolicy<Rank<4>> bulk_plus_p(const int i) const {return MDRangePolicy<Rank<4>>({ng-i, ng-i, ng-i, 0}, {n1+ng+i, n2+ng+i, n3+ng+i, nvar});};

};

/**
 * Construct a grid which starts at 0 and covers the entire global space
 */
Grid::Grid(std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng_in, int nvar_in)
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
}

/**
 * Construct a sub-grid starting at some point in a global space
 */
Grid::Grid(std::vector<int> fullshape, std::vector<int> startn, std::vector<int> shape, std::vector<GReal> startx, std::vector<GReal> endx, int ng_in, int nvar_in)
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
}

/**
 * Function to return native coordinates of a grid
 * TODO is it more instruction-efficient to split this per location or have a separate one for centers?
 */
template<typename T>
KOKKOS_INLINE_FUNCTION void Grid::coord(int i, int j, int k, Loci loc, T X) const
{
    X[0] = 0;
    switch (loc)
    {
    case face1:
        X[1] = startx1 + (n1start + i - ng) * dx1;
        X[2] = startx2 + (n2start + j + 0.5 - ng) * dx2;
        X[3] = startx3 + (n3start + k + 0.5 - ng) * dx3;
        break;
    case face2:
        X[1] = startx1 + (n1start + i + 0.5 - ng) * dx1;
        X[2] = startx2 + (n2start + j - ng) * dx2;
        X[3] = startx3 + (k + 0.5 - ng) * dx3;
        break;
    case face3:
        X[1] = startx1 + (n1start + i + 0.5 - ng) * dx1;
        X[2] = startx2 + (n2start + j + 0.5 - ng) * dx2;
        X[3] = startx3 + (n3start + k - ng) * dx3;
        break;
    case center:
        X[1] = startx1 + (n1start + i + 0.5 - ng) * dx1;
        X[2] = startx2 + (n2start + j + 0.5 - ng) * dx2;
        X[3] = startx3 + (n3start + k + 0.5 - ng) * dx3;
        break;
    case corner:
        X[1] = startx1 + (n1start + i - ng) * dx1;
        X[2] = startx2 + (n2start + j - ng) * dx2;
        X[3] = startx3 + (n3start + k - ng) * dx3;
        break;
    }
}
