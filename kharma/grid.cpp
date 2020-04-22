
#include "grid.hpp"


#if FAST_CARTESIAN
Grid::Grid(MeshBlock* pmb)
{
    // TODO use parthenon more directly via pass-through,
    // or else take mesh bounds and re-calculate zone locations on the fly...
    // at least this isn't a full copy or anything.  Just pointers.
    x1f = pmb->pcoord->x1f;
    x2f = pmb->pcoord->x2f;
    x3f = pmb->pcoord->x3f;

    x1v = pmb->pcoord->x1v;
    x2v = pmb->pcoord->x2v;
    x3v = pmb->pcoord->x3v;
}
#endif

/**
 * Construct a grid which starts at 0 and covers the entire global space
 */
Grid::Grid(CoordinateEmbedding* coordinates, MeshBlock* pmb): coords(coordinates)
{
    // TODO use parthenon more directly via pass-through,
    // or else take mesh bounds and re-calculate zone locations on the fly...
    // at least this isn't a full copy or anything.  Just pointers.
    x1f = pmb->pcoord->x1f;
    x2f = pmb->pcoord->x2f;
    x3f = pmb->pcoord->x3f;

    x1v = pmb->pcoord->x1v;
    x2v = pmb->pcoord->x2v;
    x3v = pmb->pcoord->x3v;

    init_grids(*this);
}

/**
 * Initialize any cached geometry that Grid will need to return.
 *
 * This needs to be defined *outside* of the Grid object, because of some
 * fun issues with C++ Lambda capture, which Kokkos brings to the fore
 */
#if FAST_CARTESIAN || NO_CACHE
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