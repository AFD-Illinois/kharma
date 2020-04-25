
#include "grid.hpp"


#if FAST_CARTESIAN
/**
 * Construct a cartesian grid
 */
Grid::Grid(MeshBlock* pmb): pmy_block(pmb)
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

    // Cartesian is not spherical
    spherical = false;
}
#else
/**
 * Construct a grid according to preferences set in the package
 */
Grid::Grid(MeshBlock* pmb): pmy_block(pmb)
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

    auto pkg = pmb->packages["GRMHD"];
    std::string base_str = pkg->Param<std::string>("c_base");
    std::string transform_str = pkg->Param<std::string>("c_transform");
    GReal startx1 = pkg->Param<GReal>("c_startx1");
    GReal a = pkg->Param<GReal>("c_a");
    GReal hslope = pkg->Param<GReal>("c_hslope");
    GReal mks_smooth = pkg->Param<GReal>("c_mks_smooth");
    GReal poly_xt = pkg->Param<GReal>("c_poly_xt");
    GReal poly_alpha = pkg->Param<GReal>("c_poly_alpha");

    // This is effectively a constructor for the CoordinateEmbedding object
    // but moving this logic there makes it Parthenon-reliant,
    // and the alternative is like 18 difference constructors for the various argument lists
    // so here we are.
    SomeBaseCoords base;
    if (base_str == "spherical_minkowski") {
        base.emplace<SphMinkowskiCoords>(SphMinkowskiCoords());
        spherical = true;
    } else if (base_str == "cartesian_minkowski" || base_str == "minkowski") {
        base.emplace<CartMinkowskiCoords>(CartMinkowskiCoords());
        spherical = false;
    } else if (base_str == "spherical_ks" || base_str == "ks") {
        base.emplace<SphKSCoords>(SphKSCoords(a));
        spherical = true;
    } else if (base_str == "spherical_bl" || base_str == "bl") {
        base.emplace<SphBLCoords>(SphBLCoords(a));
        spherical = true;
    } else {
        throw std::invalid_argument("Unsupported base coordinates!");
    }

    SomeTransform transform;
    if (transform_str == "null") {
        if (spherical) {
            transform.emplace<SphNullTransform>(SphNullTransform());
        } else {
            transform.emplace<CartNullTransform>(CartNullTransform());
        }
    } else if (base_str == "cartesian_null") {
        if (!spherical) throw std::invalid_argument("Transform is for cartesian coordinates!");
        transform.emplace<CartNullTransform>(CartNullTransform());
    } else if (base_str == "spherical_null") {
        if (!spherical) throw std::invalid_argument("Transform is for spherical coordinates!");
        transform.emplace<SphNullTransform>(SphNullTransform());
    } else if (transform_str == "modified" || transform_str == "mks") {
        if (!spherical) throw std::invalid_argument("Transform is for spherical coordinates!");
        transform.emplace<ModifyTransform>(ModifyTransform(hslope));
    } else if (transform_str == "funky" || transform_str == "fmks") {
        if (!spherical) throw std::invalid_argument("Transform is for spherical coordinates!");
        transform.emplace<FunkyTransform>(FunkyTransform(startx1, hslope, mks_smooth, poly_xt, poly_alpha));
    } else {
        throw std::invalid_argument("Unsupported coordinate transform!");
    }

    coords = new CoordinateEmbedding(base, transform);

    init_grids(*this);
}
#endif

/**
 * Construct a grid with an existing CoordinateEmbedding object
 */
Grid::Grid(MeshBlock* pmb, CoordinateEmbedding* coordinates): pmy_block(pmb), coords(coordinates)
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
    int n1 = G.pmy_block->ncells1;
    int n2 = G.pmy_block->ncells2;
    G.gcon_direct = GeomTensor2("gcon", NLOC, n2, n1, NDIM, NDIM);
    G.gcov_direct = GeomTensor2("gcov", NLOC, n2, n1, NDIM, NDIM);
    G.gdet_direct = GeomScalar("gdet", NLOC, n2, n1);
    G.conn_direct = GeomTensor3("conn", n2, n1, NDIM, NDIM, NDIM);

    // Member variables have an implicit this->
    // Kokkos captures pointers to objects, not full objects
    // Hence, you *CANNOT* use this->, or members, from inside kernels
    auto gcon_local = G.gcon_direct;
    auto gcov_local = G.gcov_direct;
    auto gdet_local = G.gdet_direct;
    auto conn_local = G.conn_direct;
    CoordinateEmbedding cs = *(G.coords);

    G.pmy_block->par_for("init_geom", 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA (const int& j, const int& i) {
            GReal X[NDIM];
            Real gcov_loc[NDIM][NDIM], gcon_loc[NDIM][NDIM];
            for (int loc=0; loc < NLOC; ++loc) {
                G.coord(0, j, i, (Loci)loc, X);
                cs.gcov_native(X, gcov_loc);
                gdet_local(loc, j, i) = cs.gcon_native(gcov_loc, gcon_loc);
                DLOOP2 {
                    gcov_local(loc, j, i, mu, nu) = gcov_loc[mu][nu];
                    gcon_local(loc, j, i, mu, nu) = gcon_loc[mu][nu];
                }
            }
        }
    );
    G.pmy_block->par_for("init_geom", 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA (const int& j, const int& i) {
            GReal X[NDIM];
            G.coord(0, j, i, Loci::center, X);
            Real conn_loc[NDIM][NDIM][NDIM];
            cs.conn_native(X, conn_loc);
            DLOOP3 conn_local(j, i, mu, nu, lam) = conn_loc[mu][nu][lam];
        }
    );

    FLAG("Grid metric init");
}
#endif