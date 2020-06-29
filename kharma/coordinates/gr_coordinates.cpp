/*
 * Coordinate functions for GR 
 */

#include "decs.hpp"

#include "gr_coordinates.hpp"

#include "Kokkos_Core.hpp"

#include "parameter_input.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

// This file doesn't have MeshBlock access, so it uses raw Kokkos calls
using namespace Kokkos;

// Internal function for initializing cache
void init_GRCoordinates(GRCoordinates& G, int n1, int n2, int n3);

#if FAST_CARTESIAN
/**
 * Construct a cartesian GRCoordinates object
 */
GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin)
{
    // Cartesian is not spherical
    spherical = false;
}
#else
/**
 * Construct a GRCoordinates object according to preferences set in the package
 */
GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin)
{
    // This is effectively a constructor for the CoordinateEmbedding object,
    // but in KHARMA, that object is only used through this one.
    // And I want the option to use that code elsewhere as it's quite general & nice
    // TODO are string allocations/comparisons bad in this constructor?
    std::string base_str = pin->GetOrAddString("coordinates", "base", "cartesian_minkowski");
    std::string transform_str = pin->GetOrAddString("coordinates", "transform", "null");
    GReal startx1 = pin->GetReal("parthenon/mesh", "x1min"); // This was needed for mesh.  Die without it.
    GReal a = pin->GetOrAddReal("coordinates", "a", 0.0); // The rest have defaults
    GReal hslope = pin->GetOrAddReal("coordinates", "hslope", 0.3);
    GReal mks_smooth = pin->GetOrAddReal("coordinates", "mks_smooth", 0.5);
    GReal poly_xt = pin->GetOrAddReal("coordinates", "poly_xt", 0.82);
    GReal poly_alpha = pin->GetOrAddReal("coordinates", "poly_alpha", 14.0);

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
    } else if (transform_str == "cartesian_null") {
        if (!spherical) throw std::invalid_argument("Transform is for cartesian coordinates!");
        transform.emplace<CartNullTransform>(CartNullTransform());
    } else if (transform_str == "spherical_null") {
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

    n1 = rs.nx1 + 2*NGHOST;
    n2 = rs.nx2 > 1 ? rs.nx2 + 2*NGHOST : 1;
    n3 = rs.nx3 > 1 ? rs.nx3 + 2*NGHOST : 1;

    init_GRCoordinates(*this, n1, n2, n3);
}
#endif

// OTHER CONSTRUCTORS: Same between implementations

GRCoordinates::GRCoordinates(const GRCoordinates &src, int coarsen): UniformCartesian(src, coarsen)
{
    std::cerr << "Calling the questionable constructor" << std::endl;
    spherical = src.spherical;
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
#if FAST_CARTESIAN || NO_CACHE
void init_GRCoordinates(GRCoordinates& G, int n1, int n2, int n3) {}
#else
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
    CoordinateEmbedding cs = *(G.coords);

    Kokkos::parallel_for("init_geom", MDRangePolicy<Rank<2>>({0,0}, {n2, n1}),
        KOKKOS_LAMBDA_2D {
            GReal X[GR_DIM];
            Real gcov_loc[GR_DIM][GR_DIM], gcon_loc[GR_DIM][GR_DIM];
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
    Kokkos::parallel_for("init_geom", MDRangePolicy<Rank<2>>({0,0}, {n2, n1}),
        KOKKOS_LAMBDA_2D {
            GReal X[GR_DIM];
            G.coord(0, j, i, Loci::center, X);
            Real conn_loc[GR_DIM][GR_DIM][GR_DIM];
            cs.conn_native(X, conn_loc);
            DLOOP3 conn_local(j, i, mu, nu, lam) = conn_loc[mu][nu][lam];
        }
    );

    FLAG("GRCoordinates metric init");
}
#endif