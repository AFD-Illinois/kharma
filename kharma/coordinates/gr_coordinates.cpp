/*
 * Coordinate functions for GR 
 */

#include "decs.hpp"

#include "debug.hpp"
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
 * Fast Cartesian GRCoordinates just use the underlying UniformCartesian object for everything
 */
GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin) {}
#else
/**
 * Construct a GRCoordinates object with a transformation according to preferences set in the package
 */
GRCoordinates::GRCoordinates(const RegionSize &rs, ParameterInput *pin): UniformCartesian(rs, pin)
{
    // This is effectively a constructor for the CoordinateEmbedding object,
    // but in KHARMA, that object is only used through this one.
    // And I want the option to use that code elsewhere as it's quite general & nice
    std::string base_str = pin->GetString("coordinates", "base"); // Require every problem to specify very basic geometry
    std::string transform_str = pin->GetString("coordinates", "transform");
    GReal startx1 = pin->GetReal("parthenon/mesh", "x1min");
    GReal a = pin->GetReal("coordinates", "a");
    GReal hslope = pin->GetOrAddReal("coordinates", "hslope", 0.3); // The rest have very common defaults
    GReal mks_smooth = pin->GetOrAddReal("coordinates", "mks_smooth", 0.5);
    GReal poly_xt = pin->GetOrAddReal("coordinates", "poly_xt", 0.82);
    GReal poly_alpha = pin->GetOrAddReal("coordinates", "poly_alpha", 14.0);

    SomeBaseCoords base;
    if (base_str == "spherical_minkowski") {
        base.emplace<SphMinkowskiCoords>(SphMinkowskiCoords());
    } else if (base_str == "cartesian_minkowski" || base_str == "minkowski") {
        base.emplace<CartMinkowskiCoords>(CartMinkowskiCoords());
    } else if (base_str == "spherical_ks" || base_str == "ks") {
        base.emplace<SphKSCoords>(SphKSCoords(a));
    } else if (base_str == "spherical_bl" || base_str == "bl") {
        base.emplace<SphBLCoords>(SphBLCoords(a));
    } else {
        throw std::invalid_argument("Unsupported base coordinates!");
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
        if (!spherical) throw std::invalid_argument("Transform is for spherical coordinates!");
        transform.emplace<ModifyTransform>(ModifyTransform(hslope));
    } else if (transform_str == "funky" || transform_str == "fmks") {
        if (!spherical) throw std::invalid_argument("Transform is for spherical coordinates!");
        transform.emplace<FunkyTransform>(FunkyTransform(startx1, hslope, mks_smooth, poly_xt, poly_alpha));
    } else {
        throw std::invalid_argument("Unsupported coordinate transform!");
    }

    coords = CoordinateEmbedding(base, transform);

    n1 = rs.nx1 + 2*NGHOST;
    n2 = rs.nx2 > 1 ? rs.nx2 + 2*NGHOST : 1;
    n3 = rs.nx3 > 1 ? rs.nx3 + 2*NGHOST : 1;

    init_GRCoordinates(*this, n1, n2, n3);
}
#endif

// OTHER CONSTRUCTORS: Same between implementations

GRCoordinates::GRCoordinates(const GRCoordinates &src, int coarsen): UniformCartesian(src, coarsen)
{
    //std::cerr << "Calling coarsen constructor" << std::endl;
    coords = src.coords;
    n1 = src.n1/coarsen;
    n2 = src.n2/coarsen;
    n3 = src.n3/coarsen;
    init_GRCoordinates(*this, n1, n2, n3);
}

GRCoordinates::GRCoordinates(const GRCoordinates &src): UniformCartesian(src)
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

GRCoordinates GRCoordinates::operator=(const GRCoordinates& src)
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