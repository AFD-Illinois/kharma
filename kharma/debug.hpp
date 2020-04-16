/**
 * Tools for debuggging, printing, whatever
 */
 #pragma once

#include "decs.hpp"
#include "utils.hpp"

/**
 * These are obviously unreasonably slow.  Don't use in production
 */
template <typename PType>
void print_a_zone(const PType P, const int i, const int j, const int k) {
    auto Q = Kokkos::create_mirror_view(P);
    Kokkos::deep_copy(Q, P);
    std::cerr << Q.label() << string_format(" cell %d %d %d is [%f %f %f %f %f %f %f %f]", i, j, k,
                            Q(i,j,k,0), Q(i,j,k,1), Q(i,j,k,2), Q(i,j,k,3), Q(i,j,k,4),
                            Q(i,j,k,5), Q(i,j,k,6), Q(i,j,k,7)) << std::endl;
}
void print_a_geom_tensor(const GeomTensor g, const Loci loc, const int i, const int j) {
    auto h = Kokkos::create_mirror_view(g);
    Kokkos::deep_copy(h, g);
    std::cerr << h.label() << string_format(" element %d %d diagonal is [%f %f %f %f]", i, j,
                            h(loc,i,j,0,0), h(loc,i,j,1,1), h(loc,i,j,2,2),
                            h(loc,i,j,3,3)) << std::endl;
}
void print_a_geom_scalar(const GeomScalar g, const Loci loc, const int i, const int j) {
    auto h = Kokkos::create_mirror_view(g);
    Kokkos::deep_copy(h, g);
    std::cerr << h.label() << string_format(" at %d %d is %f", i, j, h(loc,i,j)) << std::endl;
}
void print_a_geom(const Grid G, const int i, const int j) {
    std::cerr << "GRID VARS" << std::endl;
    for (int loc=0; loc < NLOC; ++loc) {
        print_a_geom_tensor(G.gcon_direct, (Loci) loc, i, j);
        print_a_geom_tensor(G.gcov_direct, (Loci) loc, i, j);
        print_a_geom_scalar(G.gdet_direct, (Loci) loc, i, j);
    }
}
void print_a_scalar(const GridScalar scalar, const int i, const int j, const int k) {
    auto scalar_h = Kokkos::create_mirror_view(scalar);
    Kokkos::deep_copy(scalar_h, scalar);
    std::cerr << scalar_h.label() << string_format(" at %d %d %d is %f", i, j, k, scalar_h(i, j, k)) << std::endl;

}
void print_a_vec(const GridVector vec, const int i, const int j, const int k) {
    auto vec_h = Kokkos::create_mirror_view(vec);
    Kokkos::deep_copy(vec_h, vec);
    std::cerr << vec_h.label() << string_format(" at %d %d %d is [%f %f %f %f]", i, j, k,
                        vec_h(i,j,k,0), vec_h(i,j,k,1), vec_h(i,j,k,2), vec_h(i,j,k,3)) << std::endl;

}
void print_derived_at(const GridDerived D, const int i, const int j, const int k) {
    std::cerr << "DERIVED VARS" << std::endl;
    print_a_vec(D.ucon, i, j, k);
    print_a_vec(D.ucov, i, j, k);
    print_a_vec(D.bcon, i, j, k);
    print_a_vec(D.bcov, i, j, k);
}

template<typename T>
void printf_device(T to_print)
{
#if defined( Kokkos_ENABLE_CUDA )
#else
    cerr << to_print << endl;
#endif
}

template<typename T>
void printf_device_vector(T vector)
{
#if defined( Kokkos_ENABLE_CUDA )
#else
    cerr << vector[0] << " " << vector[1] << " " << vector[2] << " " << vector[3] << endl;
#endif
}

// TODO generalized NaN check?

void count_print_flags(const GridInt pflags)
{
    int n_tot = 0, n_neg_in = 0, n_max_iter = 0;
    int n_utsq = 0, n_gamma = 0, n_neg_u = 0, n_neg_rho = 0, n_neg_both = 0;

    auto h_pflags = create_mirror_view(pflags);
    deep_copy(h_pflags, pflags);

    for(int i=0; i < h_pflags.extent(0); ++i)
        for(int j=0; j < h_pflags.extent(1); ++j)
            for(int k=0; k < h_pflags.extent(2); ++k) {
                if (h_pflags(i, j, k) != 0) ++n_tot;
                if (h_pflags(i, j, k) == ERR_NEG_INPUT) ++n_neg_in;
                if (h_pflags(i, j, k) == ERR_MAX_ITER) ++n_max_iter;
                if (h_pflags(i, j, k) == ERR_UTSQ) ++n_utsq;
                if (h_pflags(i, j, k) == ERR_GAMMA) ++n_gamma;
                if (h_pflags(i, j, k) == ERR_RHO_NEGATIVE) ++n_neg_rho;
                if (h_pflags(i, j, k) == ERR_U_NEGATIVE) ++n_neg_u;
                if (h_pflags(i, j, k) == ERR_BOTH_NEGATIVE) ++n_neg_both;
            }
    cerr << "PFLAGS:" << endl;
    cerr << "Negative input: " << n_neg_in << endl;
    cerr << "Hit max iter: " << n_max_iter << endl;
    cerr << "Velocity invalid: " << n_utsq << endl;
    cerr << "Gamma invalid: " << n_gamma << endl;
    cerr << "Negative rho: " << n_neg_rho << endl;
    cerr << "Negative U: " << n_neg_u << endl;
    cerr << "Negative rho & U: " << n_neg_both << endl << endl;
}

// TODO count_print_floors when that happens