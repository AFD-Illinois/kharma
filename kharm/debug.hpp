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
void print_a_cell(const PType P, const int i, const int j, const int k) {
    auto Q = Kokkos::create_mirror_view(P);
    Kokkos::deep_copy(Q, P);
    std::cerr << Q.label() << string_format(" cell %d %d %d is [%f %f %f %f %f %f %f %f]", i, j, k,
                            Q(i,j,k,0), Q(i,j,k,1), Q(i,j,k,2), Q(i,j,k,3), Q(i,j,k,4),
                            Q(i,j,k,5), Q(i,j,k,6), Q(i,j,k,7)) << std::endl;
}
template <typename GTType>
void print_a_grid_cell(const GTType g, const int i, const int j) {
    auto h = Kokkos::create_mirror_view(g);
    Kokkos::deep_copy(h, g);
    std::cerr << h.label() << string_format(" element %d %d diagonal is [%f %f %f %f]", i, j,
                            h(Loci::center,i,j,0,0), h(Loci::center,i,j,1,1), h(Loci::center,i,j,1,1),
                            h(Loci::center,i,j,1,1)) << std::endl;
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
    std::cerr << string_format("DERIVED VARS",i,j,k) << std::endl;
    print_a_vec(D.ucon, i, j, k);
    print_a_vec(D.ucov, i, j, k);
    print_a_vec(D.bcon, i, j, k);
    print_a_vec(D.bcov, i, j, k);
}