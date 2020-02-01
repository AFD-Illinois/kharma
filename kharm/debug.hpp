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
void print_derived_at(const GridDerived D, const int i, const int j, const int k) {
    auto ucon_h = Kokkos::create_mirror_view(D.ucon);
    auto ucov_h = Kokkos::create_mirror_view(D.ucov);
    auto bcon_h = Kokkos::create_mirror_view(D.bcon);
    auto bcov_h = Kokkos::create_mirror_view(D.bcov);

    Kokkos::deep_copy(ucon_h, D.ucon);
    Kokkos::deep_copy(ucov_h, D.ucov);
    Kokkos::deep_copy(bcon_h, D.bcon);
    Kokkos::deep_copy(bcov_h, D.bcov);

    std::cerr << string_format("DERIVED LOCATION %d %d %d",i,j,k) << std::endl;
    std::cerr << ucon_h.label() << string_format(" is [%f %f %f %f]",
                                    ucon_h(i,j,k,0), ucon_h(i,j,k,0), ucon_h(i,j,k,0), ucon_h(i,j,k,0)) << std::endl;
    std::cerr << ucov_h.label() << string_format(" is [%f %f %f %f]",
                                    ucov_h(i,j,k,0), ucov_h(i,j,k,0), ucov_h(i,j,k,0), ucov_h(i,j,k,0)) << std::endl;
    std::cerr << bcon_h.label() << string_format(" is [%f %f %f %f]",
                                    bcon_h(i,j,k,0), bcon_h(i,j,k,0), bcon_h(i,j,k,0), bcon_h(i,j,k,0)) << std::endl;
    std::cerr << bcov_h.label() << string_format(" is [%f %f %f %f]",
                                    bcov_h(i,j,k,0), bcov_h(i,j,k,0), bcov_h(i,j,k,0), bcov_h(i,j,k,0)) << std::endl;
}