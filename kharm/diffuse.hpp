/*
 * diffuse: Simple non-physical diffusion kernel for tests
 */

#include "decs.hpp"

using namespace Kokkos;

void diffuse_all(GridVars in, GridVars out)
{
  MDRangePolicy<Rank<3>> in_range({1,1,1}, {in.extent(0)-1, in.extent(1)-1, in.extent(2)-1});
  int np = in.extent(3);
  Kokkos::parallel_for("diff_all", in_range,
               KOKKOS_LAMBDA (const int i, const int j, const int k) {
    for (int p=0; p < np; ++p)
      out(i, j, k, p) = in(i-1, j, k, p) + in(i, j-1, k, p) + in(i, j, k-1, p) +
                        in(i, j, k, p) + in(i+1, j, k, p) + in(i, j+1, k, p) + in(i, j, k+1, p) / 7;
  });
}

