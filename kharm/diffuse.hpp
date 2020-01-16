
#include "decs.hpp"

using namespace Kokkos;

KOKKOS_INLINE_FUNCTION void diffuse_all(GridPrims in, GridPrims out)
{
  MDRangePolicy<Rank<3>> in_range({1,1,1}, {in.extent(0)-1, in.extent(1)-1, in.extent(2)-1});
  parallel_for("diff_all", in_range,
               KOKKOS_LAMBDA (const int i, const int j, const int k) {
    int np = in.extent(3);
    for (int p=0; p < np; ++p)
      out(i, j, k, p) = in(i-1, j, k, p) + in(i, j-1, k, p) + in(i, j, k-1, p) +
                        in(i, j, k, p) + in(i+1, j, k, p) + in(i, j+1, k, p) + in(i, j, k+1, p) / 7;
  });
}

