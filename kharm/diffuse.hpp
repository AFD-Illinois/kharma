/*
 * diffuse: Simple non-physical diffusion kernel for tests
 */

#include "grid.hpp"
#include "decs.hpp"

using namespace Kokkos;

template <class Coords>
void diffuse_all(const Grid<Coords> &G, const GridVars in, GridVars out)
{
    int np = G.nvar;
    Kokkos::parallel_for("diff_all", G.bulk_ng(),
        KOKKOS_LAMBDA (const int i, const int j, const int k) {
            for (int p=0; p < np; ++p)
                out(i, j, k, p) = in(i-1, j, k, p) + in(i, j-1, k, p) + in(i, j, k-1, p) +
                                in(i, j, k, p) + in(i+1, j, k, p) + in(i, j+1, k, p) + in(i, j, k+1, p) / 7;
        }
    );
}

