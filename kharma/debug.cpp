// Debug tools that aren't templated

#include "decs.hpp"

#include "mesh/mesh.hpp"

using namespace Kokkos;

double MaxDivB(std::shared_ptr<Container<Real>>& rc, IndexDomain domain)
{
    FLAG("Calculating divB");
    MeshBlock *pmb = rc->pmy_block;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    GRCoordinates G = pmb->coords;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    double max_divb;
    Kokkos::Max<double> max_reducer(max_divb);
    Kokkos::parallel_reduce("divB", MDRangePolicy<Rank<3>>({ks+1, js+1, is+1}, {ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA_3D_REDUCE {
            double local_divb = fabs(0.25*(
                              P(prims::B1, k, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B1, k, j-1, i) * G.gdet(Loci::center, j-1, i)
                            + P(prims::B1, k-1, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B1, k-1, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B1, k, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B1, k, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            - P(prims::B1, k-1, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B1, k-1, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            )/G.dx1v(i) +
                            0.25*(
                              P(prims::B2, k, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B2, k, j, i-1) * G.gdet(Loci::center, j, i-1)
                            + P(prims::B2, k-1, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B2, k-1, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B2, k, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B2, k, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            - P(prims::B2, k-1, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B2, k-1, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            )/G.dx2v(j) +
                            0.25*(
                              P(prims::B3, k, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B3, k, j-1, i) * G.gdet(Loci::center, j-1, i)
                            + P(prims::B3, k, j, i-1) * G.gdet(Loci::center, j, i-1)
                            + P(prims::B3, k, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            - P(prims::B3, k-1, j, i) * G.gdet(Loci::center, j, i)
                            - P(prims::B3, k-1, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B3, k-1, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B3, k-1, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            )/G.dx3v(k));
            if (local_divb > local_result) local_result = local_divb;
        }
    , max_reducer);

    return max_divb;
}