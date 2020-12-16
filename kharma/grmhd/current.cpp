

#include "current.hpp"

TaskStatus CalculateCurrent(MeshBlockData<Real> *rc0, MeshBlockData<Real> *rc1, const double& dt)
{
    FLAG("Calculating current");

    auto pmb = rc0->GetBlockPointer();
    auto& P_old = rc0->Get("c.c.bulk.prims").data;
    auto& P_new = rc1->Get("c.c.bulk.prims").data;
    auto& jcon = rc1->Get("c.c.bulk.jcon").data;
    auto& G = pmb->coords;

    IndexDomain domain = IndexDomain::entire;
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    GridVars P_c("P_c", n3, n2, n1);

    // Calculate time-centered P.  Parthenon can do this with whole containers, but we only need P,(B)
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("get_center_P", 0, NPRIM-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_VARS {
            P_c(p, k, j, i) = 0.5*(P_old(p, k, j, i) + P_new(p, k, j, i));
        }
    );

    // Calculate j^{\mu} using centered differences for active zones
    // TODO 
    domain = IndexDomain::interior;
    is = pmb->cellbounds.is(domain); ie = pmb->cellbounds.ie(domain);
    js = pmb->cellbounds.js(domain); je = pmb->cellbounds.je(domain);
    ks = pmb->cellbounds.ks(domain); ke = pmb->cellbounds.ke(domain);
    pmb->par_for("jcon_calc", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {

            // For each direction mu...
            DLOOP1 {
                // Get sqrt{-g}*F^{mu nu} at neighboring points
                Real gF0p = G.gdet(Loci::center, j, i) * get_Fcon(G, P_new, 0, mu, k, j, i);
                Real gF0m = G.gdet(Loci::center, j, i) * get_Fcon(G, P_old, 0, mu, k, j, i);
                Real gF1p = G.gdet(Loci::center, j, i+1) * get_Fcon(G, P_c, 1, mu, k, j, i+1);
                Real gF1m = G.gdet(Loci::center, j, i-1) * get_Fcon(G, P_c, 1, mu, k, j, i-1);
                Real gF2p = G.gdet(Loci::center, j+1, i) * get_Fcon(G, P_c, 2, mu, k, j+1, i);
                Real gF2m = G.gdet(Loci::center, j-1, i) * get_Fcon(G, P_c, 2, mu, k, j-1, i);
                Real gF3p = G.gdet(Loci::center, j, i) * get_Fcon(G, P_c, 3, mu, k+1, j, i);
                Real gF3m = G.gdet(Loci::center, j, i) * get_Fcon(G, P_c, 3, mu, k-1, j, i);

                // Difference: D_mu F^{mu nu} = 4 \pi j^nu
                jcon(mu, k, j, i) = 1. / (sqrt(4. * M_PI) * G.gdet(Loci::center, j, i)) *
                                    ((gF0p - gF0m) / dt +
                                    (gF1p - gF1m) / (2. * G.dx1v(i)) +
                                    (gF2p - gF2m) / (2. * G.dx2v(j)) +
                                    (gF3p - gF3m) / (2. * G.dx3v(k)));
            }
        }
    );

    FLAG("Calculated");
    return TaskStatus::complete;
}

// Calculate field rotation rate
// void omega_calc(struct GridGeom *G, struct FluidState *S, GridDouble *omega)
// {
//   static GridDouble *gFcov01, *gFcov13;

//   //TODO test inverting these loops, esp if allows writing to omega sooner
// #pragma omp parallel for simd collapse(3)
//   DLOOP2 {
//     ZLOOP {
//       double gFmunu = gFcon_calc(G, S, mu, nu, k, j, i);
//       (*gFcov01)[k][j][i] += gFmunu*G->gcov[CENT][mu][0][j][i]*G->gcov[CENT][nu][1][j][i];
//       (*gFcov13)[k][j][i] += gFmunu*G->gcov[CENT][mu][1][j][i]*G->gcov[CENT][nu][3][j][i];
//     }
//   }

// #pragma omp parallel for simd collapse(2)
//   ZLOOP {
//     (*omega)[k][j][i] = (*gFcov01)[k][j][i]/(*gFcov13)[k][j][i];
//   }
// }