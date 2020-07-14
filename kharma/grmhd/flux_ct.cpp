
#include "decs.hpp"

#include "flux_ct.hpp"

#include "basic_types.hpp"
#include "interface/container.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

using namespace parthenon;

namespace GRMHD {

/**
 * Constrained transport.  Modify B-field fluxes to preserve divB==0 condition to machine precision per-step
 */
parthenon::TaskStatus FluxCT(std::shared_ptr<Container<Real>>& rc)
{
    FLAG("Flux CT");
    MeshBlock *pmb = rc->pmy_block;
    GridVars F1 = rc->Get("c.c.bulk.cons").flux[X1DIR];
    GridVars F2 = rc->Get("c.c.bulk.cons").flux[X2DIR];
    GridVars F3 = rc->Get("c.c.bulk.cons").flux[X3DIR];

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    GridScalar emf1("emf1", n3, n2, n1);
    GridScalar emf2("emf2", n3, n2, n1);
    GridScalar emf3("emf3", n3, n2, n1);

    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("flux_ct_emf", ks+1, ke, js+1, je, is+1, ie,
        KOKKOS_LAMBDA_3D {
            emf3(k, j, i) =  0.25 * (F1(prims::B2, k, j, i) + F1(prims::B2, k, j-1, i) - F2(prims::B1, k, j, i) - F2(prims::B1, k, j, i-1));
            emf2(k, j, i) = -0.25 * (F1(prims::B3, k, j, i) + F1(prims::B3, k-1, j, i) - F3(prims::B1, k, j, i) - F3(prims::B1, k, j, i-1));
            emf1(k, j, i) =  0.25 * (F2(prims::B3, k, j, i) + F2(prims::B3, k-1, j, i) - F3(prims::B2, k, j, i) - F3(prims::B2, k, j-1, i));
        }
    );

    // Rewrite EMFs as fluxes, after Toth
    // TODO split to cover more ground?
    pmb->par_for("flux_ct", ks, ke-1, js, je-1, is, ie-1,
        KOKKOS_LAMBDA_3D {
            F1(prims::B1, k, j, i) =  0.0;
            F1(prims::B2, k, j, i) =  0.5 * (emf3(k, j, i) + emf3(k, j+1, i));
            F1(prims::B3, k, j, i) = -0.5 * (emf2(k, j, i) + emf2(k+1, j, i));

            F2(prims::B1, k, j, i) = -0.5 * (emf3(k, j, i) + emf3(k, j, i+1));
            F2(prims::B2, k, j, i) =  0.0;
            F2(prims::B3, k, j, i) =  0.5 * (emf1(k, j, i) + emf1(k+1, j, i));

            F3(prims::B1, k, j, i) =  0.5 * (emf2(k, j, i) + emf2(k, j, i+1));
            F3(prims::B2, k, j, i) = -0.5 * (emf1(k, j, i) + emf1(k, j+1, i));
            F3(prims::B3, k, j, i) =  0.0;
        }
    );
    FLAG("CT Finished");

    return TaskStatus::complete;
}

}