

#include "containers.hpp"

#include "decs.hpp"

using namespace Kokkos;

TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string>& stage_name,
                            Integrator* integrator) {
    const Real beta = integrator->beta[stage-1];
    Real dt = integrator->dt;
    Container<Real>& base = pmb->real_containers.Get();
    Container<Real>& cin = pmb->real_containers.Get(stage_name[stage-1]);
    Container<Real>& cout = pmb->real_containers.Get(stage_name[stage]);
    Container<Real>& dudt = pmb->real_containers.Get("dUdt");
    parthenon::Update::AverageContainers(cin, base, beta);
    parthenon::Update::UpdateContainer(cin, dudt, beta*dt, cout);
    return TaskStatus::complete;
}

TaskStatus CopyField(std::string& var, Container<Real>& rc0, Container<Real>& rc1)
{
    MeshBlock *pmb = rc0.pmy_block;
    GridVars v0 = rc0.Get(var).data;
    GridVars v1 = rc1.Get(var).data;
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // TODO revisit this when par_for is restored to glory
    Kokkos::parallel_for("copy_field", MDRangePolicy<Rank<4>>({0, ks, js, is}, {NPRIM, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA_VARS {
            v1(p, k, j, i) = v0(p, k, j, i);
        }
    );
    return TaskStatus::complete;
}