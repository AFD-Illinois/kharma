

#include "containers.hpp"

TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string>& stage_name,
                            Integrator* integrator) {
    const Real beta = integrator->beta[stage-1];
    Container<Real>& base = pmb->real_containers.Get();
    Container<Real>& cin = pmb->real_containers.Get(stage_name[stage-1]);
    Container<Real>& cout = pmb->real_containers.Get(stage_name[stage]);
    Container<Real>& dudt = pmb->real_containers.Get("dUdt");
    parthenon::Update::AverageContainers(cin, base, beta);
    parthenon::Update::UpdateContainer(cin, dudt, beta*pmb->pmy_mesh->dt, cout);
    return TaskStatus::complete;
}