

#include "containers.hpp"

#include "decs.hpp"

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

TaskStatus CopyFluxes(std::string& var, Container<Real>& rc0, Container<Real>& rc1)
{
    MeshBlock *pmb = rc0.pmy_block;
    GridVars v0 = rc0.Get(var).data;
    GridVars v1 = rc1.Get(var).data;

    pmb->par_for("copy_fluxes", 0, NPRIM-1, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke,
        KOKKOS_LAMBDA_VARS {
            v1(p, i, j, k) = v0(p, i, j, k);
        }
    );
    return TaskStatus::complete;
}