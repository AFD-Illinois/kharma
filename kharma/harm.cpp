/*
 * HARM driver-specific things -- i.e. call the GRMHD physics module in
 * the correct RK2 LLF steps we know and love
 */

#include <iostream>

#include "parthenon_manager.hpp"
#include "bvals/boundary_conditions.hpp"
#include "bvals/bvals.hpp"
#include "driver/multistage.hpp"

#include "grmhd.hpp"
#include "harm.hpp"

// Parthenon requires we override certain things
namespace parthenon {

    Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput>& pin) {
        Packages_t packages;

        // Turn off GRMHD only if set to false in input file
        bool do_hydro = pin->GetOrAddBoolean("Physics", "GRMHD", true);
        bool do_electrons = pin->GetOrAddBoolean("Physics", "howes_electrons", false);

        // enable other packages as needed
        bool do_scalars = pin->GetOrAddBoolean("Physics", "scalars", false);

        if (do_hydro) {
            packages["GRMHD"] = GRMHD::Initialize(pin.get());
        }

        // TODO scalars. Or will Parthenon?
        // if (do_scalars) {
        //     packages["scalars"] = BetterScalars::Initialize(pin.get());
        // }

        // TODO electrons, like scalars but...

        return std::move(packages);
    }

} // namespace parthenon

// Implement HARMDriver class methods
TaskList HARMDriver::MakeTaskList(MeshBlock *pmb, int stage)
{
    TaskList tl;
    // we're going to populate our last with multiple kinds of tasks
    // these lambdas just clean up the interface to adding tasks of the relevant kinds
    auto AddMyTask =
    [&tl, pmb, stage, this] (BlockStageNamesIntegratorTaskFunc func, TaskID dep) {
    return tl.AddTask<BlockStageNamesIntegratorTask>(func,dep,pmb,stage,
                                                        stage_name,integrator);
    };
    auto AddContainerTask =
    [&tl] (ContainerTaskFunc func, TaskID dep, Container<Real>& rc) {
    return tl.AddTask<ContainerTask>(func,dep,rc);
    };
    auto AddTwoContainerTask =
    [&tl] (TwoContainerTaskFunc f, TaskID dep, Container<Real>& rc1, Container<Real>& rc2) {
    return tl.AddTask<TwoContainerTask>(f,dep,rc1,rc2);
    };

    TaskID none(0);
    // first make other useful containers
    if (stage == 1) {
        Container<Real>& base = pmb->real_containers.Get();
        pmb->real_containers.Add("dUdt", base);
        for (int i=1; i<integrator->nstages; i++)
            pmb->real_containers.Add(stage_name[i], base);
    }

    // pull out the container we'll use to get fluxes and/or compute RHSs
    Container<Real>& sc0  = pmb->real_containers.Get(stage_name[stage-1]);
    // pull out a container we'll use to store dU/dt.
    // This is just -flux_divergence in this example
    Container<Real>& dudt = pmb->real_containers.Get("dUdt");
    // pull out the container that will hold the updated state
    // effectively, sc1 = sc0 + dudt*dt
    Container<Real>& sc1  = pmb->real_containers.Get(stage_name[stage]);

    auto start_recv = AddContainerTask(Container<Real>::StartReceivingTask, none, sc1);

    auto advect_flux = AddContainerTask(GRMHD::CalculateFluxes, none, sc0);

    auto send_flux = AddContainerTask(Container<Real>::SendFluxCorrectionTask,
                                    advect_flux, sc0);
    auto recv_flux = AddContainerTask(Container<Real>::ReceiveFluxCorrectionTask,
                                    advect_flux, sc0);

    // compute the divergence of fluxes of conserved variables
    auto flux_div = AddTwoContainerTask(parthenon::Update::FluxDivergence,
                                        recv_flux, sc0, dudt);

    // apply du/dt to all independent fields in the container
    auto update_container = AddMyTask(UpdateContainer, flux_div);

    // update ghost cells
    auto send = AddContainerTask(Container<Real>::SendBoundaryBuffersTask,
                                update_container, sc1);
    auto recv = AddContainerTask(Container<Real>::ReceiveBoundaryBuffersTask,
                                send, sc1);
    auto fill_from_bufs = AddContainerTask(Container<Real>::SetBoundariesTask,
                                            recv, sc1);
    auto clear_comm_flags = AddContainerTask(Container<Real>::ClearBoundaryTask,
                                            fill_from_bufs, sc1);

    auto prolongBound = tl.AddTask<BlockTask>([](MeshBlock *pmb) {
    pmb->pbval->ProlongateBoundaries(0.0, 0.0);
    return TaskStatus::complete;
    }, fill_from_bufs, pmb);

    // set physical boundaries
    auto set_bc = AddContainerTask(parthenon::ApplyBoundaryConditions,
                                    prolongBound, sc1);

    // fill in derived fields
    auto fill_derived = AddContainerTask(parthenon::FillDerivedVariables::FillDerived,
                                        set_bc, sc1);

    // estimate next time step
    if (stage == integrator->nstages) {
        auto new_dt = AddContainerTask([](Container<Real>& rc) {
            MeshBlock *pmb = rc.pmy_block;
            pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
            return TaskStatus::complete;
        }, fill_derived, sc1);

        // Update refinement
        if (pmesh->adaptive) {
                auto tag_refine = tl.AddTask<BlockTask>([](MeshBlock *pmb) {
                    pmb->pmr->CheckRefinementCondition();
                    return TaskStatus::complete;
                }, fill_derived, pmb);
        }
        // Purge stages
        auto purge_stages = tl.AddTask<BlockTask>([](MeshBlock *pmb) {
                pmb->real_containers.PurgeNonBase();
                return TaskStatus::complete;
            }, fill_derived, pmb);
    }
    return tl;
}