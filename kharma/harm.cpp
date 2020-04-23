/*
 * HARM driver-specific things -- i.e. call the GRMHD physics module in
 * the correct RK2 LLF steps we know and love
 */

#include <iostream>

#include "parthenon_manager.hpp"
#include "bvals/boundary_conditions.hpp"
#include "bvals/bvals.hpp"
#include "driver/multistage.hpp"

#include "decs.hpp"
#include "containers.hpp"
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

        // TODO electrons, like scalars but w/heating step...

        return std::move(packages);
    }

} // namespace parthenon

// Implement HARMDriver class methods
TaskList HARMDriver::MakeTaskList(MeshBlock *pmb, int stage)
{
    TaskList tl;

    TaskID none(0);
    // first make other useful containers
    if (stage == 1) {
        Container<Real> &base = pmb->real_containers.Get();
        pmb->real_containers.Add("dUdt", base);
        for (int i=1; i<integrator->nstages; i++)
            pmb->real_containers.Add(stage_name[i], base);
    }

    // pull out the container we'll use to get fluxes and/or compute RHSs
    Container<Real>& sc0  = pmb->real_containers.Get(stage_name[stage-1]);
      // pull out a container we'll use to store dU/dt.
    Container<Real> &dudt = pmb->real_containers.Get("dUdt");
    // pull out the container that will hold the updated state
    Container<Real>& sc1  = pmb->real_containers.Get(stage_name[stage]);

    auto start_recv = AddContainerTask(tl, Container<Real>::StartReceivingTask, none, sc1);

    // Fill the primitives array P by calculating U_to_P everywhere
    auto fill_prims = AddContainerTask(tl, parthenon::FillDerivedVariables::FillDerived,
                                        start_recv, sc0);
    // Calculate the LLF fluxes in each direction
    auto calculate_flux = AddContainerTask(tl, GRMHD::CalculateFluxes, fill_prims, sc0);
    // Apply these to create a single update dU/dt
    auto apply_flux = AddTwoContainerTask(tl, GRMHD::ApplyFluxes, calculate_flux, sc0, dudt);

    // TODO make sure these are nops for a single-level problem
    auto send_flux = AddContainerTask(tl, Container<Real>::SendFluxCorrectionTask,
                                    apply_flux, sc0);
    auto recv_flux = AddContainerTask(tl, Container<Real>::ReceiveFluxCorrectionTask,
                                    apply_flux, sc0);

    // Apply dU/dt to update values from the last stage to fill the current one
    auto flux_div = AddUpdateTask(tl, pmb, stage, stage_name, integrator, UpdateContainer, recv_flux);

    // update ghost cells
    auto send = AddContainerTask(tl, Container<Real>::SendBoundaryBuffersTask,
                                flux_div, sc1);
    auto recv = AddContainerTask(tl, Container<Real>::ReceiveBoundaryBuffersTask,
                                send, sc1);
    auto fill_from_bufs = AddContainerTask(tl, Container<Real>::SetBoundariesTask,
                                            recv, sc1);
    auto clear_comm_flags = AddContainerTask(tl, Container<Real>::ClearBoundaryTask,
                                            fill_from_bufs, sc1);

    auto prolong_bound = tl.AddTask<BlockTask>([](MeshBlock *pmb) {
        pmb->pbval->ProlongateBoundaries(0.0, 0.0);
        return TaskStatus::complete;
    }, fill_from_bufs, pmb);

    // set physical boundaries
    auto set_bc = AddContainerTask(tl, parthenon::ApplyBoundaryConditions,
                                    prolong_bound, sc1);

    // fill in derived fields. TODO HARM has a special relationship to this w.r.t. U vs P.  Make sure this respects that.
    auto fill_derived = AddContainerTask(tl, parthenon::FillDerivedVariables::FillDerived,
                                        set_bc, sc1);
#if DEBUG
    // Additionally propagate the fluxes
    auto copy_flux1 = AddCopyTask(tl, CopyFluxes, recv_flux, "c.c.bulk.F1", sc0, sc1);
    auto copy_flux2 = AddCopyTask(tl, CopyFluxes, recv_flux, "c.c.bulk.F2", sc0, sc1);
    auto copy_flux3 = AddCopyTask(tl, CopyFluxes, recv_flux, "c.c.bulk.F3", sc0, sc1);
#endif

    // estimate next time step
    if (stage == integrator->nstages) {
        auto new_dt = AddContainerTask(tl, [](Container<Real>& rc) {
            MeshBlock *pmb = rc.pmy_block;
            pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
            return TaskStatus::complete;
        }, set_bc, sc1);

        // Update refinement
        if (pmesh->adaptive) {
            auto tag_refine = tl.AddTask<BlockTask>([](MeshBlock *pmb) {
                pmb->pmr->CheckRefinementCondition();
                return TaskStatus::complete;
            }, set_bc, pmb);
        }

        // Purge stages
        auto purge_stages = tl.AddTask<BlockTask>([](MeshBlock *pmb) {
            pmb->real_containers.PurgeNonBase();
            return TaskStatus::complete;
        }, set_bc, pmb);
    }
    return tl;
}