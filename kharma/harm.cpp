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

#include "bondi.hpp"
#include "boundaries.hpp"
#include "containers.hpp"
#include "fixup.hpp"
#include "grmhd.hpp"
#include "harm.hpp"

// Parthenon requires we override certain things
namespace parthenon {

    Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput>& pin) {
        Packages_t packages;

        // Turn off GRMHD only if set to false in input file
        bool do_grmhd = pin->GetOrAddBoolean("Physics", "GRMHD", true);
        bool do_electrons = pin->GetOrAddBoolean("Physics", "howes_electrons", false);

        // enable other packages as needed
        bool do_scalars = pin->GetOrAddBoolean("Physics", "scalars", false);

        if (do_grmhd) {
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

/**
 * All the tasks which constitute advancing the fluid in a mesh block by a stage.
 * This includes calculation of necessary derived variables, reconstruction, calculation of fluxes,
 * Application of fluxes and a source term to update zones, and finally calculation of the next
 * timestep.
 * 
 * This section is heavily documented to avoid bugs.
 */
TaskList HARMDriver::MakeTaskList(MeshBlock *pmb, int stage)
{
    TaskList tl;

    TaskID none(0);
    // Parthenon separates out stages of higher-order integrators with "containers"
    // (a bundle of arrays capable of holding all Fields in the FluidState)
    // One container per stage, filled and used to update the base container over the course of the step
    // Additionally an accumulator dUdt is provided to temporarily store this stage's contribution to the RHS
    // TODO: I believe the base container is guaranteed to hold last step's product until the end of this step,
    // but need to check this.
    if (stage == 1) {
        Container<Real> &base = pmb->real_containers.Get();
        pmb->real_containers.Add("dUdt", base);
        for (int i=1; i<integrator->nstages; i++)
            pmb->real_containers.Add(stage_name[i], base);
    }

    // pull out the container we'll use to get fluxes and/or compute RHSs
    Container<Real>& sc0  = pmb->real_containers.Get(stage_name[stage-1]);
    // pull out a container we'll use to store dU/dt.
    Container<Real>& dudt = pmb->real_containers.Get("dUdt");
    // pull out the container that will hold the updated state
    Container<Real>& sc1  = pmb->real_containers.Get(stage_name[stage]);

    // TODO what does this do exactly?
    auto start_recv = AddContainerTask(tl, Container<Real>::StartReceivingTask, none, sc1);

    // Calculate the LLF fluxes in each direction
    // This uses the primitives (P) to calculate fluxes to update the conserved variables (U)
    // Hence the two should reflect *exactly* the same fluid state, which I'll term "lockstep"
    auto calculate_flux = AddContainerTask(tl, GRMHD::CalculateFluxes, start_recv, sc0);
    // TODO this will be split and Flux_CT added separately afterward

    // Exchange flux corrections due to AMR and physical boundaries
    // Note this does NOT fix vector components since we bundle primitives
    auto send_flux = AddContainerTask(tl, Container<Real>::SendFluxCorrectionTask,
                                    calculate_flux, sc0);
    auto recv_flux = AddContainerTask(tl, Container<Real>::ReceiveFluxCorrectionTask,
                                    calculate_flux, sc0);

    // TODO HARM's fix_flux for vector components

    // Apply fluxes to create a single update dU/dt
    auto flux_divergence = AddTwoContainerTask(tl, Update::FluxDivergence, recv_flux, sc0, dudt);
    auto source_term = AddTwoContainerTask(tl, GRMHD::SourceTerm, flux_divergence, sc0, dudt);
    // Apply dU/dt to the stage's initial state sc0 to obtain the stage final state sc1
    // Note this *only fills U* of sc1, so sc1 is out of lockstep
    auto update_container = AddUpdateTask(tl, pmb, stage, stage_name, integrator, UpdateContainer, source_term);

    // Update ghost cells.  Only performed on U of sc1
    auto send = AddContainerTask(tl, Container<Real>::SendBoundaryBuffersTask,
                                update_container, sc1);
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

    // Set physical boundaries
    // ApplyCustomBoundaries is only used for the Bondi test problem outer bound
    // Note custom boundaries must but need only update U.
    // TODO add physical inflow check to ApplyCustomBoundaries
    auto set_parthenon_bc = AddContainerTask(tl, parthenon::ApplyBoundaryConditions,
                                            prolong_bound, sc1);
    auto set_custom_bc = AddContainerTask(tl, ApplyCustomBoundaries, set_parthenon_bc, sc1);

    // Fill primitives, bringing U and P back into lockstep
    auto fill_derived = AddContainerTask(tl, parthenon::FillDerivedVariables::FillDerived,
                                        set_custom_bc, sc1);

    // Apply floor values to sc1.  Note that all floor operations must *preserve* lockstep
    // TODO with some attention to FillDerived, this can be eliminated.  Currently a subtle bug somewhere though
    //auto apply_floors = AddContainerTask(tl, ApplyFloors, fill_derived, sc1);

    // estimate next time step
    if (stage == integrator->nstages) {
        auto new_dt = AddContainerTask(tl, [](Container<Real>& rc) {
            MeshBlock *pmb = rc.pmy_block;
            pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
            return TaskStatus::complete;
        }, fill_derived, sc0);

        // Update refinement
        if (pmesh->adaptive) {
            auto tag_refine = tl.AddTask<BlockTask>([](MeshBlock *pmb) {
                pmb->pmr->CheckRefinementCondition();
                return TaskStatus::complete;
            }, fill_derived, pmb);
        }
    }
    return tl;
}