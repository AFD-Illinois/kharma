/*
 * HARM driver-specific things -- i.e. call the GRMHD physics module in
 * the correct LLF steps we know and love ~equiv. to step.c
 */

#include <iostream>

#include "parthenon/parthenon.hpp"

#include "decs.hpp"

#include "bondi.hpp"
#include "boundaries.hpp"
#include "containers.hpp"
#include "fixup.hpp"
#include "grmhd.hpp"
#include "harm.hpp"

// Any restart
#include "iharm_restart.hpp"

// Parthenon requires we override certain things. TODO move?
namespace parthenon {

Properties_t ParthenonManager::ProcessProperties(std::unique_ptr<ParameterInput>& pin)
{
    // TODO actually use this?  Just globals, basically, maybe useful for debug flags etc.
    Properties_t properties;

    // Mostly this function is where I've chosen to mess with all Parthenon's parameters before
    // handing them over.  This includes reading restarts, setting native boundaries from KS, etc.

    // If we're restarting, read the restart file for a bunch of parameters
    std::string prob = pin->GetString("parthenon/job", "problem_id");
    if (prob == "iharm_restart") {
        ReadIharmRestartHeader(pin->GetString("iharm_restart", "fname"), pin);
    }

    // TODO somehow only parse the coordinate system once, so we can know exactly whether we're spherical/modified
    // So far every non-null transform is exp(x1) but who knows
    std::string cb = pin->GetString("coordinates", "base");
    std::string ctf = pin->GetString("coordinates", "transform");
    if (ctf != "null") {
        // Set Rin such that we have 5 zones completely inside the event horizon
        // If xeh = log(Rhor), xin = log(Rin), and xout = log(Rout),
        // then we want xeh = xin + 5.5 * (xout - xin) / N1TOT, or solving/replacing:
        int n1tot = pin->GetInteger("parthenon/mesh", "nx1");
        GReal Rout = pin->GetReal("coordinates", "r_out");
        Real a = pin->GetReal("coordinates", "a");
        GReal Rhor = 1 + sqrt(1 - a*a);
        GReal x1max = log(Rout);
        GReal x1min = (n1tot * log(Rhor) / 5.5 - x1max) / (-1. + n1tot / 5.5);
        if (x1min < 0.0) {
            throw std::invalid_argument("Not enough radial zones were specified to put 5 zones inside EH!");
        }
        //cerr << "Setting x1min: " << x1min << " x1max " << x1max << " based on BH with a=" << a << endl;
        pin->SetReal("parthenon/mesh", "x1min", x1min);
        pin->SetReal("parthenon/mesh", "x1max", x1max);
    }
    // Assumption: if we're in a spherical system...
    if (cb == "spherical_ks" || cb == "ks" || cb == "spherical_bl" || cb == "bl" || cb == "spherical_minkowski") {
        // ...then we definitely want spherical boundary conditions
        // TODO only set all this if it isn't already
        pin->SetString("parthenon/mesh", "ix1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ox1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ix2_bc", "reflecting");
        pin->SetString("parthenon/mesh", "ox2_bc", "reflecting");
        pin->SetString("parthenon/mesh", "ix3_bc", "periodic");
        pin->SetString("parthenon/mesh", "ox3_bc", "periodic");

        // We also know the bounds for most transforms in spherical.  Set them.
        if (ctf == "none") {
            pin->SetReal("parthenon/mesh", "x2min", 0.0);
            pin->SetReal("parthenon/mesh", "x2max", M_PI);
            pin->SetReal("parthenon/mesh", "x3min", 0.0);
            pin->SetReal("parthenon/mesh", "x3max", 2*M_PI);
        } else if (ctf == "modified" || ctf == "mks" || ctf == "funky" || ctf == "fmks") {
            pin->SetReal("parthenon/mesh", "x2min", 0.0);
            pin->SetReal("parthenon/mesh", "x2max", 1.0);
            pin->SetReal("parthenon/mesh", "x3min", 0.0);
            pin->SetReal("parthenon/mesh", "x3max", 2*M_PI);
        } // TODO any other transforms/systems
    }

    return properties;
}

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput>& pin)
{
    Packages_t packages;

    // Turn off GRMHD only if set to false in input file
    bool do_grmhd = pin->GetOrAddBoolean("Packages", "GRMHD", true);
    bool do_grhd = pin->GetOrAddBoolean("Packages", "GRHD", false);
    bool do_electrons = pin->GetOrAddBoolean("Packages", "howes_electrons", false);

    // enable other packages as needed
    bool do_scalars = pin->GetOrAddBoolean("Packages", "scalars", false);

    // Just one base package: integrated B-fields, or not.
    if (do_grmhd) {
        packages["GRMHD"] = GRMHD::Initialize(pin.get());
    } else if (do_grhd) {

    }

    // Scalars can be added 
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
    // TODO: Figure out when the step beginning and end are both accessible, for calculating jcon
    if (stage == 1) {
        auto& base = pmb->real_containers.Get();
        pmb->real_containers.Add("dUdt", base);
        for (int i=1; i<integrator->nstages; i++)
            pmb->real_containers.Add(stage_name[i], base);
    }

    // pull out the container we'll use to get fluxes and/or compute RHSs
    auto& sc0  = pmb->real_containers.Get(stage_name[stage-1]);
    // pull out a container we'll use to store dU/dt.
    auto& dudt = pmb->real_containers.Get("dUdt");
    // pull out the container that will hold the updated state
    auto& sc1  = pmb->real_containers.Get(stage_name[stage]);

    // TODO what does this do exactly?
    auto t_start_recv = tl.AddTask(&Container<Real>::StartReceiving, sc1.get(), none,
                                   BoundaryCommSubset::all);

    // Calculate the LLF fluxes in each direction
    // This uses the primitives (P) to calculate fluxes to update the conserved variables (U)
    // Hence the two should reflect *exactly* the same fluid state, hereafter "lockstep"

    // Flux calc can be done with separate kernels called from CalculateFlux,
    // or a merged kernel in ReconAndFlux.  TODO actual concurrency
    // Assumes valid P, spits valid U fluxes
    TaskID t_calculate_flux1, t_calculate_flux2, t_calculate_flux3;
    if (pmb->packages["GRMHD"]->Param<bool>("merge_recon")) {
        auto t_calculate_flux1 = tl.AddTask(GRMHD::ReconAndFlux, t_start_recv, sc0, X1DIR);
        auto t_calculate_flux2 = tl.AddTask(GRMHD::ReconAndFlux, t_start_recv, sc0, X2DIR);
        auto t_calculate_flux3 = tl.AddTask(GRMHD::ReconAndFlux, t_start_recv, sc0, X3DIR);
    } else {
        auto t_calculate_flux1 = tl.AddTask(GRMHD::CalculateFlux, t_start_recv, sc0, X1DIR);
        auto t_calculate_flux2 = tl.AddTask(GRMHD::CalculateFlux, t_start_recv, sc0, X2DIR);
        auto t_calculate_flux3 = tl.AddTask(GRMHD::CalculateFlux, t_start_recv, sc0, X3DIR);
    }
    auto t_calculate_flux = t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;

    // TODO add these sensibly for AMR/SMR runs (below Fix and/or CT?)
//   auto t_send_flux =
//       tl.AddTask(&Container<Real>::SendFluxCorrection, sc0.get(), t_calculate_flux);
//   auto t_recv_flux =
//       tl.AddTask(&Container<Real>::ReceiveFluxCorrection, sc0.get(), t_calculate_flux);

    // These operate totally on fluxes
    auto t_fix_flux = tl.AddTask(FixFlux, t_calculate_flux, sc0);
    auto t_flux_ct = tl.AddTask(GRMHD::FluxCT, t_fix_flux, sc0);

    // Apply fluxes to create a single update dU/dt
    auto t_flux_divergence = tl.AddTask(Update::FluxDivergence, t_flux_ct, sc0, dudt);
    auto t_source_term = tl.AddTask(GRMHD::AddSourceTerm, t_flux_divergence, sc0, dudt);
    // Apply dU/dt to the stage's initial state sc0 to obtain the stage final state sc1
    // Note this *only fills U* of sc1, so sc1 is out of lockstep
    auto t_update_container = tl.AddTask(UpdateContainer, t_source_term, pmb, stage, stage_name, integrator);

    // Update ghost cells.  Only performed on U of sc1
  auto t_send =
      tl.AddTask(&Container<Real>::SendBoundaryBuffers, sc1.get(), t_update_container);
  auto t_recv = tl.AddTask(&Container<Real>::ReceiveBoundaryBuffers, sc1.get(), t_send);
  auto t_fill_from_bufs = tl.AddTask(&Container<Real>::SetBoundaries, sc1.get(), t_recv);
  auto t_clear_comm_flags = tl.AddTask(&Container<Real>::ClearBoundary, sc1.get(),
                                     t_fill_from_bufs, BoundaryCommSubset::all);

    // TODO add sensibly for AMR runs
    // auto t_prolong_bound = tl.AddTask([](MeshBlock *pmb) {
    //     pmb->pbval->ProlongateBoundaries(0.0, 0.0);
    //     return TaskStatus::complete;
    // }, t_fill_from_bufs, pmb);

    // Set physical boundaries
    // ApplyCustomBoundaries is a catch-all for things HARM needs done:
    // Inflow checks, renormalizations, Bondi outer boundary
    auto t_set_parthenon_bc = tl.AddTask(parthenon::ApplyBoundaryConditions,
                                            t_fill_from_bufs, sc1);
    auto t_set_custom_bc = tl.AddTask(ApplyCustomBoundaries, t_set_parthenon_bc, sc1);

    // Fill primitives, bringing U and P back into lockstep
    auto t_fill_derived = tl.AddTask(parthenon::FillDerivedVariables::FillDerived,
                                        t_set_custom_bc, sc1);

    // estimate next time step
    if (stage == integrator->nstages) {
        auto new_dt = tl.AddTask(
            [](std::shared_ptr<Container<Real>> &rc) {
                MeshBlock *pmb = rc->pmy_block;
                pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
                return TaskStatus::complete;
            }, t_fill_derived, sc1);

        // Update refinement
        if (pmesh->adaptive) {
            auto tag_refine = tl.AddTask([](MeshBlock *pmb) {
                pmb->pmr->CheckRefinementCondition();
                return TaskStatus::complete;
            }, t_fill_derived, pmb);
        }
    }
    return tl;
}