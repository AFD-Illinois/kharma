/* 
 *  File: kharma_step.cpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *  
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "kharma_driver.hpp"

// TODO CLEAN
//Packages
#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "b_cleanup.hpp"
#include "b_ct.hpp"
#include "electrons.hpp"
#include "grmhd.hpp"
#include "wind.hpp"
// Other headers
#include "boundaries.hpp"
#include "flux.hpp"
#include "kharma.hpp"
#include "implicit.hpp"
#include "resize_restart.hpp"

#include <parthenon/parthenon.hpp>
#include <interface/update.hpp>
#include <amr_criteria/refinement_package.hpp>

TaskCollection KHARMADriver::MakeTaskCollection(BlockList_t &blocks, int stage)
{
    DriverType driver_type = blocks[0]->packages.Get("Driver")->Param<DriverType>("type");
    Flag("MakeTaskCollection");
    TaskCollection tc;
    switch (driver_type) {
    case DriverType::kharma:
        tc = MakeDefaultTaskCollection(blocks, stage);
        break;
    case DriverType::imex:
        tc = MakeImExTaskCollection(blocks, stage);
        break;
    case DriverType::simple:
        tc = MakeSimpleTaskCollection(blocks, stage);
        break;
    }
    EndFlag();
    return tc;
}

TaskCollection KHARMADriver::MakeDefaultTaskCollection(BlockList_t &blocks, int stage)
{
    // Reminder that this list is created BEFORE any of the list contents are run!
    // Prints or function calls here will likely not do what you want: instead, add to the list by calling tl.AddTask()

    // TaskCollections are a collection of TaskRegions.
    // Each TaskRegion can operate on eash meshblock separately, i.e. one MeshBlockData object (slower),
    // or on a collection of MeshBlock objects called the MeshData
    TaskCollection tc;
    const TaskID t_none(0);

    // Which packages we load affects which tasks we'll add to the list
    auto& pkgs = pmesh->packages.AllPackages();
    auto& driver_pkg   = pkgs.at("Driver")->AllParams();
    const bool use_b_cleanup = pkgs.count("B_Cleanup");
    const bool use_b_ct = pkgs.count("B_CT");
    const bool use_electrons = pkgs.count("Electrons");
    const bool use_jcon = pkgs.count("Current");

    // Allocate/copy the things we need
    // TODO these can now be reduced by including the var lists/flags which actually need to be allocated
    // TODO except the Copy they can be run on step 1 only
    if (stage == 1) {
        auto &base = pmesh->mesh_data.Get();
        // Fluxes
        pmesh->mesh_data.Add("dUdt");
        for (int i = 1; i < integrator->nstages; i++)
            pmesh->mesh_data.Add(integrator->stage_name[i]);
        // Preserve state for time derivatives if we need to output current
        if (use_jcon) {
            pmesh->mesh_data.Add("preserve");
            // Above only copies on allocate -- ensure we copy every step
            Copy<MeshData<Real>>({Metadata::Cell}, base.get(), pmesh->mesh_data.Get("preserve").get());
        }
    }

    Flag("MakeTaskCollection::fluxes");

    static std::vector<std::string> sync_vars;
    if (sync_vars.size() == 0) {
        // Build the universe of variables to let Parthenon see when exchanging boundaries.
        // This is built to exclude incidental variables like B field initialization stuff, EMFs, etc.
        // "Boundaries" packs in buffers e.g. Dirichlet boundaries
        using FC = Metadata::FlagCollection;
        auto sync_flags = FC({Metadata::GetUserFlag("Primitive"), Metadata::Conserved,
                              Metadata::Face, Metadata::GetUserFlag("Boundaries")}, true);
        sync_vars = KHARMA::GetVariableNames(&(pmesh->packages), sync_flags);
    }

    // Flux region: calculate and apply fluxes to update conserved values
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &flux_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = flux_region[i];
        // Container names: 
        // '_full_step_init' refers to the fluid state at the start of the full time step (Si in iharm3d)
        // '_sub_step_init' refers to the fluid state at the start of the sub step (Ss in iharm3d)
        // '_sub_step_final' refers to the fluid state at the end of the sub step (Sf in iharm3d)
        // '_flux_src' refers to the mesh object corresponding to -divF + S
        auto &md_full_step_init = pmesh->mesh_data.GetOrAdd("base", i);
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage - 1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_flux_src       = pmesh->mesh_data.GetOrAdd("dUdt", i);
        // TODO this doesn't work still for some reason, even if the shallow copy has all variables
        auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage]+std::to_string(i), md_sub_step_final, sync_vars);

        // Start receiving flux corrections and ghost cells
        auto t_start_recv_bound = tl.AddTask(t_none, parthenon::StartReceiveBoundBufs<parthenon::BoundaryType::any>, md_sync);
        auto t_start_recv_flux = t_start_recv_bound;
        if (pmesh->multilevel || use_b_ct)
            t_start_recv_flux = tl.AddTask(t_none, parthenon::StartReceiveFluxCorrections, md_sub_step_init);

        // Calculate the flux of each variable through each face
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U) through each face.
        const KReconstruction::Type& recon = driver_pkg.Get<KReconstruction::Type>("recon");
        auto t_fluxes = KHARMADriver::AddFluxCalculations(t_start_recv_flux, tl, recon, md_sub_step_init.get());

        // Any package modifications to the fluxes.  e.g.:
        // 1. Flux-CT calculations for B field transport
        // 2. Zero fluxes through poles
        // etc
        auto t_fix_flux = tl.AddTask(t_fluxes, Packages::FixFlux, md_sub_step_init.get());

        // If we're in AMR, correct fluxes from neighbors
        auto t_flux_bounds = t_fix_flux;
        if (pmesh->multilevel || use_b_ct) {
            auto t_emf = t_flux_bounds;
            if (use_b_ct) {
                // Pull out a container of only EMF to synchronize
                auto &md_emf_only = pmesh->mesh_data.AddShallow("EMF", std::vector<std::string>{"B_CT.emf"}); // TODO this gets weird if we partition
                auto t_emf_local = tl.AddTask(t_flux_bounds, B_CT::CalculateEMF, md_sub_step_init.get());
                t_emf = KHARMADriver::AddBoundarySync(t_emf_local, tl, md_emf_only);
            }
            auto t_load_send_flux = tl.AddTask(t_emf, parthenon::LoadAndSendFluxCorrections, md_sub_step_init);
            auto t_recv_flux = tl.AddTask(t_load_send_flux, parthenon::ReceiveFluxCorrections, md_sub_step_init);
            t_flux_bounds = tl.AddTask(t_recv_flux, parthenon::SetFluxCorrections, md_sub_step_init);
        }

        // Apply the fluxes to calculate a change in cell-centered values "md_flux_src"
        auto t_flux_div = tl.AddTask(t_flux_bounds, Update::FluxDivergence<MeshData<Real>>, md_sub_step_init.get(), md_flux_src.get());

        // Add any source terms: geometric \Gamma * T, wind, damping, etc etc
        // Also where CT sets the change in face fields
        auto t_sources = tl.AddTask(t_flux_div, Packages::AddSource, md_sub_step_init.get(), md_flux_src.get());

        // Perform the update using the source term
        // Add any proportion of the step start required by the integrator (e.g., RK2)
        // TODO splitting this is stupid, dig into Parthenon & fix
        auto t_avg_data_c = tl.AddTask(t_sources, Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
                                    std::vector<MetadataFlag>({Metadata::Independent, Metadata::Cell}),
                                    md_sub_step_init.get(), md_full_step_init.get(),
                                    integrator->gam0[stage-1], integrator->gam1[stage-1],
                                    md_sub_step_final.get());
        auto t_avg_data = t_avg_data_c;
        if (use_b_ct) {
            t_avg_data = tl.AddTask(t_avg_data_c, WeightedSumDataFace,
                                    std::vector<MetadataFlag>({Metadata::Independent, Metadata::Face}),
                                    md_sub_step_init.get(), md_full_step_init.get(),
                                    integrator->gam0[stage-1], integrator->gam1[stage-1],
                                    md_sub_step_final.get());
        }
        // apply du/dt to the result
        auto t_update_c = tl.AddTask(t_avg_data, Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
                                    std::vector<MetadataFlag>({Metadata::Independent, Metadata::Cell}),
                                    md_sub_step_final.get(), md_flux_src.get(),
                                    1.0, integrator->beta[stage-1] * integrator->dt,
                                    md_sub_step_final.get());
        auto t_update = t_update_c;
        if (use_b_ct) {
            t_update = tl.AddTask(t_update_c, WeightedSumDataFace,
                                    std::vector<MetadataFlag>({Metadata::Independent, Metadata::Face}),
                                    md_sub_step_final.get(), md_flux_src.get(),
                                    1.0, integrator->beta[stage-1] * integrator->dt,
                                    md_sub_step_final.get());
        }

        // UtoP needs a guess in order to converge, so we copy in sc0
        // (but only the fluid primitives!)  Copying and syncing ensures that solves of the same zone
        // on adjacent ranks are seeded with the same value, which keeps them (more) similar
        auto t_copy_prims = t_update;
        if (integrator->nstages > 1) {
            t_copy_prims = tl.AddTask(t_none, Copy<MeshData<Real>>, std::vector<MetadataFlag>({Metadata::GetUserFlag("HD"), Metadata::GetUserFlag("Primitive")}),
                                                md_sub_step_init.get(), md_sub_step_final.get());
        }

        KHARMADriver::AddBoundarySync(t_copy_prims | t_update, tl, md_sync);
    }

    EndFlag();
    Flag("MakeTaskCollection::fixes");

    // Fix Region: prims/cons sync, floors, fixes, boundary conditions which need primitives
    TaskRegion &fix_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = fix_region[i];
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage-1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage]+std::to_string(i), md_sub_step_final, sync_vars);

        // At this point, we've sync'd all internal boundaries using the conserved
        // variables. The physical boundaries (pole, inner/outer) are trickier,
        // since they must be applied to the primitive variables rho,u,u1,u2,u3
        // but should apply to conserved forms of everything else.

        // This call fills the fluid primitive values in all physical zones, that is, including MPI boundaries but
        // not the physical boundaries (which haven't been filled yet!)
        // This relies on the primitives being calculated identically in MPI boundaries, vs their corresponding
        // physical zones in the adjacent mesh block.  To ensure this, we seed the solver with the same values
        // in each case, by synchronizing them along with the conserved values above.
        auto t_utop = tl.AddTask(t_none, Packages::MeshUtoP, md_sub_step_final.get(), IndexDomain::entire, false);
        // As soon as we have primitive variables, apply floors
        auto t_floors = tl.AddTask(t_utop, Packages::MeshApplyFloors, md_sub_step_final.get(), IndexDomain::entire);

        // Then, fix any inversions which failed. Fixups average the adjacent zones, so we want to work from
        // post-floor data. Floors are re-applied after fixups.
        auto t_fix_p = tl.AddTask(t_floors, Inverter::MeshFixUtoP, md_sub_step_final.get());

        // Domain (non-internal) boundary conditions:
        // This is a parthenon call, but in spherical coordinates it will call the KHARMA functions in
        // boundaries.cpp, which apply physical boundary conditions based on the primitive variables of GRHD,
        // and based on the conserved forms for everything else.  Note that because this is called *after*
        // UtoP (since it needs bulk fluid primitives to apply GRMHD boundaries), this function
        // must call UtoP *again* (for everything except the GRHD variables) to fill P in the ghost zones.
        // This is why KHARMA packages need to implement their UtoP functions in the form
        // UtoP(rc, domain, coarse): so that they can be run over just the boundary domains here.
        auto t_set_bc = tl.AddTask(t_fix_p, parthenon::ApplyBoundaryConditionsOnCoarseOrFineMD, md_sync, false);

        // Add primitive-variable source terms:
        // In order to calculate dissipation, we must know the entropy at the beginning and end of the substep,
        // and this must be calculated from the fluid primitive variables rho,u (and for stability, obey floors!).
        // Only now do we have the end-of-step primitives in consistent, corrected forms.
        // Luckily, ApplyElectronHeating should *not* need another synchronization of the ghost zones, as it is applied to
        // all zones and has a stencil of only one zone.  As with UtoP, this trusts that evaluations 
        // of the same zone match between MeshBlocks.

        // Any package- (likely, problem-) specific source terms which must be applied to primitive variables
        // Apply these only after the final step so they're operator-split
        auto t_prim_source = t_set_bc;
        if (stage == integrator->nstages) {
            t_prim_source = tl.AddTask(t_set_bc, Packages::MeshApplyPrimSource, md_sub_step_final.get());
        }
        // Electron heating goes where it does in HARMDriver, for the same reasons
        auto t_heat_electrons = t_prim_source;
        if (use_electrons) {
            t_heat_electrons = tl.AddTask(t_prim_source, Electrons::MeshApplyElectronHeating,
                                          md_sub_step_init.get(), md_sub_step_final.get());
        }

        // Make sure *all* conserved vars are synchronized at step end
        auto t_ptou = tl.AddTask(t_heat_electrons, Flux::MeshPtoU, md_sub_step_final.get(), IndexDomain::entire, false);

        auto t_step_done = t_ptou;

        // Estimate next time step based on ctop
        if (stage == integrator->nstages) {
            auto t_new_dt =
                tl.AddTask(t_step_done, Update::EstimateTimestep<MeshData<Real>>, md_sub_step_final.get());

            // Update refinement
            if (pmesh->adaptive) {
                auto tag_refine = tl.AddTask(
                    t_step_done, parthenon::Refinement::Tag<MeshData<Real>>, md_sub_step_final.get());
            }
        }
    }

    EndFlag();
    Flag("MakeTaskCollection::extras");

    // B Field cleanup: this is a separate solve so it's split out
    // It's also really slow when enabled so we don't care too much about limiting regions, etc.
    if (use_b_cleanup && (stage == integrator->nstages) && B_Cleanup::CleanupThisStep(pmesh, tm.ncycle)) {
        TaskRegion &cleanup_region = tc.AddRegion(1);
        auto &tl = cleanup_region[0];
        auto &md_sub_step_final = pmesh->mesh_data.Get(integrator->stage_name[stage]);
        tl.AddTask(t_none, B_Cleanup::CleanupDivergence, md_sub_step_final);
    }

    // TODO TODO make faster for large num_partitions, also this should be shared whole between drivers
    // Second boundary sync:
    // ensure that primitive variables in ghost zones are *exactly*
    // identical to their physical counterparts, now that they have been
    // modified on each rank.
    const auto &two_sync = pkgs.at("Driver")->Param<bool>("two_sync");
    if (two_sync) {
        for (int i = 0; i < num_partitions; i++) {
            auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
            auto &md_sync = pmesh->mesh_data.AddShallow("sync"+integrator->stage_name[stage]+std::to_string(i), md_sub_step_final, sync_vars);
            KHARMADriver::AddFullSyncRegion(tc, md_sync);
        }
    }

    EndFlag();

    return tc;
}
