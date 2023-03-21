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
#include "electrons.hpp"
#include "grmhd.hpp"
#include "wind.hpp"
// Other headers
#include "boundaries.hpp"
#include "debug.hpp"
#include "flux.hpp"
#include "resize_restart.hpp"
#include "implicit.hpp"

#include <parthenon/parthenon.hpp>
#include <interface/update.hpp>
#include <amr_criteria/refinement_package.hpp>

TaskCollection KHARMADriver::MakeTaskCollection(BlockList_t &blocks, int stage)
{
    std::string driver_type = blocks[0]->packages.Get("Driver")->Param<std::string>("type");
    if (driver_type == "imex") {
        return MakeImExTaskCollection(blocks, stage);
    } else if (driver_type == "simple") {
        return MakeSimpleTaskCollection(blocks, stage);
    } else {
        return MakeDefaultTaskCollection(blocks, stage);
    }
}

TaskCollection KHARMADriver::MakeDefaultTaskCollection(BlockList_t &blocks, int stage)
{
    Flag("Generating default task collection");
    // Reminder that this list is created BEFORE any of the list contents are run!
    // Prints or function calls here will likely not do what you want: instead, add to the list by calling tl.AddTask()

    // TaskCollections are a collection of TaskRegions.
    // Each TaskRegion can operate on eash meshblock separately, i.e. one MeshBlockData object (slower),
    // or on a collection of MeshBlock objects called the MeshData
    TaskCollection tc;
    const TaskID t_none(0);

    // Which packages we load affects which tasks we'll add to the list
    auto& pkgs = blocks[0]->packages.AllPackages();
    auto& driver_pkg   = pkgs.at("Driver")->AllParams();
    const bool use_b_cleanup = pkgs.count("B_Cleanup");
    const bool use_electrons = pkgs.count("Electrons");
    const bool use_jcon = pkgs.count("Current");

    // Allocate the fluid states ("containers") we need for each block
    for (auto& pmb : blocks) {
        // first make other useful containers
        auto &base = pmb->meshblock_data.Get();
        if (stage == 1) {
            pmb->meshblock_data.Add("dUdt", base);
            for (int i = 1; i < integrator->nstages; i++)
                pmb->meshblock_data.Add(integrator->stage_name[i], base);
            
            if (use_jcon) {
                // At the end of the step, updating "mbd_sub_step_final" updates the base
                // So we have to keep a copy at the beginning to calculate jcon
                pmb->meshblock_data.Add("preserve", base);
            }
        }
    }

    //auto t_heating_test = tl.AddTask(t_none, Electrons::ApplyHeating, base.get());

    // Big packed region: get and apply new fluxes on all the zones we control
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = single_tasklist_per_pack_region[i];
        // Container names: 
        // '_full_step_init' refers to the fluid state at the start of the full time step (Si in iharm3d)
        // '_sub_step_init' refers to the fluid state at the start of the sub step (Ss in iharm3d)
        // '_sub_step_final' refers to the fluid state at the end of the sub step (Sf in iharm3d)
        // '_flux_src' refers to the mesh object corresponding to -divF + S
        auto &md_full_step_init = pmesh->mesh_data.GetOrAdd("base", i);
        auto &md_sub_step_init  = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage - 1], i);
        auto &md_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        auto &md_flux_src       = pmesh->mesh_data.GetOrAdd("dUdt", i);

        // Start receiving flux corrections and ghost cells
        namespace cb = parthenon::cell_centered_bvars;
        auto t_start_recv_bound = tl.AddTask(t_none, cb::StartReceiveBoundBufs<parthenon::BoundaryType::any>, md_sub_step_final);
        auto t_start_recv_flux = t_start_recv_bound;
        if (pmesh->multilevel)
            t_start_recv_flux = tl.AddTask(t_none, cb::StartReceiveFluxCorrections, md_sub_step_init);

        // Calculate the flux of each variable through each face
        // This reconstructs the primitives (P) at faces and uses them to calculate fluxes
        // of the conserved variables (U) through each face.
        const KReconstruction::Type& recon = driver_pkg.Get<KReconstruction::Type>("recon");
        auto t_fluxes = KHARMADriver::AddFluxCalculations(t_start_recv_bound, tl, recon, md_sub_step_init.get());

        // If we're in AMR, correct fluxes from neighbors
        auto t_flux_bounds = t_fluxes;
        if (pmesh->multilevel) {
            tl.AddTask(t_fluxes, cb::LoadAndSendFluxCorrections, md_sub_step_init);
            auto t_recv_flux = tl.AddTask(t_fluxes, cb::ReceiveFluxCorrections, md_sub_step_init);
            t_flux_bounds = tl.AddTask(t_recv_flux, cb::SetFluxCorrections, md_sub_step_init);
        }

        // Any package modifications to the fluxes.  e.g.:
        // 1. CT calculations for B field transport
        // 2. Zero fluxes through poles
        // etc 
        auto t_fix_flux = tl.AddTask(t_flux_bounds, Packages::FixFlux, md_sub_step_init.get());

        // Apply the fluxes to calculate a change in cell-centered values "md_flux_src"
        auto t_flux_div = tl.AddTask(t_fix_flux, Update::FluxDivergence<MeshData<Real>>, md_sub_step_init.get(), md_flux_src.get());

        // Add any source terms: geometric \Gamma * T, wind, damping, etc etc
        auto t_sources = tl.AddTask(t_flux_div, Packages::AddSource, md_sub_step_init.get(), md_flux_src.get());

        // Perform the update using the source term
        // Add any proportion of the step start required by the integrator (e.g., RK2)
        auto t_avg_data = tl.AddTask(t_sources, Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
                                    std::vector<MetadataFlag>({Metadata::Independent}),
                                    md_sub_step_init.get(), md_full_step_init.get(),
                                    integrator->gam0[stage-1], integrator->gam1[stage-1],
                                    md_sub_step_final.get());
        // apply du/dt to the result
        auto t_update = tl.AddTask(t_sources, Update::WeightedSumData<std::vector<MetadataFlag>, MeshData<Real>>,
                                    std::vector<MetadataFlag>({Metadata::Independent}),
                                    md_sub_step_final.get(), md_flux_src.get(),
                                    1.0, integrator->beta[stage-1] * integrator->dt,
                                    md_sub_step_final.get());

        // UtoP needs a guess in order to converge, so we copy in sc0
        // (but only the fluid primitives!)  Copying and syncing ensures that solves of the same zone
        // on adjacent ranks are seeded with the same value, which keeps them (more) similar
        auto t_copy_prims = t_update;
        if (integrator->nstages > 1) {
            t_copy_prims = tl.AddTask(t_none, Copy, std::vector<MetadataFlag>({Metadata::GetUserFlag("HD"), Metadata::GetUserFlag("Primitive")}),
                                                md_sub_step_init.get(), md_sub_step_final.get());
        }

        KHARMADriver::AddMPIBoundarySync(t_copy_prims, tl, md_sub_step_final);
    }

    // Smaller meshblock region.  This gets touchy because we want to keep ghost zones updated,
    // so very commented
    TaskRegion &async_region = tc.AddRegion(blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
        auto &pmb = blocks[i];
        auto &tl = async_region[i];
        //auto &base = pmb->meshblock_data.Get();
        auto &mbd_sub_step_init = pmb->meshblock_data.Get(integrator->stage_name[stage-1]);
        auto &mbd_sub_step_final = pmb->meshblock_data.Get(integrator->stage_name[stage]);

        // At this point, we've sync'd all internal boundaries using the conserved
        // variables. The physical boundaries (pole, inner/outer) are trickier,
        // since they must be applied to the primitive variables rho,u,u1,u2,u3
        // but should apply to conserved forms of everything else.

        // This call fills the fluid primitive values in all physical zones, that is, including MPI boundaries but
        // not the physical boundaries (which haven't been filled yet!)
        // This relies on the primitives being calculated identically in MPI boundaries, vs their corresponding
        // physical zones in the adjacent mesh block.  To ensure this, we seed the solver with the same values
        // in each case, by synchronizing them along with the conserved values above.
        auto t_utop = tl.AddTask(t_none, Packages::BlockUtoP, mbd_sub_step_final.get(), IndexDomain::entire, false);
        // As soon as we have primitive variables, apply floors
        auto t_floors = tl.AddTask(t_utop, Packages::BlockApplyFloors, mbd_sub_step_final.get(), IndexDomain::entire);

        // Then, fix any inversions which failed. Fixups average the adjacent zones, so we want to work from
        // post-floor data. Floors are re-applied after fixups.
        auto t_fix_p = tl.AddTask(t_floors, Inverter::FixUtoP, mbd_sub_step_final.get());

        // Domain (non-internal) boundary conditions:
        // This is a parthenon call, but in spherical coordinates it will call the KHARMA functions in
        // boundaries.cpp, which apply physical boundary conditions based on the primitive variables of GRHD,
        // and based on the conserved forms for everything else.  Note that because this is called *after*
        // UtoP (since it needs bulk fluid primitives to apply GRMHD boundaries), this function
        // must call UtoP *again* (for everything except the GRHD variables) to fill P in the ghost zones.
        // This is why KHARMA packages need to implement their UtoP functions in the form
        // UtoP(rc, domain, coarse): so that they can be run over just the boundary domains here.
        auto t_set_bc = tl.AddTask(t_fix_p, parthenon::ApplyBoundaryConditions, mbd_sub_step_final);

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
            t_prim_source = tl.AddTask(t_set_bc, Packages::BlockApplyPrimSource, mbd_sub_step_final.get());
        }
        // Electron heating goes where it does in HARMDriver, for the same reasons
        auto t_heat_electrons = t_prim_source;
        if (use_electrons) {
            t_heat_electrons = tl.AddTask(t_prim_source, Electrons::ApplyElectronHeating,
                                          mbd_sub_step_init.get(), mbd_sub_step_final.get());
        }

        auto t_step_done = t_heat_electrons;

        // Estimate next time step based on ctop
        if (stage == integrator->nstages) {
            auto t_new_dt =
                tl.AddTask(t_step_done, Update::EstimateTimestep<MeshBlockData<Real>>, mbd_sub_step_final.get());

            // Update refinement
            if (pmesh->adaptive) {
                auto tag_refine = tl.AddTask(
                    t_step_done, parthenon::Refinement::Tag<MeshBlockData<Real>>, mbd_sub_step_final.get());
            }
        }
    }

    // Second boundary sync:
    // ensure that primitive variables in ghost zones are *exactly*
    // identical to their physical counterparts, now that they have been
    // modified on each rank.
    const auto &two_sync = pkgs.at("Driver")->Param<bool>("two_sync");
    if (two_sync) KHARMADriver::AddFullSyncRegion(pmesh, tc, stage);

    Flag("Generated");
    return tc;
}
