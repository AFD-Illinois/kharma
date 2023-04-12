
/* 
 *  File: kharma_driver.cpp
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

#include "boundaries.hpp"
#include "flux.hpp"
// GetFlux
#include "get_flux.hpp"

std::shared_ptr<KHARMAPackage> KHARMADriver::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    Flag("Initializing KHARMA Driver");
    // This function builds and returns a "KHARMAPackage" object, which is a light
    // superset of Parthenon's "StateDescriptor" class for packages.
    // The most important part of this object is a member of type "Params",
    // which acts more or less like a Python dictionary:
    // it puts values into a map of names->objects, where "objects" are usually
    // floats, strings, and ints, but can be arbitrary classes.
    // This "dictionary" is mostly immutable, and should always be treated as immutable,
    // except in the "Globals" package.
    auto pkg = std::make_shared<KHARMAPackage>("Driver");
    Params &params = pkg->AllParams();

    // Driver options
    // The two current drivers are "kharma" or "imex", with the former being the usual KHARMA
    // driver (formerly HARM driver), and the latter supporting implicit stepping of some or all variables
    // Mostly, packages should react to e.g. the "sync_prims" option rather than the driver name
    bool do_emhd = pin->GetOrAddBoolean("emhd", "on", false);
    std::string driver_type = pin->GetOrAddString("driver", "type", (do_emhd) ? "imex" : "kharma");
    if (driver_type == "harm") driver_type = "kharma"; // TODO enum rather than strings?
    params.Add("type", driver_type);

    // Record whether we marked the prims or cons as "FillGhost." This also translates to whether we consider
    // primitive or conserved state to be the ground truth when updating values in a step.
    // Currently "imex" and "simple" drivers both sync primitive vars
    bool sync_prims = !(driver_type == "kharma");
    params.Add("sync_prims", sync_prims);

    // Synchronize boundary variables twice. Ensures KHARMA is agnostic to the breakdown
    // of meshblocks, at the cost of twice the MPI overhead, for potentially worse strong scaling.
    // On by default, disable only after testing that, e.g., divB meets your requirements
    bool two_sync = pin->GetOrAddBoolean("driver", "two_sync", true);
    params.Add("two_sync", two_sync);

    // Don't even error on this. Use LLF unless the user is very clear otherwise.
    std::string flux = pin->GetOrAddString("driver", "flux", "llf");
    params.Add("use_hlle", (flux == "hlle"));

    // Reconstruction scheme: plm, weno5, ppm...
    // Allow an old parameter location
    std::string recon = pin->GetOrAddString("driver", "reconstruction",
                                            pin->GetOrAddString("GRMHD", "reconstruction", "weno5"));
    if (recon == "donor_cell") {
        params.Add("recon", KReconstruction::Type::donor_cell);
    } else if (recon == "linear_vl") {
        params.Add("recon", KReconstruction::Type::linear_vl);
    } else if (recon == "linear_mc") {
        params.Add("recon", KReconstruction::Type::linear_mc);
    } else if (recon == "weno5") {
        params.Add("recon", KReconstruction::Type::weno5);
    } else {
        std::cerr << "Reconstruction type not supported!  Supported reconstructions:" << std::endl;
        std::cerr << "donor_cell, linear_mc, linear_vl, weno5" << std::endl;
        throw std::invalid_argument("Unsupported reconstruction algorithm!");
    }

    // Field flags related to driver operation are defined outside any particular driver
    // When using the Implicit package we need to globally distinguish implicitly and explicitly-updated variables
    // All independent variables should be marked one or the other,
    // so we define the flags here to avoid loading order issues
    Metadata::AddUserFlag("Implicit");
    Metadata::AddUserFlag("Explicit");

    // Keep track of numbers of variables
    params.Add("n_explicit_vars", 0, true);
    params.Add("n_implicit_vars", 0, true);

    return pkg;
}

void KHARMADriver::AddFullSyncRegion(Mesh* pmesh, TaskCollection& tc, int stage)
{
    const TaskID t_none(0);

    // Parthenon's call for bounds is MeshBlock, it sucks
    int nblocks = pmesh->block_list.size();
    TaskRegion &async_region2 = tc.AddRegion(nblocks);
    for (int i = 0; i < nblocks; i++) {
        auto &pmb = pmesh->block_list[i];
        auto &tl  = async_region2[i];
        auto &mbd_sub_step_final = pmb->meshblock_data.Get(integrator->stage_name[stage]);
        tl.AddTask(t_none, parthenon::ApplyBoundaryConditions, mbd_sub_step_final);
    }

    // MPI boundary exchange, done over MeshData objects/partitions at once
    const int num_partitions = pmesh->DefaultNumPartitions(); // Usually 1
    TaskRegion &bound_sync = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = bound_sync[i];
        // This is a member function of KHARMADriver, so it inherits 'integrator'
        auto &mbd_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        AddMPIBoundarySync(t_none, tl, mbd_sub_step_final);
    }
}

TaskID KHARMADriver::AddMPIBoundarySync(const TaskID t_start, TaskList &tl, std::shared_ptr<MeshData<Real>> mc1)
{
    auto t_start_sync = t_start;

    if (0) { //(mc1->GetMeshPointer()->packages.Get("Driver")->Param<bool>("sync_prims")) {
        TaskID t_all_ptou[mc1->NumBlocks() * BOUNDARY_NFACES];
        TaskID t_ptou_final(0);
        int i_task = 0;
        for (int i_block = 0; i_block < mc1->NumBlocks(); i_block++) {
            auto &rc = mc1->GetBlockData(i_block);
            for (int i_bnd = 0; i_bnd < BOUNDARY_NFACES; i_bnd++) {
                if (rc->GetBlockPointer()->boundary_flag[i_bnd] == BoundaryFlag::block ||
                    rc->GetBlockPointer()->boundary_flag[i_bnd] == BoundaryFlag::periodic) {
                    t_all_ptou[i_task] = tl.AddTask(t_start, Flux::BlockPtoU_Send, rc.get(), BoundaryDomain((BoundaryFace) i_bnd), false);
                    t_ptou_final = t_ptou_final | t_all_ptou[i_task];
                    i_task++;
                }
            }
        }
        t_start_sync = t_ptou_final;
    }

    auto t_sync_done = parthenon::cell_centered_bvars::AddBoundaryExchangeTasks(t_start_sync, tl, mc1, mc1->GetMeshPointer()->multilevel);
    auto t_bounds = t_sync_done;

    // TODO(BSP) careful about how AMR interacts with below
    Kokkos::fence();

    // If we're "syncing primitive variables" but just exchanged cons.B, we need to recover the prims
    if (mc1->GetMeshPointer()->packages.Get("Driver")->Param<bool>("sync_prims")) {
        TaskID t_all_utop[mc1->NumBlocks() * BOUNDARY_NFACES];
        TaskID t_utop_final(0);
        int i_task = 0;
        for (int i_block = 0; i_block < mc1->NumBlocks(); i_block++) {
            auto &rc = mc1->GetBlockData(i_block);
            for (int i_bnd = 0; i_bnd < BOUNDARY_NFACES; i_bnd++) {
                if (rc->GetBlockPointer()->boundary_flag[i_bnd] == BoundaryFlag::block ||
                    rc->GetBlockPointer()->boundary_flag[i_bnd] == BoundaryFlag::periodic) {
                    t_all_utop[i_task] = tl.AddTask(t_sync_done, Packages::BlockUtoPExceptMHD, rc.get(), BoundaryDomain((BoundaryFace) i_bnd), false);
                    t_utop_final = t_utop_final | t_all_utop[i_task];
                    i_task++;
                }
            }
        }
        t_bounds = t_utop_final;
    }

    return t_bounds;
}

void KHARMADriver::SyncAllBounds(std::shared_ptr<MeshData<Real>> md, bool apply_domain_bounds)
{
    Flag("Syncing all bounds");
    TaskID t_none(0);

    // 1. PtoU on the interior to ensure we're up-to-date
    Flux::MeshPtoU(md.get(), IndexDomain::interior, false);

    // 2. Sync MPI bounds
    // This call syncs the primitive variables when using the ImEx driver, and cons
    //
    TaskCollection tc;
    auto tr = tc.AddRegion(1);
    AddMPIBoundarySync(t_none, tr[0], md);
    while (!tr.Execute());

    if (apply_domain_bounds) {
        // 3. Apply physical bounds block-by-block
        // TODO clean this up when ApplyBoundaryConditions gets a MeshData version
        for (auto &pmb : md->GetMeshPointer()->block_list) {
            auto& rc = pmb->meshblock_data.Get();
            // Physical boundary conditions
            parthenon::ApplyBoundaryConditions(rc);
        }
    }

    Flag("Sync'd");
}

TaskID KHARMADriver::AddFluxCalculations(TaskID& t_start, TaskList& tl, KReconstruction::Type recon, MeshData<Real> *md)
{
    // Calculate fluxes in each direction using given reconstruction
    // Must be spelled out so as to generate each templated version of GetFlux<> to be available at runtime
    // Details in flux/get_flux.hpp
    using RType = KReconstruction::Type;
    TaskID t_calculate_flux1, t_calculate_flux2, t_calculate_flux3;
    switch (recon) {
    case RType::donor_cell:
        t_calculate_flux1 = tl.AddTask(t_start, Flux::GetFlux<RType::donor_cell, X1DIR>, md);
        t_calculate_flux2 = tl.AddTask(t_start, Flux::GetFlux<RType::donor_cell, X2DIR>, md);
        t_calculate_flux3 = tl.AddTask(t_start, Flux::GetFlux<RType::donor_cell, X3DIR>, md);
        break;
    case RType::linear_mc:
        t_calculate_flux1 = tl.AddTask(t_start, Flux::GetFlux<RType::linear_mc, X1DIR>, md);
        t_calculate_flux2 = tl.AddTask(t_start, Flux::GetFlux<RType::linear_mc, X2DIR>, md);
        t_calculate_flux3 = tl.AddTask(t_start, Flux::GetFlux<RType::linear_mc, X3DIR>, md);
        break;
    case RType::linear_vl:
        t_calculate_flux1 = tl.AddTask(t_start, Flux::GetFlux<RType::linear_vl, X1DIR>, md);
        t_calculate_flux2 = tl.AddTask(t_start, Flux::GetFlux<RType::linear_vl, X2DIR>, md);
        t_calculate_flux3 = tl.AddTask(t_start, Flux::GetFlux<RType::linear_vl, X3DIR>, md);
        break;
    case RType::weno5:
        t_calculate_flux1 = tl.AddTask(t_start, Flux::GetFlux<RType::weno5, X1DIR>, md);
        t_calculate_flux2 = tl.AddTask(t_start, Flux::GetFlux<RType::weno5, X2DIR>, md);
        t_calculate_flux3 = tl.AddTask(t_start, Flux::GetFlux<RType::weno5, X3DIR>, md);
        break;
    case RType::ppm:
    case RType::mp5:
    case RType::weno5_lower_poles:
        std::cerr << "Reconstruction type not supported!  Supported reconstructions:" << std::endl
                  << "donor_cell, linear_mc, linear_vl, weno5" << std::endl;
        throw std::invalid_argument("Unsupported reconstruction algorithm!");
    }
    return t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;
}