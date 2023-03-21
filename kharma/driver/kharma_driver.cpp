
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
    // driver, and the latter supporting implicit stepping of some or all variables
    // Mostly, packages should react to the "sync_prims" option and any option they 
    bool do_emhd = pin->GetOrAddBoolean("emhd", "on", false);
    std::string driver_type = pin->GetOrAddString("driver", "type", (do_emhd) ? "imex" : "kharma");
    params.Add("type", driver_type);

    // Record whether we marked the prims or cons as "FillGhost." This also translates to whether we consider
    // primitive or conserved state to be the ground truth when updating values in a step.
    bool sync_prims = !(driver_type == "kharma" || driver_type == "harm");
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

    // MPI boundary exchange, done over MeshData objects/partitions at once
    const int num_partitions = pmesh->DefaultNumPartitions(); // Usually 1
    TaskRegion &bound_sync = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        auto &tl = bound_sync[i];
        // This is a member function of KHARMADriver, so it inherits 'integrator'
        auto &mbd_sub_step_final = pmesh->mesh_data.GetOrAdd(integrator->stage_name[stage], i);
        AddMPIBoundarySync(t_none, tl, mbd_sub_step_final);
    }

    // Parthenon's call for bounds is MeshBlock, it sucks
    int nblocks = pmesh->block_list.size();
    TaskRegion &async_region2 = tc.AddRegion(nblocks);
    for (int i = 0; i < nblocks; i++) {
        auto &pmb = pmesh->block_list[i];
        auto &tl  = async_region2[i];
        auto &mbd_sub_step_final = pmb->meshblock_data.Get(integrator->stage_name[stage]);
        tl.AddTask(t_none, parthenon::ApplyBoundaryConditions, mbd_sub_step_final);
    }

}

TaskID KHARMADriver::AddMPIBoundarySync(TaskID t_start, TaskList &tl, std::shared_ptr<MeshData<Real>> mc1)
{
    // Readability
    using parthenon::cell_centered_bvars::SendBoundBufs;
    using parthenon::cell_centered_bvars::ReceiveBoundBufs;
    using parthenon::cell_centered_bvars::SetBounds;
    constexpr auto local = parthenon::BoundaryType::local;
    constexpr auto nonlocal = parthenon::BoundaryType::nonlocal;
    // Send all, receive/set local after sending
    auto send =
        tl.AddTask(t_start, parthenon::cell_centered_bvars::SendBoundBufs<nonlocal>, mc1);

    auto t_send_local =
        tl.AddTask(t_start, parthenon::cell_centered_bvars::SendBoundBufs<local>, mc1);
    auto t_recv_local =
        tl.AddTask(t_start, parthenon::cell_centered_bvars::ReceiveBoundBufs<local>, mc1);
    auto t_set_local =
        tl.AddTask(t_recv_local, parthenon::cell_centered_bvars::SetBounds<local>, mc1);

    // Receive/set nonlocal
    auto t_recv = tl.AddTask(
        t_start, parthenon::cell_centered_bvars::ReceiveBoundBufs<nonlocal>, mc1);
    auto t_set = tl.AddTask(t_recv, parthenon::cell_centered_bvars::SetBounds<nonlocal>, mc1);

    // TODO add AMR prolongate/restrict here (and/or maybe option not to?)

    return t_set | t_set_local;
}

void KHARMADriver::SyncAllBounds(std::shared_ptr<MeshData<Real>> md, bool apply_domain_bounds)
{
    Flag("Syncing all bounds");
    TaskID t_none(0);

    // If we're using the ImEx driver, where primitives are fundamental, AddMPIBoundarySync()
    // will only sync those, and we can call PtoU over everything after.
    // If "AddMPIBoundarySync" means syncing conserved variables, we have to call PtoU *before*
    // the MPI sync operation, then recover the primitive vars *again* afterward.
    auto pmesh = md->GetMeshPointer();
    bool sync_prims = pmesh->packages.Get("Driver")->Param<bool>("sync_prims");

    // TODO clean this up when ApplyBoundaryConditions gets a MeshData version
    auto &block_list = pmesh->block_list;

    if (sync_prims) {
        // If we're syncing the primitive vars, we just sync once
        TaskCollection tc;
        auto tr = tc.AddRegion(1);
        AddMPIBoundarySync(t_none, tr[0], md);
        while (!tr.Execute());

        // Then PtoU
        for (auto &pmb : block_list) {
            auto& rc = pmb->meshblock_data.Get();

            Flag("Block fill Conserved");
            Flux::BlockPtoU(rc.get(), IndexDomain::entire, false);

            if (apply_domain_bounds) {
                Flag("Block physical bounds");
                // Physical boundary conditions
                parthenon::ApplyBoundaryConditions(rc);
            }
        }
    } else {
        // If we're syncing the conserved vars...
        // Honestly, the easiest way through this sync is:
        // 1. PtoU everywhere
        for (auto &pmb : block_list) {
            auto& rc = pmb->meshblock_data.Get();
            Flag("Block fill conserved");
            Flux::BlockPtoU(rc.get(), IndexDomain::entire, false);
        }

        // 2. Sync MPI bounds like a normal step
        TaskCollection tc;
        auto tr = tc.AddRegion(1);
        AddMPIBoundarySync(t_none, tr[0], md);
        while (!tr.Execute());

        // 3. UtoP everywhere
        for (auto &pmb : block_list) {
            auto& rc = pmb->meshblock_data.Get();

            Flag("Block fill Derived");
            // Fill P again, including ghost zones
            // But, sice we sync'd GRHD primitives already,
            // leave those off
            // (like we do in a normal boundary sync)
            Packages::BlockUtoPExceptMHD(rc.get(), IndexDomain::entire);

            if (apply_domain_bounds) {
                Flag("Block physical bounds");
                // Physical boundary conditions
                parthenon::ApplyBoundaryConditions(rc);
            }
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
        std::cerr << "Reconstruction type not supported!  Supported reconstructions:" << std::endl;
        std::cerr << "donor_cell, linear_mc, linear_vl, weno5" << std::endl;
        throw std::invalid_argument("Unsupported reconstruction algorithm!");
    }
    return t_calculate_flux1 | t_calculate_flux2 | t_calculate_flux3;
}