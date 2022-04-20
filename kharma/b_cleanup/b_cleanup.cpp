/* 
 *  File: b_cleanup.cpp
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

#include <parthenon/parthenon.hpp>

#include "b_cleanup.hpp"

// For a bunch of utility functions
#include "b_flux_ct.hpp"

#include "boundaries.hpp"
#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "mpi.hpp"

using namespace parthenon;

namespace B_Cleanup
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    auto pkg = std::make_shared<StateDescriptor>("B_Cleanup");
    Params &params = pkg->AllParams();

    // OPTIONS
    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // Solver options
    // Allow setting tolerance relative to starting value.  Off by default
    Real rel_tolerance = pin->GetOrAddReal("b_cleanup", "rel_tolerance", 1.);
    params.Add("rel_tolerance", rel_tolerance);
    // Instead set absolute tolerance corresponding roughly to max divB on grid
    // Note this returns divB max about 1 decade greater, i.e. ~1e-14
    Real abs_tolerance = pin->GetOrAddReal("b_cleanup", "abs_tolerance", 1e-11);
    params.Add("abs_tolerance", abs_tolerance);
    // TODO why does this need to be so large?
    Real sor_factor = pin->GetOrAddReal("b_cleanup", "sor_factor", 10.);
    params.Add("sor_factor", sor_factor);
    int max_iterations = pin->GetOrAddInteger("b_cleanup", "max_iterations", 1e8);
    params.Add("max_iterations", max_iterations);
    int check_interval = pin->GetOrAddInteger("b_cleanup", "check_interval", 2e2);
    params.Add("check_interval", check_interval);
    bool fail_without_convergence = pin->GetOrAddBoolean("b_cleanup", "fail_without_convergence", true);
    params.Add("fail_without_convergence", fail_without_convergence);
    bool warn_without_convergence = pin->GetOrAddBoolean("b_cleanup", "warn_without_convergence", false);
    params.Add("warn_without_convergence", warn_without_convergence);

    // TODO find a way to add this to the list every N steps
    int cleanup_interval = pin->GetOrAddInteger("b_cleanup", "cleanup_interval", 0);
    params.Add("cleanup_interval", cleanup_interval);

    // Someday we could use the fancy Parthenon tools to fill a sparse matrix representing
    // div(grad(p)).  However, this is complicated by the averaging to centers/corners required
    // when preserving divB at corners for flux-CT
    // It would be much easier to set up for a lone/cleanup+Dedner setup

    // FIELDS
    std::vector<int> s_vector({NVEC});
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    // Scalar potential, solution to div^2 p = div B
    // Thus when we subtract the gradient, div (B - div p) == 0!
    pkg->AddField("p", m);

    // Scalar laplacian div^2 p. No need to sync this, we write/read it only on physical zones
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("lap", m);
    // Gradient of potential, for use when computing laplacian & for final subtraction
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
    pkg->AddField("dB", m);

    // If there's not another B field transport (dangerous!), take care of it ourselves.
    // Allocate the field, register most of the B_FluxCT callbacks
    // TODO check if B is allocated and set this if not
    bool manage_field = pin->GetOrAddBoolean("b_cleanup", "manage_field", false);
    params.Add("manage_field", manage_field);
    if (manage_field) {
        MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
        MetadataFlag isMHD = packages.Get("GRMHD")->Param<MetadataFlag>("MHDFlag");

        // B fields.  "Primitive" form is field, "conserved" is flux
        // Note: when changing metadata, keep these in lockstep with grmhd.cpp!!
        // See notes there about changes for the Imex driver
        std::vector<MetadataFlag> flags_prim, flags_cons;
        auto imex_driver = pin->GetString("driver", "type") == "imex";
        if (!imex_driver) {
            flags_prim = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Derived,
                                                    isPrimitive, isMHD, Metadata::Vector});
            flags_cons = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                    Metadata::Restart, Metadata::Conserved, isMHD, Metadata::WithFluxes, Metadata::Vector});
        } else {
            flags_prim = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::FillGhost, Metadata::Restart,
                                                    isPrimitive, isMHD, Metadata::Vector});
            flags_cons = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Independent,
                                                    Metadata::Conserved, isMHD, Metadata::WithFluxes, Metadata::Vector});
        }

        m = Metadata(flags_prim, s_vector);
        pkg->AddField("prims.B", m);
        m = Metadata(flags_cons, s_vector);
        pkg->AddField("cons.B", m);

        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("divB", m);

        pkg->FillDerivedMesh = B_FluxCT::FillDerivedMesh;
        pkg->FillDerivedBlock = B_FluxCT::FillDerivedBlock;
        pkg->PostStepDiagnosticsMesh = B_FluxCT::PostStepDiagnostics;

        // List (vector) of HistoryOutputVar that will all be enrolled as output variables
        parthenon::HstVar_list hst_vars = {};
        // The definition of MaxDivB we care about actually changes per-transport. Use our function.
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, B_FluxCT::MaxDivB, "MaxDivB"));
        // add callbacks for HST output to the Params struct, identified by the `hist_param_key`
        pkg->AddParam<>(parthenon::hist_param_key, hst_vars);
    }

    return pkg;
}

void CleanupDivergence(std::shared_ptr<MeshData<Real>>& md)
{
    Flag(md.get(), "Cleaning up divB");
    // Local Allreduce values since we're just calling things
    AllReduce<Real> update_norm, divB_norm, divB_max, P_norm;

    auto pmesh = md->GetMeshPointer();
    auto pkg = pmesh->packages.Get("B_Cleanup");
    auto max_iters = pkg->Param<int>("max_iterations");
    auto check_interval = pkg->Param<int>("check_interval");
    auto rel_tolerance = pkg->Param<Real>("rel_tolerance");
    auto abs_tolerance = pkg->Param<Real>("abs_tolerance");
    auto fail_flag = pkg->Param<bool>("fail_without_convergence");
    auto warn_flag = pkg->Param<bool>("warn_without_convergence");
    auto verbose = pkg->Param<int>("verbose");
    bool sync_prims = pmesh->packages.Get("GRMHD")->Param<std::string>("driver_type") == "imex";

    if (MPIRank0() && verbose > 0) {
        std::cout << "Cleaning divB to relative tolerance " << rel_tolerance;
        std::cout << " and absolute tolerance " << abs_tolerance << std::endl;
        if (warn_flag) std::cout << "Convergence failure will produce a warning." << std::endl;
        if (fail_flag) std::cout << "Convergence failure will produce an error." << std::endl;
    }

    // Calculate existing divB max & sum for checking relative error later
    divB_max.val = 0.;
    B_FluxCT::MaxDivBTask(md.get(), divB_max.val);
    divB_max.StartReduce(MPI_MAX);

    divB_norm.val = 0.;
    B_Cleanup::CalcSumDivB(md.get(), divB_norm.val);
    divB_norm.StartReduce(MPI_SUM);

    // Wait on results
    while (divB_max.CheckReduce() == TaskStatus::incomplete);
    while (divB_norm.CheckReduce() == TaskStatus::incomplete);

    if (MPIRank0() && verbose > 0) {
        std::cout << "Starting divB max is " << divB_max.val << " and sum is " << divB_norm.val << std::endl;
    }
    // These two aren't *strictly* comparable, but we're unlikely to do any good if this is true
    if (divB_max.val < abs_tolerance) {
        std::cout << "Starting divB is within tolerance, exiting." << std::endl;
        return;
    }

    // TODO Unmark everything but P as FillGhost, for efficiency. Re-mark before last sync
    // auto vars = rc->GetVariablesByFlag(std::vector<MetadataFlag>({isImplicit, flag}), true).labels();
    // for (auto& var : vars) {
    //     // etc
    // }

    // set P = divB as guess
    B_Cleanup::InitP(md.get());

    bool is_converged = false;
    int iter = 0;
    while ( (!is_converged) && (iter < max_iters) ) {
        // Update our guess at the potential 
        B_Cleanup::UpdateP(md.get());

        // Boundary sync. We really only need p syncd here...
        // Last option prevents updating physical boundaries, which we want to *solve* instead
        KBoundaries::SyncAllBounds(pmesh, sync_prims, false);

        if (iter % check_interval == 0) {
            Flag("Iteration:");
            // Calculate the new norm & relative error in eliminating divB
            update_norm.val = 0.;
            B_Cleanup::SumError(md.get(), update_norm.val);
            update_norm.StartReduce(MPI_SUM);
            // P_norm.val = 0.;
            // B_Cleanup::SumP(md.get(), P_norm.val);
            // P_norm.StartReduce(MPI_SUM);
            divB_max.val = 0.;
            MaxError(md.get(), divB_max.val);
            divB_max.StartReduce(MPI_MAX);
            // Wait on both reductions to move on
            while (update_norm.CheckReduce() == TaskStatus::incomplete);
            //while (P_norm.CheckReduce() == TaskStatus::incomplete);
            while (divB_max.CheckReduce() == TaskStatus::incomplete);
            if (MPIRank0()) {
                std::cout << "divB step " << iter << " total relative error is " << update_norm.val / divB_norm.val
                        << " Max absolute error is " << divB_max.val << std::endl;
                // std::cout << "P norm is " << P_norm.val << std::endl;
            }

            // This behaves identically on ranks, unless we've broken a fundamental assumption
            is_converged = (update_norm.val / divB_norm.val < rel_tolerance) && (divB_max.val < abs_tolerance);
        }

        iter++;
    }
    if (iter >= max_iters) {
        if (fail_flag) {
            throw std::runtime_error("Failed to converge when cleaning magnetic field divergence!");
        } else if (warn_flag) {
            cerr << "KHARMA WARNING: Failed to converge when cleaning magnetic field divergence!!!!" << endl;
        }
    }

    if (MPIRank0() && verbose > 0) {
        std::cout << "Applying magnetic field correction" << std::endl;
    }

    // Update the magnetic field on physical zones using our solution
    B_Cleanup::ApplyP(md.get());
    // Synchronize to update ghost zones
    KBoundaries::SyncAllBounds(pmesh, sync_prims);

    // Recalculate divB max for one last check
    divB_max.val = 0.;
    B_FluxCT::MaxDivBTask(md.get(), divB_max.val);
    divB_max.StartReduce(MPI_MAX);
    while (divB_max.CheckReduce() == TaskStatus::incomplete);

    if (MPIRank0()) {
        std::cout << "Final divB max is " << divB_max.val << std::endl;
    }

    Flag(md.get(), "Cleaned");
}

TaskStatus CalcSumDivB(MeshData<Real> *md, Real& reduce_sum)
{
    Flag(md, "Calculating & summing divB");
    auto pm = md->GetParentPointer();
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Get variables
    auto B = md->PackVariables(std::vector<std::string>{"cons.B"});
    auto divB = md->PackVariables(std::vector<std::string>{"divB"});

    const int ndim = B.GetNdim();

    // Total divB.
    Real divB_total;
    pmb0->par_reduce("SumDivB", 0, B.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE {
            const auto& G = B.GetCoords(b);
            divB(b, 0, k, j, i) = B_FluxCT::corner_div(G, B, b, k, j, i, ndim > 2);
            local_result += abs(divB(b, 0, k, j, i));
        }
    , Kokkos::Sum<Real>(divB_total));

    // Parthenon/caller will take care of MPI reduction
    reduce_sum += divB_total;
    return TaskStatus::complete;
}

TaskStatus InitP(MeshData<Real> *md)
{
    Flag(md, "Initializing P");
    auto pm = md->GetParentPointer();
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Pack variables
    auto P = md->PackVariables(std::vector<std::string>{"p"});
    auto divB = md->PackVariables(std::vector<std::string>{"divB"});

    // Initialize P = divB
    pmb0->par_for("init_p", 0, P.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D {
            P(b, 0, k, j, i) = divB(b, 0, k, j, i);
        }
    );

    return TaskStatus::complete;
}

TaskStatus UpdateP(MeshData<Real> *md)
{
    Flag(md, "Updating P");
    auto pmesh = md->GetParentPointer();
    const int ndim = pmesh->ndim;
    const IndexRange ib = md->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = md->GetBoundsK(IndexDomain::entire);
    const IndexRange ib_l = IndexRange{ib.s, ib.e-1};
    const IndexRange jb_l = (ndim > 1) ? IndexRange{jb.s, jb.e-1} : jb;
    const IndexRange kb_l = (ndim > 2) ? IndexRange{kb.s, kb.e-1} : kb;
    const IndexRange ib_r = IndexRange{ib.s+1, ib.e-1};
    const IndexRange jb_r = (ndim > 1) ? IndexRange{jb.s+1, jb.e-1} : jb;
    const IndexRange kb_r = (ndim > 2) ? IndexRange{kb.s+1, kb.e-1} : kb;

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    const int n1 = pmb0->cellbounds.ncellsi(IndexDomain::interior);

    // Options
    auto pkg = md->GetMeshPointer()->packages.Get("B_Cleanup");
    const auto omega = pkg->Param<double>("sor_factor");

    // Pack variables
    auto P = md->PackVariables(std::vector<std::string>{"p"});
    auto lap = md->PackVariables(std::vector<std::string>{"lap"});
    auto dB = md->PackVariables(std::vector<std::string>{"dB"});
    auto divB = md->PackVariables(std::vector<std::string>{"divB"});

    // TODO Multigrid for faster than ~N^2 convergence
    // TODO don't sync all boundaries here, we only need p

    // dB = grad(p), defined at cell centers
    // Need a halo one zone *left*, as corner_div will read that.
    // Therefore P's ghosts need to be up to date!
    pmb0->par_for("gradient_P", 0, P.GetDim(5) - 1, kb_l.s, kb_l.e, jb_l.s, jb_l.e, ib_l.s, ib_l.e,
        KOKKOS_LAMBDA_MESH_3D {
            const auto& G = P.GetCoords(b);
            double b1, b2, b3;
            B_FluxCT::center_grad(G, P, b, k, j, i, ndim > 2, b1, b2, b3);
            dB(b, V1, k, j, i) = b1;
            dB(b, V2, k, j, i) = b2;
            dB(b, V3, k, j, i) = b3;
        }
    );

    // lap = div(dB), defined at cell corners
    // Then apply a damped Jacobi iteration
    pmb0->par_for("laplacian_dB", 0, lap.GetDim(5) - 1, kb_r.s, kb_r.e, jb_r.s, jb_r.e, ib_r.s, ib_r.e,
        KOKKOS_LAMBDA_MESH_3D {
            const auto& G = lap.GetCoords(b);
            // This is the inverse diagonal element of a fictional a_ij Laplacian operator
            // denoted D^-1 below. Note it's not quite what a_ij might work out to for our "laplacian"
            const double dt = (ndim > 2) ? (-1./6) * G.dx1v(i) * G.dx2v(j) * G.dx3v(k) : (-1./4) * G.dx1v(i) * G.dx2v(j);
            lap(b, 0, k, j, i) = B_FluxCT::corner_div(G, dB, b, k, j, i, ndim > 2);
            // In matrix notation the following would be:
            // x^k+1 = omega*D^-1*(b - (L + U) x^k) + (1-omega)*x^k
            // But since we can't actually calculate L+U, we use A*x-D*x
            //P(b, 0, k, j, i) = omega*dt*(divB(b, 0, k, j, i) - (lap(b, 0, k, j, i) - 1/dt*P(b, 0, k, j, i)))
            //                    + (1 - omega)*P(b, 0, k, j, i);
            // ...or more simply...
            P(b, 0, k, j, i) += omega*dt*(divB(b, 0, k, j, i) - lap(b, 0, k, j, i));

        }
    );

    Flag("Updated");
    return TaskStatus::complete;
}

TaskStatus SumError(MeshData<Real> *md, Real& reduce_sum)
{
    Flag(md, "Summing remaining error term");
    auto pm = md->GetParentPointer();
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Get variables
    auto lap = md->PackVariables(std::vector<std::string>{"lap"});
    auto divB = md->PackVariables(std::vector<std::string>{"divB"});

    // TODO this can be done as
    // 1. (K*lap - divB) as here
    // 2. (div of (B - dB)), simulating the actual result
    // The latter would require a full/scratch vector temporary, and
    // setting FillGhost on dB, but the sync is in the right spot
    Real err_total;
    pmb0->par_reduce("SumError", 0, lap.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE {
            local_result += abs(lap(b, 0, k, j, i) - divB(b, 0, k, j, i));
        }
    , Kokkos::Sum<Real>(err_total));

    // Parthenon/caller will take care of MPI reduction
    reduce_sum += err_total;
    return TaskStatus::complete;
}

TaskStatus MaxError(MeshData<Real> *md, Real& reduce_max)
{
    Flag(md, "Max new divB");
    auto pm = md->GetParentPointer();
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Get variables
    auto lap = md->PackVariables(std::vector<std::string>{"lap"});
    auto divB = md->PackVariables(std::vector<std::string>{"divB"});

    // TODO this can be done as
    // 1. (K*lap - divB) as here
    // 2. (div of (B - dB)), simulating the actual result
    // The latter would require a full/scratch vector temporary, and
    // setting FillGhost on dB, but the sync is in the right spot
    Real err_max;
    pmb0->par_reduce("SumError", 0, lap.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE {
            const double new_err = abs(lap(b, 0, k, j, i) - divB(b, 0, k, j, i));
            if (new_err > local_result) local_result = new_err;
        }
    , Kokkos::Max<Real>(err_max));

    // Parthenon/caller will take care of MPI reduction
    reduce_max += err_max;
    return TaskStatus::complete;
}

TaskStatus SumP(MeshData<Real> *md, Real& reduce_sum)
{
    Flag(md, "Summing P");
    auto pm = md->GetParentPointer();
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Get variables
    auto P = md->PackVariables(std::vector<std::string>{"p"});

    Real P_total;
    pmb0->par_reduce("SumError", 0, P.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D_REDUCE {
            local_result += abs(P(b, 0, k, j, i));
        }
    , Kokkos::Sum<Real>(P_total));

    // Parthenon/caller will take care of MPI reduction
    reduce_sum += P_total;
    return TaskStatus::complete;
}

TaskStatus ApplyP(MeshData<Real> *md)
{
    Flag(md, "Applying divB correction");
    auto pm = md->GetParentPointer();
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Pack variables
    auto P = md->PackVariables(std::vector<std::string>{"p"});
    auto B = md->PackVariables(std::vector<std::string>{"cons.B"});

    const int ndim = B.GetNdim();

    // Apply B -= grad(p) to actually remove divergence
    pmb0->par_for("apply_dp", 0, P.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D {
            const auto& G = P.GetCoords(b);
            double b1, b2, b3;
            B_FluxCT::center_grad(G, P, b, k, j, i, ndim > 2, b1, b2, b3);
            B(b, V1, k, j, i) -= b1;
            B(b, V2, k, j, i) -= b2;
            if (ndim > 2) {
                B(b, V3, k, j, i) -= b3;
            } else {
                B(b, V3, k, j, i) = 0;
            }
        }
    );

    B_FluxCT::UtoP(md);

    return TaskStatus::complete;
}

// TODO get this working later. Needs:
// 1. Some way to call every X steps (just return converged if off-cadence?)
// 2. Parameters, mesh pointer, or just driver pointer as arg
// 3. Is this a good idea here?  More broadly? e.g. for MPI sync, sources, etc?
// void AddBCleanupTasks(TaskList& tl, const TaskID& t_dep, AllReduce<Real>& update_norm) {
//     TaskID t_none(0);

//     auto pkg = md->GetMeshPointer()->packages.Get("B_Cleanup");
//     auto max_iters = pkg->Param<int>("max_iterations");
//     auto check_interval = pkg->Param<int>("check_interval");
//     auto fail_flag = pkg->Param<bool>("fail_without_convergence");
//     auto warn_flag = pkg->Param<bool>("warn_without_convergence");

//     const int num_partitions = md->GetMeshPointer()->DefaultNumPartitions();
//     TaskRegion &solver_region = tc.AddRegion(num_partitions);
//     for (int i = 0; i < num_partitions; i++) {
//         int reg_dep_id = 0;

//         auto &t_solver = tl.AddIteration("B field cleanup");
//         t_solver.SetMaxIterations(max_iters);
//         t_solver.SetCheckInterval(check_interval);
//         t_solver.SetFailWithMaxIterations(fail_flag);
//         t_solver.SetWarnWithMaxIterations(warn_flag);
//         auto t_start_recv = t_solver.AddTask(t_dep, &MeshData<Real>::StartReceiving, md.get(),
//                                         BoundaryCommSubset::all);

//         auto t_update = t_solver.AddTask(t_start_recv, B_Cleanup::UpdatePhi,
//                                     md.get(), mdelta.get());

//         auto t_norm = t_solver.AddTask(t_update, B_Cleanup::SumDeltaPhi,
//                                 mdelta.get(), &update_norm.val);
//         solver_region.AddRegionalDependencies(reg_dep_id, i, t_norm);
//         reg_dep_id++;
//         auto t_start_reduce_norm = (i == 0 ? t_solver.AddTask(t_norm, &AllReduce<Real>::StartReduce,
//                                                         &update_norm, MPI_SUM)
//                                         : t_none);
//         auto finish_reduce_norm =
//             t_solver.AddTask(start_reduce_norm, &AllReduce<Real>::CheckReduce, &update_norm);
//         auto t_report_norm = (i == 0 ? t_solver.AddTask(
//                                         finish_reduce_norm,
//                                         [](Real *norm) {
//                                             if (Globals::my_rank == 0) {
//                                                 std::cout << "Update norm = " << *norm << std::endl;
//                                             }
//                                             *norm = 0.0;
//                                             return TaskStatus::complete;
//                                         },
//                                         &update_norm.val)
//                                 : none);

//         auto t_send = t_solver.AddTask(t_update, cell_centered_bvars::SendBoundaryBuffers, md);

//         auto t_recv =
//             t_solver.AddTask(t_start_recv, cell_centered_bvars::ReceiveBoundaryBuffers, md);

//         auto t_setb = t_solver.AddTask(t_recv | t_update, cell_centered_bvars::SetBoundaries, md);

//         auto t_clear = t_solver.AddTask(t_send | t_setb | t_report_norm, &MeshData<Real>::ClearBoundary,
//                                     md.get(), BoundaryCommSubset::all);

//         auto t_check = t_solver.SetCompletionTask(
//             t_clear, B_Cleanup::CheckConvergence, md.get(), mdelta.get());
//         // mark task so that dependent tasks (below) won't execute
//         // until all task lists have completed it
//         solver_region.AddRegionalDependencies(reg_dep_id, i, t_check);
//         reg_dep_id++;
//     }
// }

} // namespace B_Cleanup
