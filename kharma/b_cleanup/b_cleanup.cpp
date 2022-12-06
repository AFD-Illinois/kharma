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
#include <solvers/bicgstab_solver.hpp>

#include "b_cleanup.hpp"

// For a bunch of utility functions
#include "b_flux_ct.hpp"

#include "boundaries.hpp"
#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "mpi.hpp"

using namespace parthenon;
using namespace parthenon::solvers;

// TODO get the transport manager working later
// Needs a call every X steps option, probably return a TaskList or TaskRegion

namespace B_Cleanup
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    Flag("Initializing B Field Cleanup");
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
    // TODO add an absolute tolerance to the Parthenon BiCGStab solver
    Real abs_tolerance = pin->GetOrAddReal("b_cleanup", "abs_tolerance", 1e-11);
    params.Add("abs_tolerance", abs_tolerance);
    int max_iterations = pin->GetOrAddInteger("b_cleanup", "max_iterations", 1e8);
    params.Add("max_iterations", max_iterations);
    int check_interval = pin->GetOrAddInteger("b_cleanup", "check_interval", 20);
    params.Add("check_interval", check_interval);
    bool fail_without_convergence = pin->GetOrAddBoolean("b_cleanup", "fail_without_convergence", true);
    params.Add("fail_without_convergence", fail_without_convergence);
    bool warn_without_convergence = pin->GetOrAddBoolean("b_cleanup", "warn_without_convergence", false);
    params.Add("warn_without_convergence", warn_without_convergence);
    bool always_solve = pin->GetOrAddBoolean("b_cleanup", "always_solve", false);
    params.Add("always_solve", always_solve);

    // TODO find a way to add this to the list every N steps
    int cleanup_interval = pin->GetOrAddInteger("b_cleanup", "cleanup_interval", 0);
    params.Add("cleanup_interval", cleanup_interval);

    // Finally, initialize the solver
    // Translate parameters
    params.Add("bicgstab_max_iterations", max_iterations);
    params.Add("bicgstab_check_interval", check_interval);
    params.Add("bicgstab_abort_on_fail", fail_without_convergence);
    params.Add("bicgstab_warn_on_fail", warn_without_convergence);
    params.Add("bicgstab_print_checks", true);

    // Sparse matrix.  Never built, we leave it blank
    pkg->AddParam<std::string>("spm_name", "");
    // Solution
    pkg->AddParam<std::string>("sol_name", "p");
    // RHS.  Must not just be "divB" as that field does not sync boundaries
    pkg->AddParam<std::string>("rhs_name", "divB_RHS");
    // Construct a solver. We don't need the template parameter, so we use 'int'
    BiCGStabSolver<int> solver(pkg.get(), rel_tolerance, SparseMatrixAccessor());
    // Set callback
    solver.user_MatVec = CornerLaplacian;

    params.Add("solver", solver);

    // FIELDS
    std::vector<int> s_vector({NVEC});
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::FillGhost});
    // Scalar potential, solution to div^2 p = div B
    pkg->AddField("p", m);
    // Gradient of potential; temporary for gradient calc
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
    pkg->AddField("dB", m);
    // Field divergence as RHS, i.e. including boundary sync
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    pkg->AddField("divB_RHS", m);


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

    auto pmesh = md->GetMeshPointer();
    auto pkg = pmesh->packages.Get("B_Cleanup");
    auto max_iters = pkg->Param<int>("max_iterations");
    auto check_interval = pkg->Param<int>("check_interval");
    auto rel_tolerance = pkg->Param<Real>("rel_tolerance");
    auto abs_tolerance = pkg->Param<Real>("abs_tolerance");
    auto fail_flag = pkg->Param<bool>("fail_without_convergence");
    auto warn_flag = pkg->Param<bool>("warn_without_convergence");
    auto always_solve = pkg->Param<bool>("always_solve");
    auto verbose = pkg->Param<int>("verbose");
    auto solver = pkg->Param<BiCGStabSolver<int>>("solver");
    MetadataFlag isMHD = pmesh->packages.Get("GRMHD")->Param<MetadataFlag>("MHDFlag");


    if (MPIRank0() && verbose > 0) {
        std::cout << "Cleaning divB to relative tolerance " << rel_tolerance << std::endl;
        if (warn_flag) std::cout << "Convergence failure will produce a warning." << std::endl;
        if (fail_flag) std::cout << "Convergence failure will produce an error." << std::endl;
    }

    // Calculate/print inital max divB exactly as we would during run
    const double divb_start = B_FluxCT::GlobalMaxDivB(md.get());
    if (divb_start < rel_tolerance && !always_solve) {
        // If divB is "pretty good" and we allow not solving...
        if (MPIRank0())
            std::cout << "Magnetic field divergence of " << divb_start << " is below tolerance. Skipping B field cleanup." << std::endl;
        return;
    } else {
        if(MPIRank0())
            std::cout << "Starting magnetic field divergence: " << divb_start << std::endl;
    }

    // Initialize the divB variable, which we'll be solving against.
    // This gets signed divB on all physical corners (total (N+1)^3)
    // and syncs ghost zones
    B_FluxCT::CalcDivB(md.get(), "divB_RHS");
    KBoundaries::SyncAllBounds(md);

    // Add a solver container and associated MeshData
    for (auto& pmb : pmesh->block_list) {
        auto &base = pmb->meshblock_data.Get();
        pmb->meshblock_data.Add("solve", base);
    }
    // The "solve" container really only needs the RHS, the solution, and the scratch array dB
    // This does not affect the main container, but saves a *lot* of time not syncing
    // static variables.
    // There's no MeshData-wide 'Remove' so we go block-by-block
    for (auto& pmb : pmesh->block_list) {
        auto rc_s = pmb->meshblock_data.Get("solve");
        auto varlabels = rc_s->GetVariablesByFlag({isMHD}, true).labels();
        for (auto varlabel : varlabels) {
            rc_s->Remove(varlabel);
        }
    }
    auto &msolve = pmesh->mesh_data.GetOrAdd("solve", 0);

    // Create a TaskCollection of just the solve,
    // execute it to perform BiCGStab iteration
    TaskID t_none(0);
    TaskCollection tc;
    auto tr = tc.AddRegion(1);
    auto t_solve_step = solver.CreateTaskList(t_none, 0, tr, md, msolve);
    while (!tr.Execute());
    // Make sure solution's ghost zones are sync'd
    KBoundaries::SyncAllBounds(msolve);

    // Apply the result
    if (MPIRank0() && verbose > 0) {
        std::cout << "Applying magnetic field correction" << std::endl;
    }
    // Update the magnetic field on physical zones using our solution
    B_Cleanup::ApplyP(msolve.get(), md.get());

    // Synchronize to update ghost zones
    KBoundaries::SyncAllBounds(md);

    // Recalculate divB max for one last check
    const double divb_end = B_FluxCT::GlobalMaxDivB(md.get());
    if (MPIRank0()) {
        std::cout << "Magnetic field divergence after cleanup: " << divb_end << std::endl;
    }

    Flag(md.get(), "Cleaned");
}

TaskStatus ApplyP(MeshData<Real> *msolve, MeshData<Real> *md)
{
    Flag(md, "Applying correction from P");
    // Apply on physical zones only, we'll be syncing/updating ghosts
    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto P = msolve->PackVariables(std::vector<std::string>{"p"});
    auto B = md->PackVariables(std::vector<std::string>{"cons.B"});

    const int ndim = P.GetNdim();

    // dB = grad(p), defined at cell centers, subtract to make field divergence-free
    pmb0->par_for("gradient_P", 0, P.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D {
            const auto& G = P.GetCoords(b);
            double b1, b2, b3;
            B_FluxCT::center_grad(G, P, b, k, j, i, ndim > 2, b1, b2, b3);
            B(b, V1, k, j, i) -= b1;
            B(b, V2, k, j, i) -= b2;
            B(b, V3, k, j, i) -= b3;
        }
    );

    B_FluxCT::UtoP(md, IndexDomain::entire);

    return TaskStatus::complete;
}

TaskStatus CornerLaplacian(MeshData<Real>* md, const std::string& p_var, const std::string& lap_var)
{
    Flag(md, "Calculating & summing divB");
    // Cover ghost cells; maximize since both ops have stencil >1
    const IndexRange ib = md->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = md->GetBoundsK(IndexDomain::entire);
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto P = md->PackVariables(std::vector<std::string>{p_var});
    auto lap = md->PackVariables(std::vector<std::string>{lap_var});
    auto dB = md->PackVariables(std::vector<std::string>{"dB"}); // Temp

    const int ndim = P.GetNdim();

    const IndexRange ib_l = IndexRange{ib.s, ib.e-1};
    const IndexRange jb_l = (ndim > 1) ? IndexRange{jb.s, jb.e-1} : jb;
    const IndexRange kb_l = (ndim > 2) ? IndexRange{kb.s, kb.e-1} : kb;
    const IndexRange ib_r = IndexRange{ib.s+1, ib.e-1};
    const IndexRange jb_r = (ndim > 1) ? IndexRange{jb.s+1, jb.e-1} : jb;
    const IndexRange kb_r = (ndim > 2) ? IndexRange{kb.s+1, kb.e-1} : kb;

    // dB = grad(p), defined at cell centers
    // Need a halo one zone *left*, as corner_div will read that.
    // Therefore B's ghosts need to be up to date!
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
    pmb0->par_for("laplacian_dB", 0, lap.GetDim(5) - 1, kb_r.s, kb_r.e, jb_r.s, jb_r.e, ib_r.s, ib_r.e,
        KOKKOS_LAMBDA_MESH_3D {
            const auto& G = lap.GetCoords(b);
            // This is the inverse diagonal element of a fictional a_ij Laplacian operator
            lap(b, 0, k, j, i) = B_FluxCT::corner_div(G, dB, b, k, j, i, ndim > 2);
        }
    );

    return TaskStatus::complete;
}

} // namespace B_Cleanup
