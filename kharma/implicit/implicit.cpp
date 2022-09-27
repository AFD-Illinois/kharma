/* 
 *  File: implicit.cpp
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

// Floors.  Apply limits to fluid values to maintain integrable state

#include "implicit.hpp"

#include "debug.hpp"
#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"

#include <batched/dense/KokkosBatched_LU_Decl.hpp>
#include <batched/dense/impl/KokkosBatched_LU_Serial_Impl.hpp>
#include <batched/dense/KokkosBatched_Trsv_Decl.hpp>
using namespace KokkosBatched;

namespace Implicit
{

std::vector<std::string> get_ordered_names(MeshBlockData<Real> *rc, const MetadataFlag& flag, bool only_implicit=false) {
    auto pmb0 = rc->GetBlockPointer();
    MetadataFlag isImplicit = pmb0->packages.Get("Implicit")->Param<MetadataFlag>("ImplicitFlag");
    MetadataFlag isExplicit = pmb0->packages.Get("Implicit")->Param<MetadataFlag>("ExplicitFlag");
    std::vector<std::string> out;
    auto vars = rc->GetVariablesByFlag(std::vector<MetadataFlag>({isImplicit, flag}), true).labels();
    for (int i=0; i < vars.size(); ++i) {
        if (rc->Contains(vars[i])) {
            out.push_back(vars[i]);
        }
    }
    if (!only_implicit) {
        vars = rc->GetVariablesByFlag(std::vector<MetadataFlag>({isExplicit, flag}), true).labels();
        for (int i=0; i < vars.size(); ++i) {
            if (rc->Contains(vars[i])) {
                out.push_back(vars[i]);
            }
        }
    }
    return out;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin)
{
    Flag("Initializing Implicit Package");
    auto pkg = std::make_shared<StateDescriptor>("Implicit");
    Params &params = pkg->AllParams();

    // Implicit solver parameters
    Real jacobian_delta = pin->GetOrAddReal("implicit", "jacobian_delta", 4.e-8);
    params.Add("jacobian_delta", jacobian_delta);
    Real rootfind_tol = pin->GetOrAddReal("implicit", "rootfind_tol", 1.e-9);
    params.Add("rootfind_tol", rootfind_tol);
    Real linesearch_lambda = pin->GetOrAddReal("implicit", "linesearch_lambda", 1.0);
    params.Add("linesearch_lambda", linesearch_lambda);
    int max_nonlinear_iter = pin->GetOrAddInteger("implicit", "max_nonlinear_iter", 3);
    params.Add("max_nonlinear_iter", max_nonlinear_iter);

    // Denote failures/non-converged zones with the same flag as UtoP
    // This does NOT share the same mapping of values
    // TODO currently unused
    // Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    // pkg->AddField("pflag", m);

    // When using this package we'll need to distinguish Implicitly and Explicitly-updated variables
    MetadataFlag isImplicit = Metadata::AllocateNewFlag("Implicit");
    params.Add("ImplicitFlag", isImplicit);
    MetadataFlag isExplicit = Metadata::AllocateNewFlag("Explicit");
    params.Add("ExplicitFlag", isExplicit);

    // Anything we need to run from this package on callbacks
    // None of this will be crucial for the step
    // pkg->PostFillDerivedBlock = Implicit::PostFillDerivedBlock;
    // pkg->PostStepDiagnosticsMesh = Implicit::PostStepDiagnostics;

    Flag("Initialized");
    return pkg;
}

TaskStatus Step(MeshData<Real> *mci, MeshData<Real> *mc0, MeshData<Real> *dudt,
                MeshData<Real> *mc_solver, const Real& dt)
{
    Flag(mci, "Implicit Iteration start, i");
    Flag(mc0, "Implicit Iteration start, 0");
    Flag(dudt, "Implicit Iteration start, dudt");
    auto pmb0 = mci->GetBlockData(0)->GetBlockPointer();

    const auto& implicit_par = pmb0->packages.Get("Implicit")->AllParams();
    const int iter_max = implicit_par.Get<int>("max_nonlinear_iter");
    const Real rootfind_tol = implicit_par.Get<Real>("rootfind_tol");
    const Real lambda = implicit_par.Get<Real>("linesearch_lambda");
    const Real delta = implicit_par.Get<Real>("jacobian_delta");
    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");

    EMHD_parameters emhd_params;
    if (pmb0->packages.AllPackages().count("EMHD")) {
        const auto& pars = pmb0->packages.Get("EMHD")->AllParams();
        emhd_params = pars.Get<EMHD_parameters>("emhd_params");
    }

    // I don't normally do this, but we *really* care about variable ordering here.
    // The implicit variables need to be first, so we know how to iterate over just them to fill
    // just the residual & Jacobian we care about, which makes the solve much faster.
    // This strategy is ugly but potentially gives us complete control,
    // in case Kokkos's un-pivoted LU proves problematic
    MetadataFlag isPrimitive = pmb0->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
     auto& rci = mci->GetBlockData(0); // MeshBlockData object, more member functions
    auto ordered_prims = get_ordered_names(rci.get(), isPrimitive);
    auto ordered_cons = get_ordered_names(rci.get(), Metadata::Conserved);
    //cerr << "Ordered prims:"; for(auto prim: ordered_prims) cerr << " " << prim; cerr << endl;
    //cerr << "Ordered cons:"; for(auto con: ordered_cons) cerr << " " << con; cerr << endl;

    // Initial state.  Also mapping template
    PackIndexMap prims_map, cons_map;
    auto& Pi_all = mci->PackVariables(ordered_prims, prims_map);
    auto& Ui_all = mci->PackVariables(ordered_cons, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    // Current sub-step starting state.
    auto& Ps_all = mc0->PackVariables(ordered_prims);
    auto& Us_all = mc0->PackVariables(ordered_cons);
    // Flux divergence plus explicit source terms. This is what we'd be adding.
    auto& dUdt_all = dudt->PackVariables(ordered_cons);
    // Guess at initial state. We update only the implicit primitive vars
    auto& P_solver_all = mc_solver->PackVariables(get_ordered_names(rci.get(), isPrimitive, true));

    // Sizes and scratchpads
    const int nblock = Ui_all.GetDim(5);
    const int nvar = Ui_all.GetDim(4);
    const int nfvar = P_solver_all.GetDim(4);
    auto bounds = pmb0->cellbounds;
    const int n1 = bounds.ncellsi(IndexDomain::entire);
    const int n2 = bounds.ncellsj(IndexDomain::entire);
    const int n3 = bounds.ncellsk(IndexDomain::entire);

    // RETURN if there aren't any implicit variables to evolve
    //cerr << "Solve size " << nfvar << " on prim size " << nvar << endl;
    if (nfvar == 0) return TaskStatus::complete;

    // The norm of the residual.  We store this to avoid the main kernel
    // also being a 2-stage reduction, which is complex and sucks.
    ParArray4D<Real> norm_all("norm_all", nblock, n3, n2, n1);

    // Prep Jacobian and delta arrays.
    // This lays out memory correctly & allows splitting kernel as/if we need.
    const Real alpha = 1, tiny = SMALL;
    const bool am_rank0 = MPIRank0();

    // Get meshblock array bounds from Parthenon
    const IndexDomain domain = IndexDomain::interior;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);
    const IndexRange block = IndexRange{0, nblock - 1};

    // Allocate scratch space
    // It is impossible to declare runtime-sized arrays in CUDA
    // of e.g. length var[nvar] (recall nvar can change at runtime in KHARMA)
    // Instead we copy to scratch!
    // This allows flexibility in structuring the kernel, and the results can be sliced
    // to avoid a bunch of indices in all the device-side operations
    // See grmhd_functions.hpp for the other approach with overloads
    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    const size_t var_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);
    const size_t fvar_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nfvar, n1);
    const size_t tensor_size_in_bytes = parthenon::ScratchPad3D<Real>::shmem_size(nfvar, nvar, n1);
    // Allocate enough to cache:
    // jacobian (2D)
    // residual, deltaP (implicit only)
    // Pi/Ui, Ps/Us, dUdt, P_solver, dUi, two temps (all vars)
    const size_t total_scratch_bytes = tensor_size_in_bytes + (2) * fvar_size_in_bytes + (10) * var_size_in_bytes;

    // Iterate.  This loop is outside the kokkos kernel in order to print max_norm
    // There are generally a low and similar number of iterations between
    // different zones, so probably acceptable speed loss.
    for (int iter=0; iter < iter_max; iter++) {
        // Flags per iter, since debugging here will be rampant
        Flag(mc_solver, "Implicit Iteration:");

        parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "implicit_solve", pmb0->exec_space,
            total_scratch_bytes, scratch_level, block.s, block.e, kb.s, kb.e, jb.s, jb.e,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
                const auto& G = Ui_all.GetCoords(b);
                // Scratchpads for implicit vars
                ScratchPad3D<Real> jacobian_s(member.team_scratch(scratch_level), nfvar, nfvar, n1);
                ScratchPad2D<Real> residual_s(member.team_scratch(scratch_level), nfvar, n1);
                ScratchPad2D<Real> delta_prim_s(member.team_scratch(scratch_level), nfvar, n1);
                // Scratchpads for all vars
                ScratchPad2D<Real> dUi_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> tmp1_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> tmp2_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> tmp3_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> Pi_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> Ui_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> Ps_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> Us_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> dUdt_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> P_solver_s(member.team_scratch(scratch_level), nvar, n1);

                // Copy some file contents to scratchpads, so we can slice them
                PLOOP {
                    parthenon::par_for_inner(member, ib.s, ib.e,
                        [&](const int& i) {
                            Pi_s(ip, i) = Pi_all(b)(ip, k, j, i);
                            Ui_s(ip, i) = Ui_all(b)(ip, k, j, i);
                            Ps_s(ip, i) = Ps_all(b)(ip, k, j, i);
                            Us_s(ip, i) = Us_all(b)(ip, k, j, i);
                            dUdt_s(ip, i) = dUdt_all(b)(ip, k, j, i);
                            P_solver_s(ip, i) = Ps_all(b)(ip, k, j, i);
                            dUi_s(ip, i) = 0.;
                        }
                    );
                }
                member.team_barrier();

                // Copy in the guess or current solution
                // Note this replaces the implicit portion of P_solver_s --
                // any explicit portion was initialized above
                FLOOP { // Loop over just the implicit "fluid" portion of primitive vars
                    parthenon::par_for_inner(member, ib.s, ib.e,
                        [&](const int& i) {
                            P_solver_s(ip, i) = P_solver_all(b)(ip, k, j, i);
                        }
                    );
                }
                member.team_barrier();

                parthenon::par_for_inner(member, ib.s, ib.e,
                    [&](const int& i) {
                        // Lots of slicing.  This still ends up faster & cleaner than alternatives I tried
                        auto Pi = Kokkos::subview(Pi_s, Kokkos::ALL(), i);
                        auto Ui = Kokkos::subview(Ui_s, Kokkos::ALL(), i);
                        auto Ps = Kokkos::subview(Ps_s, Kokkos::ALL(), i);
                        auto Us = Kokkos::subview(Us_s, Kokkos::ALL(), i);
                        auto dUdt = Kokkos::subview(dUdt_s, Kokkos::ALL(), i);
                        auto P_solver = Kokkos::subview(P_solver_s, Kokkos::ALL(), i);
                        // Solver variables
                        auto residual = Kokkos::subview(residual_s, Kokkos::ALL(), i);
                        auto jacobian = Kokkos::subview(jacobian_s, Kokkos::ALL(), Kokkos::ALL(), i);
                        auto delta_prim = Kokkos::subview(delta_prim_s, Kokkos::ALL(), i);
                        // Temporaries
                        auto tmp1 = Kokkos::subview(tmp1_s, Kokkos::ALL(), i);
                        auto tmp2 = Kokkos::subview(tmp2_s, Kokkos::ALL(), i);
                        auto tmp3 = Kokkos::subview(tmp3_s, Kokkos::ALL(), i);
                        // Implicit sources at starting state
                        auto dUi = Kokkos::subview(dUi_s, Kokkos::ALL(), i);
                        if (m_p.Q >= 0) {
                            EMHD::implicit_sources(G, Pi, m_p, gam, j, i, emhd_params, dUi(m_u.Q), dUi(m_u.DP));
                        }

                        // Jacobian calculation
                        // Requires calculating the residual anyway, so we grab it here
                        calc_jacobian(G, P_solver, Pi, Ui, Ps, dUdt, dUi, tmp1, tmp2, tmp3,
                                      m_p, m_u, emhd_params, nvar, nfvar, j, i, delta, gam, dt, jacobian, residual);
                        // Solve against the negative residual
                        FLOOP delta_prim(ip) = -residual(ip);

                        // if (am_rank0 && b == 0 && i == 4 && j == 4 && k == 4) {
                        //     printf("Variable ordering: rho %d uu %d u1 %d B1 %d q %d dP %d\n",
                        //             m_p.RHO, m_p.UU, m_p.U1, m_p.B1, m_p.Q, m_p.DP);
                        //     printf("Variable ordering: rho %d uu %d u1 %d B1 %d q %d dP %d\n",
                        //             m_u.RHO, m_u.UU, m_u.U1, m_u.B1, m_u.Q, m_u.DP);
                        //     // printf("P_solver: "); PLOOP printf("%g ", P_solver(ip)); printf("\n");
                        //     // printf("Pi: "); PLOOP printf("%g ", Pi(ip)); printf("\n");
                        //     // printf("Ui: "); PLOOP printf("%g ", Ui(ip)); printf("\n");
                        //     // printf("Ps: "); PLOOP printf("%g ", Ps(ip)); printf("\n");
                        //     // printf("Us: "); PLOOP printf("%g ", Us(ip)); printf("\n");
                        //     // printf("dUdt: "); PLOOP printf("%g ", dUdt(ip)); printf("\n");
                        //     printf("Initial Jacobian:\n"); for (int jp=0; jp<nvar; ++jp) {PLOOP printf("%g\t", jacobian(jp,ip)); printf("\n");}
                        //     // printf("Initial residual: "); PLOOP printf("%g ", residual(ip)); printf("\n");
                        //     // printf("Initial delta_prim: "); PLOOP printf("%g ", delta_prim(ip)); printf("\n");
                        // }

                        // Linear solve
                        // This code lightly adapted from Kokkos batched examples
                        // Replaces our inverse residual with the actual desired delta_prim
                        KokkosBatched::SerialLU<Algo::LU::Blocked>::invoke(jacobian, tiny);
                        KokkosBatched::SerialTrsv<Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsv::Blocked>
                        ::invoke(alpha, jacobian, delta_prim);

                        // Update the guess.  For now lambda == 1, choose on the fly?
                        FLOOP P_solver(ip) += lambda * delta_prim(ip);

                        calc_residual(G, P_solver, Pi, Ui, Ps, dUdt, dUi, tmp3,
                                      m_p, m_u, emhd_params, nfvar, j, i, gam, dt, residual);

                        // if (am_rank0 && b == 0 && i == 11 && j == 11 && k == 0) {
                        //     // printf("Variable ordering: rho %d uu %d u1 %d B1 %d q %d dP %d\n",
                        //     //         m_p.RHO, m_p.UU, m_p.U1, m_p.B1, m_p.Q, m_p.DP);
                        //     printf("Final residual: "); PLOOP printf("%g ", residual(ip)); printf("\n");
                        //     // printf("Final delta_prim: "); PLOOP printf("%g ", delta_prim(ip)); printf("\n");
                        //     // printf("Final P_solver: "); PLOOP printf("%g ", P_solver(ip)); printf("\n");
                        // }

                        // Store for maximum/output
                        // I would be tempted to store the whole residual, but it's of variable size
                        norm_all(b, k , j, i) = 0;
                        FLOOP norm_all(b, k, j, i) += residual(ip)*residual(ip);
                        norm_all(b, k, j, i) = sqrt(norm_all(b, k, j, i)); // TODO faster to scratch cache & copy?
                    }
                );
                member.team_barrier();

                // Copy out (the good bits of) P_solver to the existing array
                FLOOP {
                    parthenon::par_for_inner(member, ib.s, ib.e,
                        [&](const int& i) {
                            P_solver_all(b)(ip, k, j, i) = P_solver_s(ip, i);
                        }
                    );
                }
            }
        );
        
        // Take the maximum L2 norm
        Reduce<Real> max_norm;
        Kokkos::Max<Real> norm_max(max_norm.val);
        pmb0->par_reduce("max_norm", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA_MESH_3D_REDUCE {
                if (norm_all(b, k, j, i) > local_result) local_result = norm_all(b, k, j, i);
            }
        , norm_max);
        max_norm.StartReduce(0, MPI_MAX);
        while (max_norm.CheckReduce() == TaskStatus::incomplete);
        if (MPIRank0()) fprintf(stdout, "Nonlinear iter %d. Max L2 norm: %g\n", iter, max_norm.val);
        // Break if it's less than the total tolerance we set.  TODO per-zone version of this?
        if (max_norm.val < rootfind_tol) break;
    }

    Flag(mc_solver, "Implicit Iteration: final");

    return TaskStatus::complete;

}

} // namespace Implicit
