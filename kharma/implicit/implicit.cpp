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

#include "implicit.hpp"

#include "debug.hpp"
#include "grmhd.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"

std::vector<std::string> Implicit::get_ordered_names(MeshBlockData<Real> *rc, const MetadataFlag& flag, bool only_implicit)
{
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

std::shared_ptr<StateDescriptor> Implicit::Initialize(ParameterInput *pin)
{
    Flag("Initializing Implicit Package");
    auto pkg = std::make_shared<StateDescriptor>("Implicit");
    Params &params = pkg->AllParams();

    // Implicit solver parameters
    Real jacobian_delta = pin->GetOrAddReal("implicit", "jacobian_delta", 4.e-8);
    params.Add("jacobian_delta", jacobian_delta);
    Real rootfind_tol = pin->GetOrAddReal("implicit", "rootfind_tol", 1.e-12);
    params.Add("rootfind_tol", rootfind_tol);
    Real linesearch_lambda = pin->GetOrAddReal("implicit", "linesearch_lambda", 1.0);
    params.Add("linesearch_lambda", linesearch_lambda);
    int min_nonlinear_iter = pin->GetOrAddInteger("implicit", "min_nonlinear_iter", 1);
    params.Add("min_nonlinear_iter", min_nonlinear_iter);
    int max_nonlinear_iter = pin->GetOrAddInteger("implicit", "max_nonlinear_iter", 3);
    params.Add("max_nonlinear_iter", max_nonlinear_iter);
    bool use_qr = pin->GetOrAddBoolean("implicit", "use_qr", true);
    params.Add("use_qr", use_qr);

    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);

    // TODO some way to denote non-converged zones?  impflag or something?

    // When using this package we'll need to distinguish implicitly and explicitly-updated variables
    // All independent variables should be marked one or the other when this package is in use
    MetadataFlag isImplicit = Metadata::AllocateNewFlag("Implicit");
    params.Add("ImplicitFlag", isImplicit);
    MetadataFlag isExplicit = Metadata::AllocateNewFlag("Explicit");
    params.Add("ExplicitFlag", isExplicit);

    // Anything we need to run from this package on callbacks
    // Maybe a post-step L2 or flag count or similar
    // pkg->PostFillDerivedBlock = Implicit::PostFillDerivedBlock;
    // pkg->PostStepDiagnosticsMesh = Implicit::PostStepDiagnostics;

    Flag("Initialized");
    return pkg;
}

#if ENABLE_IMPLICIT

// Implicit nonlinear solve requires several linear solves per-zone
// Use Kokkos-kernels QR decomposition & triangular solve, they're fast.
#include <batched/dense/KokkosBatched_LU_Decl.hpp>
#include <batched/dense/KokkosBatched_QR_Decl.hpp>
#include <batched/dense/KokkosBatched_ApplyQ_Decl.hpp>
#include <batched/dense/KokkosBatched_Trsv_Decl.hpp>

TaskStatus Implicit::Step(MeshData<Real> *md_full_step_init, MeshData<Real> *md_sub_step_init, MeshData<Real> *md_flux_src,
                MeshData<Real> *md_solver, const Real& dt)
{
    Flag(md_full_step_init, "Implicit Iteration start, full step");
    Flag(md_sub_step_init, "Implicit Iteration start, sub step");
    Flag(md_flux_src, "Implicit Iteration start, divF and sources");
    auto pmb_full_step_init = md_full_step_init->GetBlockData(0)->GetBlockPointer();
    auto pmb_sub_step_init  = md_sub_step_init->GetBlockData(0)->GetBlockPointer();

    // Parameters
    const auto& implicit_par = pmb_full_step_init->packages.Get("Implicit")->AllParams();
    const int iter_min       = implicit_par.Get<int>("min_nonlinear_iter");
    const int iter_max       = implicit_par.Get<int>("max_nonlinear_iter");
    const Real lambda        = implicit_par.Get<Real>("linesearch_lambda");
    const Real delta         = implicit_par.Get<Real>("jacobian_delta");
    const Real rootfind_tol  = implicit_par.Get<Real>("rootfind_tol");
    const bool use_qr        = implicit_par.Get<bool>("use_qr");
    const int verbose       = implicit_par.Get<int>("verbose");
    const Real gam           = pmb_full_step_init->packages.Get("GRMHD")->Param<Real>("gamma");
    // Misc other constants for inside the kernel
    const bool am_rank0 = MPIRank0();
    const Real tiny(SMALL), alpha(1.0);

    // We need two sets of emhd_params because we need the relaxation scale
    // at the same state in the implicit source terms
    EMHD_parameters emhd_params_full_step_init, emhd_params_sub_step_init;
    if (pmb_sub_step_init->packages.AllPackages().count("EMHD")) {
        const auto& pars_full_step_init = pmb_full_step_init->packages.Get("EMHD")->AllParams();
        const auto& pars_sub_step_init  = pmb_sub_step_init->packages.Get("EMHD")->AllParams();
        emhd_params_full_step_init      = pars_full_step_init.Get<EMHD_parameters>("emhd_params");
        emhd_params_sub_step_init       = pars_sub_step_init.Get<EMHD_parameters>("emhd_params");
    }

    // I don't normally do this, but we *really* care about variable ordering here.
    // The implicit variables need to be first, so we know how to iterate over just them to fill
    // just the residual & Jacobian we care about, which makes the solve much faster.
    // This strategy is ugly but potentially gives us complete control,
    // in case Kokkos's un-pivoted LU proves problematic
    MetadataFlag isPrimitive = pmb_sub_step_init->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    auto& mbd_full_step_init  = md_full_step_init->GetBlockData(0); // MeshBlockData object, more member functions
    auto ordered_prims = get_ordered_names(mbd_full_step_init.get(), isPrimitive);
    auto ordered_cons  = get_ordered_names(mbd_full_step_init.get(), Metadata::Conserved);
    //std::cerr << "Ordered prims:"; for(auto prim: ordered_prims) std::cerr << " " << prim; std::cerr << std::endl;
    //std::cerr << "Ordered cons:"; for(auto con: ordered_cons) std::cerr << " " << con; std::cerr << std::endl;

    // Initial state.  Also mapping template
    PackIndexMap prims_map, cons_map;
    auto& P_full_step_init_all = md_full_step_init->PackVariables(ordered_prims, prims_map);
    auto& U_full_step_init_all = md_full_step_init->PackVariables(ordered_cons, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    // Current sub-step starting state.
    auto& P_sub_step_init_all = md_sub_step_init->PackVariables(ordered_prims);
    auto& U_sub_step_init_all = md_sub_step_init->PackVariables(ordered_cons);
    // Flux divergence plus explicit source terms. This is what we'd be adding.
    auto& flux_src_all = md_flux_src->PackVariables(ordered_cons);
    // Guess at initial state. We update only the implicit primitive vars
    auto& P_solver_all = md_solver->PackVariables(ordered_prims);

    // Sizes and scratchpads
    const int nblock = U_full_step_init_all.GetDim(5);
    const int nvar   = U_full_step_init_all.GetDim(4);
    // Get number of implicit variables
    auto implicit_vars = get_ordered_names(mbd_full_step_init.get(), isPrimitive, true);
    PackIndexMap implicit_prims_map;
    auto& P_full_step_init_implicit = md_full_step_init->PackVariables(implicit_vars, implicit_prims_map);
    const int nfvar = P_full_step_init_implicit.GetDim(4);

    auto bounds  = pmb_sub_step_init->cellbounds;
    const int n1 = bounds.ncellsi(IndexDomain::entire);
    const int n2 = bounds.ncellsj(IndexDomain::entire);
    const int n3 = bounds.ncellsk(IndexDomain::entire);

    // RETURN if there aren't any implicit variables to evolve
    //std::cerr << "Solve size " << nfvar << " on prim size " << nvar << std::endl;
    if (nfvar == 0) return TaskStatus::complete;

    // The norm of the residual.  We store this to avoid the main kernel
    // also being a 2-stage reduction, which is complex and sucks.
    // TODO keep this around as a field?
    ParArray4D<Real> norm_all("norm_all", nblock, n3, n2, n1);

    // Get meshblock array bounds from Parthenon
    const IndexDomain domain = IndexDomain::interior;
    const IndexRange ib      = bounds.GetBoundsI(domain);
    const IndexRange jb      = bounds.GetBoundsJ(domain);
    const IndexRange kb      = bounds.GetBoundsK(domain);
    const IndexRange block   = IndexRange{0, nblock - 1};

    // Allocate scratch space
    // It is impossible to declare runtime-sized arrays in CUDA
    // of e.g. length var[nvar] (recall nvar can change at runtime in KHARMA)
    // Instead we copy to scratch!
    // This allows flexibility in structuring the kernel, and the results can be sliced
    // to avoid a bunch of indices in all the device-side operations
    // See grmhd_functions.hpp for the other approach with overloads
    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    const size_t var_size_in_bytes    = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);
    const size_t fvar_size_in_bytes   = parthenon::ScratchPad2D<Real>::shmem_size(nfvar, n1);
    const size_t tensor_size_in_bytes = parthenon::ScratchPad3D<Real>::shmem_size(nfvar, nfvar, n1);
    // Allocate enough to cache:
    // jacobian (2D)
    // residual, deltaP (implicit only)
    // P_full_step_init/U_full_step_init, P_sub_step_init/U_sub_step_init, divF_src, P_solver, dU_implicit, two temps (all vars)
    const size_t total_scratch_bytes = tensor_size_in_bytes + (4) * fvar_size_in_bytes + (10) * var_size_in_bytes;

    // Iterate.  This loop is outside the kokkos kernel in order to print max_norm
    // There are generally a low and similar number of iterations between
    // different zones, so probably acceptable speed loss.
    for (int iter=1; iter <= iter_max; ++iter) {
        // Flags per iter, since debugging here will be rampant
        Flag(md_solver, "Implicit Iteration:");

        parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "implicit_solve", pmb_sub_step_init->exec_space,
            total_scratch_bytes, scratch_level, block.s, block.e, kb.s, kb.e, jb.s, jb.e,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
                const auto& G = U_full_step_init_all.GetCoords(b);
                // Scratchpads for implicit vars
                ScratchPad3D<Real> jacobian_s(member.team_scratch(scratch_level), nfvar, nfvar, n1);
                ScratchPad2D<Real> residual_s(member.team_scratch(scratch_level), nfvar, n1);
                ScratchPad2D<Real> delta_prim_s(member.team_scratch(scratch_level), nfvar, n1);
                ScratchPad2D<Real> trans_s(member.team_scratch(scratch_level), nfvar, n1);
                ScratchPad2D<Real> work_s(member.team_scratch(scratch_level), nfvar, n1);
                // Scratchpads for all vars
                ScratchPad2D<Real> dU_implicit_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> tmp1_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> tmp2_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> tmp3_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> P_full_step_init_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> U_full_step_init_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> P_sub_step_init_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> U_sub_step_init_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> flux_src_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> P_solver_s(member.team_scratch(scratch_level), nvar, n1);

                // Copy some file contents to scratchpads, so we can slice them
                PLOOP {
                    parthenon::par_for_inner(member, ib.s, ib.e,
                        [&](const int& i) {
                            P_full_step_init_s(ip, i) = P_full_step_init_all(b)(ip, k, j, i);
                            U_full_step_init_s(ip, i) = U_full_step_init_all(b)(ip, k, j, i);
                            P_sub_step_init_s(ip, i)  = P_sub_step_init_all(b)(ip, k, j, i);
                            U_sub_step_init_s(ip, i)  = U_sub_step_init_all(b)(ip, k, j, i);
                            flux_src_s(ip, i) = flux_src_all(b)(ip, k, j, i);
                            P_solver_s(ip, i) = P_solver_all(b)(ip, k, j, i);
                            dU_implicit_s(ip, i) = 0.;
                        }
                    );
                }
                member.team_barrier();

                // Copy in the guess or current solution
                // Note this replaces the implicit portion of P_solver_s --
                // any explicit portion was initialized above
                // FLOOP { // Loop over just the implicit "fluid" portion of primitive vars
                //     parthenon::par_for_inner(member, ib.s, ib.e,
                //         [&](const int& i) {
                //             P_solver_s(ip, i) = P_solver_all(b)(ip, k, j, i);
                //         }
                //     );
                // }
                // member.team_barrier();

                parthenon::par_for_inner(member, ib.s, ib.e,
                    [&](const int& i) {
                        // Lots of slicing.  This still ends up faster & cleaner than alternatives I tried
                        auto P_full_step_init = Kokkos::subview(P_full_step_init_s, Kokkos::ALL(), i);
                        auto U_full_step_init = Kokkos::subview(U_full_step_init_s, Kokkos::ALL(), i);
                        auto P_sub_step_init  = Kokkos::subview(P_sub_step_init_s, Kokkos::ALL(), i);
                        auto U_sub_step_init  = Kokkos::subview(U_sub_step_init_s, Kokkos::ALL(), i);
                        auto flux_src         = Kokkos::subview(flux_src_s, Kokkos::ALL(), i);
                        auto P_solver         = Kokkos::subview(P_solver_s, Kokkos::ALL(), i);
                        // Solver variables
                        auto residual   = Kokkos::subview(residual_s, Kokkos::ALL(), i);
                        auto jacobian   = Kokkos::subview(jacobian_s, Kokkos::ALL(), Kokkos::ALL(), i);
                        auto delta_prim = Kokkos::subview(delta_prim_s, Kokkos::ALL(), i);
                        auto trans = Kokkos::subview(trans_s, Kokkos::ALL(), i);
                        auto work = Kokkos::subview(work_s, Kokkos::ALL(), i);
                        // Temporaries
                        auto tmp1  = Kokkos::subview(tmp1_s, Kokkos::ALL(), i);
                        auto tmp2  = Kokkos::subview(tmp2_s, Kokkos::ALL(), i);
                        auto tmp3  = Kokkos::subview(tmp3_s, Kokkos::ALL(), i);
                        // Implicit sources at starting state
                        auto dU_implicit = Kokkos::subview(dU_implicit_s, Kokkos::ALL(), i);
                        if (m_p.Q >= 0) {
                            EMHD::implicit_sources(G, P_full_step_init, P_sub_step_init, m_p, gam, k, j, i, emhd_params_sub_step_init, 
                                                dU_implicit(m_u.Q), dU_implicit(m_u.DP));
                        }

                        // Jacobian calculation
                        // Requires calculating the residual anyway, so we grab it here
                        calc_jacobian(G, P_solver, P_full_step_init, U_full_step_init, P_sub_step_init, 
                                    flux_src, dU_implicit, tmp1, tmp2, tmp3, m_p, m_u, emhd_params_full_step_init,
                                    emhd_params_sub_step_init, nvar, nfvar, k, j, i, delta, gam, dt, jacobian, residual);
                        // Solve against the negative residual
                        FLOOP delta_prim(ip) = -residual(ip);

                        // if (am_rank0 && b == 0 && i == 11 && j == 11 && k == kb.s) {
                        //     printf("Variable ordering: rho %d uu %d u1 %d B1 %d q %d dP %d\n",
                        //             m_p.RHO, m_p.UU, m_p.U1, m_p.B1, m_p.Q, m_p.DP);
                        //     printf("Variable ordering: rho %d uu %d u1 %d B1 %d q %d dP %d\n",
                        //             m_u.RHO, m_u.UU, m_u.U1, m_u.B1, m_u.Q, m_u.DP);
                        //     printf("P_solver: "); PLOOP printf("%6.5e ", P_solver(ip)); printf("\n");
                        //     printf("Pi: "); PLOOP printf("%6.5e ", P_full_step_init(ip)); printf("\n");
                        //     printf("Ui: "); PLOOP printf("%6.5e ", U_full_step_init(ip)); printf("\n");
                        //     printf("Ps: "); PLOOP printf("%6.5e ", P_sub_step_init(ip)); printf("\n");
                        //     printf("Us: "); PLOOP printf("%6.5e ", U_sub_step_init(ip)); printf("\n");
                        //     printf("dUdt: "); PLOOP printf("%6.5e ", dU_implicit(ip)); printf("\n");
                        //     printf("Initial Jacobian:\n"); for (int jp=0; jp<nvar; ++jp) {PLOOP printf("%6.5e\t", jacobian(jp,ip)); printf("\n");}
                        //     printf("Initial residual: "); PLOOP printf("%6.5e ", residual(ip)); printf("\n");
                        //     printf("Initial delta_prim: "); PLOOP printf("%6.5e ", delta_prim(ip)); printf("\n");
                        // }

                        if (use_qr) {
                            // Linear solve by QR decomposition
                            KokkosBatched::SerialQR<KokkosBatched::Algo::QR::Unblocked>::invoke(jacobian, trans, work);
                            KokkosBatched::SerialApplyQ<KokkosBatched::Side::Left, KokkosBatched::Trans::Transpose,
                                                        KokkosBatched::Algo::ApplyQ::Unblocked>
                            ::invoke(jacobian, trans, delta_prim, work);
                        } else {
                            KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Unblocked>::invoke(jacobian, tiny);
                        }
                        KokkosBatched::SerialTrsv<KokkosBatched::Uplo::Upper, KokkosBatched::Trans::NoTranspose, 
                                                  KokkosBatched::Diag::NonUnit, KokkosBatched::Algo::Trsv::Unblocked>
                        ::invoke(alpha, jacobian, delta_prim);

                        // Update the guess.  For now lambda == 1, choose on the fly?
                        FLOOP P_solver(ip) += lambda * delta_prim(ip);

                        calc_residual(G, P_solver, P_full_step_init, U_full_step_init, P_sub_step_init, flux_src, dU_implicit, tmp3,
                                      m_p, m_u, emhd_params_full_step_init, emhd_params_sub_step_init, nfvar, k, j, i, gam, dt, residual);

                        // if (am_rank0 && b == 0 && i == 11 && j == 11 && k == kb.s) {
                        //     printf("Variable ordering: rho %d uu %d u1 %d B1 %d q %d dP %d\n",
                        //             m_p.RHO, m_p.UU, m_p.U1, m_p.B1, m_p.Q, m_p.DP);
                        //     printf("Final residual: "); PLOOP printf("%6.5e ", residual(ip)); printf("\n");
                        //     printf("Final delta_prim: "); PLOOP printf("%6.5e ", delta_prim(ip)); printf("\n");
                        //     printf("Final P_solver: "); PLOOP printf("%6.5e ", P_solver(ip)); printf("\n");
                        // }

                        // Store for maximum/output
                        // I would be tempted to store the whole residual, but it's of variable size
                        norm_all(b, k , j, i) = 0;
                        FLOOP norm_all(b, k, j, i) += residual(ip)*residual(ip);
                        norm_all(b, k, j, i) = m::sqrt(norm_all(b, k, j, i)); // TODO faster to scratch cache & copy?
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

        // If we need to print or exit on the max norm...
        if (iter >= iter_min || verbose >= 1) {
            // Take the maximum L2 norm on this rank
            static AllReduce<Real> max_norm;
            Kokkos::Max<Real> norm_max(max_norm.val);
            pmb_sub_step_init->par_reduce("max_norm", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                KOKKOS_LAMBDA_MESH_3D_REDUCE {
                    if (norm_all(b, k, j, i) > local_result) local_result = norm_all(b, k, j, i);
                }
            , norm_max);
            // Then MPI AllReduce to copy the global max to every rank
            max_norm.StartReduce(MPI_MAX);
            while (max_norm.CheckReduce() == TaskStatus::incomplete);
            if (verbose >= 1 && MPIRank0()) printf("Iteration %d max L2 norm: %g\n", iter, max_norm.val);
            // Break if it's less than the total tolerance we set.  TODO per-zone version of this?
            if (iter >= iter_min && max_norm.val < rootfind_tol) break;
        }
    }

    Flag(md_solver, "Implicit Iteration: final");

    return TaskStatus::complete;

}

#else

TaskStatus Implicit::Step(MeshData<Real> *md_full_step_init, MeshData<Real> *md_sub_step_init, MeshData<Real> *md_flux_src,
                MeshData<Real> *md_solver, const Real& dt)
{
    Flag("Dummy implicit solve");
    auto pmb_sub_step_init  = md_sub_step_init->GetBlockData(0)->GetBlockPointer();

    MetadataFlag isPrimitive = pmb_sub_step_init->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    auto& mbd_full_step_init  = md_full_step_init->GetBlockData(0); // MeshBlockData object, more member functions

    // Get number of variables
    auto ordered_cons  = get_ordered_names(mbd_full_step_init.get(), Metadata::Conserved);
    PackIndexMap cons_map;
    auto& U_full_step_init_all = md_full_step_init->PackVariables(ordered_cons, cons_map);
    const int nvar   = U_full_step_init_all.GetDim(4);

    // Get number of implicit variables
    auto implicit_vars = get_ordered_names(mbd_full_step_init.get(), isPrimitive, true);
    PackIndexMap implicit_prims_map;
    auto& P_full_step_init_implicit = md_full_step_init->PackVariables(implicit_vars, implicit_prims_map);
    const int nfvar = P_full_step_init_implicit.GetDim(4);

    // RETURN if there aren't any implicit variables to evolve
    //std::cerr << "Solve size " << nfvar << " on prim size " << nvar << std::endl;
    if (nfvar == 0) {
        return TaskStatus::complete;
    } else {
        throw std::runtime_error("Cannot evolve variables implicitly: KHARMA was compiled without implicit solver!");
    }
    Flag("End dummy implicit solve");
}

#endif
