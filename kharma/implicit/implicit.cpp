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
#include "fixup.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"

#include <batched/dense/KokkosBatched_LU_Decl.hpp>
#include <batched/dense/impl/KokkosBatched_LU_Serial_Impl.hpp>
#include <batched/dense/KokkosBatched_Trsv_Decl.hpp>
using namespace KokkosBatched;

namespace Implicit
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin)
{
    auto pkg = std::make_shared<StateDescriptor>("Implicit");
    Params &params = pkg->AllParams();

    // Implicit solver parameters
    Real jacobian_delta = pin->GetOrAddReal("implicit", "jacobian_delta", 4.e-8);
    params.Add("jacobian_delta", jacobian_delta);
    Real rootfind_tol = pin->GetOrAddReal("implicit", "rootfind_tol", 1.e-3);
    params.Add("rootfind_tol", rootfind_tol);
    Real linesearch_lambda = pin->GetOrAddReal("implicit", "linesearch_lambda", 1.0);
    params.Add("linesearch_lambda", linesearch_lambda);
    int max_nonlinear_iter = pin->GetOrAddInteger("implicit", "max_nonlinear_iter", 3);
    params.Add("max_nonlinear_iter", max_nonlinear_iter);

    // No field specific to implicit solving, but we keep around the residual since
    // we need to write the whole thing out anyway
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("pflag", m);

    // Anything we need to run from this package on callbacks
    // None of this will be crucial for the step
    // pkg->PostFillDerivedBlock = Implicit::PostFillDerivedBlock;
    // pkg->PostStepDiagnosticsMesh = Implicit::PostStepDiagnostics;

    return pkg;
}

TaskStatus Step(MeshData<Real> *mdi, MeshData<Real> *md0, MeshData<Real> *dudt,
                MeshData<Real> *md1, const Real& dt)
{
    Flag(mdi, "Implicit Iteration start, i");
    Flag(md0, "Implicit Iteration start, 0");
    Flag(dudt, "Implicit Iteration start, dudt");
    auto pmb0 = mdi->GetBlockData(0)->GetBlockPointer();

    const auto& implicit_par = pmb0->packages.Get("Implicit")->AllParams();
    const int iter_max = implicit_par.Get<int>("max_nonlinear_iter");
    const Real lambda = implicit_par.Get<Real>("linesearch_lambda");
    const Real delta = implicit_par.Get<Real>("jacobian_delta");
    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");


    //MetadataFlag isNonideal = pmb0->packages.Get("EMHD")->Param<MetadataFlag>("NonidealFlag");
    MetadataFlag isPrimitive = pmb0->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    // Initial state.  Also mapping template
    PackIndexMap prims_map, cons_map;
    auto& Pi_all = mdi->PackVariables(std::vector<MetadataFlag>{isPrimitive}, prims_map);
    auto& Ui_all = mdi->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    // Current sub-step starting state.
    auto& Ps_all = md0->PackVariables(std::vector<MetadataFlag>{isPrimitive});
    auto& Us_all = md0->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved});
    // Flux divergence plus explicit source terms. This is what we'd be adding 
    auto& dUdt_all = dudt->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved});
    // Desired final state.  Note this is prims only: we sync these, then run P->U on each node.
    // TODO REMEMBER TO COPY IN MD0 CONTENTS AS GUESS
    auto& P_solver_all = md1->PackVariables(std::vector<MetadataFlag>{isPrimitive});

    // Note this iterator, like all of KHARMA, requires nprim == ncons
    // TODO Maybe should enforce that at start?
    const int nblock = Ui_all.GetDim(5);
    const int nvar = Ui_all.GetDim(4);

    // Workspaces for iteration, include ghosts to match indices.
    // Probably should never need coarse/entire...
    auto bounds = pmb0->cellbounds; //coarse ? pmb0->c_cellbounds : pmb0->cellbounds;
    const int n1 = bounds.ncellsi(IndexDomain::entire);
    const int n2 = bounds.ncellsj(IndexDomain::entire);
    const int n3 = bounds.ncellsk(IndexDomain::entire);

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
    //const IndexRange vb = IndexRange{0, nvar - 1};

    // Allocate scratch space
    // It is impossible to declare runtime-sized arrays in CUDA
    // of e.g. length var[nvar] (recall nvar can change at runtime in KHARMA)
    // Instead we copy to scratch!
    // This allows flexibility in structuring the kernel, as
    // well as slicing, which in turn allows writing just *one* version of each operation!
    // Older versions of KHARMA solved this with overloads, it was a mess.  This is less mess.
    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    const size_t var_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);
    const size_t tensor_size_in_bytes = parthenon::ScratchPad3D<Real>::shmem_size(nvar, nvar, n1);
    // Allocate enough to cache:
    // jacobian (2D)
    // residual, deltaP, dUi, two temps
    // Pi/Ui, Ps/Us, dUdt, P_solver
    const size_t total_scratch_bytes = tensor_size_in_bytes + (12) * var_size_in_bytes;

    // Iterate.  This loop is outside the kokkos kernel in order to print max_norm
    // There are generally a low and similar number of iterations between
    // different zones, so probably acceptable speed loss.
    for (int iter=0; iter < iter_max; iter++) {
        // Flags per iter, since debugging here will be rampant
        Flag(md0, "Implicit Iteration: md0");
        Flag(md1, "Implicit Iteration: md1");

        parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "implicit_solve", pmb0->exec_space,
            total_scratch_bytes, scratch_level, block.s, block.e, kb.s, kb.e, jb.s, jb.e,
            KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
                const auto& G = Ui_all.GetCoords(b);
                ScratchPad3D<Real> jacobian_s(member.team_scratch(scratch_level), nvar, nvar, n1);
                ScratchPad2D<Real> residual_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> delta_prim_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> dUi_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> tmp1_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> tmp2_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> tmp3_s(member.team_scratch(scratch_level), nvar, n1);
                // Local versions of the variables
                ScratchPad2D<Real> Pi_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> Ui_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> Ps_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> Us_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> dUdt_s(member.team_scratch(scratch_level), nvar, n1);
                ScratchPad2D<Real> P_solver_s(member.team_scratch(scratch_level), nvar, n1);

                // Copy some file contents to scratchpads, so we can slice them
                PLOOP parthenon::par_for_inner(member, ib.s, ib.e,
                    [&](const int& i) {
                        Pi_s(ip, i) = Pi_all(b)(ip, k, j, i);
                        Ui_s(ip, i) = Ui_all(b)(ip, k, j, i);
                        Ps_s(ip, i) = Ps_all(b)(ip, k, j, i);
                        Us_s(ip, i) = Us_all(b)(ip, k, j, i);
                        dUdt_s(ip, i) = dUdt_all(b)(ip, k, j, i);
                        // Finally, P_solver should actually be initialized to Ps
                        if (iter == 0) {
                            P_solver_s(ip, i) = Ps_all(b)(ip, k, j, i);
                        } else {
                            P_solver_s(ip, i) = P_solver_all(b)(ip, k, j, i);
                        }
                    }
                );

                parthenon::par_for_inner(member, ib.s, ib.e,
                    [&](const int& i) {
                        // Lots of slicing.  This is still way faster & cleaner than alternatives, trust me
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
                            //emhd_implicit_sources(G, Si, dUi);
                        } else {
                            PLOOP dUi(ip) = 0;
                        }

                        // Jacobian calculation
                        // Requires calculating the residual anyway, so we grab it here
                        // (the array will eventually hold delta_prim, after the matrix solve)
                        calc_jacobian(G, P_solver, Ui, Us, dUdt, dUi, tmp1, tmp2, tmp3,
                                      m_p, m_u, nvar, j, i, delta, gam, dt, jacobian, residual);
                        // Initial delta prim is negative residual
                        PLOOP delta_prim(ip) = -residual(ip);

                        // if (am_rank0 && b == 0 && i == 8 && j == 8 && k == 8) {
                        //     printf("\nSample Jacobian and residual:");
                        //     for (int u=0; u < nvar; u++) {
                        //         printf("\n");
                        //         for (int v=0; v < nvar; v++) printf("%f ", jacobian(u, v));
                        //     }
                        //     printf("\nres:\n");
                        //     for (int u=0; u < nvar; u++) printf("%f ", delta_prim(u));
                        //     printf("\n");
                        // }

                        // Linear solve
                        // This code lightly adapted from Kokkos batched examples
                        KokkosBatched::SerialLU<Algo::LU::Unblocked>::invoke(jacobian, tiny);
                        KokkosBatched::SerialTrsv<Uplo::Lower,Trans::NoTranspose,Diag::NonUnit,Algo::Trsv::Unblocked>
                        ::invoke(alpha, jacobian, delta_prim);

                        // if (am_rank0 && b == 0 && i == 8 && j == 8 && k == 8) {
                        //     printf("\nTri Jacobian and dP:");
                        //     for (int u=0; u < nvar; u++) {
                        //         printf("\n");
                        //         for (int v=0; v < nvar; v++) printf("%f ", jacobian(u, v));
                        //     }
                        //     printf("\ndP:\n");
                        //     for (int u=0; u < nvar; u++) printf("%f ", delta_prim(u));
                        //     printf("\n");
                        // }

                        // Update the guess.  For now lambda == 1, choose on the fly?
                        PLOOP P_solver(ip) += lambda * delta_prim(ip);

                        calc_residual(G, P_solver, Ui, Us, dUdt, dUi, tmp3,
                                      m_p, m_u, nvar, j, i, gam, dt, residual);

                        // Store for maximum/output
                        // I would be tempted to store the whole residual, but it's of variable size
                        norm_all(b, k , j, i) = 0;
                        PLOOP norm_all(b, k, j, i) += pow(residual(ip), 2);
                        norm_all(b, k, j, i) = sqrt(norm_all(b, k, j, i)); // TODO faster to scratch cache & copy?

                    }
                );

                // Copy out P_solver to the existing array
                // This combo still works if P_solver is aliased to one of the other arrays!
                PLOOP parthenon::par_for_inner(member, ib.s, ib.e,
                    [&](const int& i) {
                        P_solver_all(b)(ip, k, j, i) = P_solver_s(ip, i);
                    }
                );
            }
        );

        // L2 norm maximum.
        Real max_norm;
        Kokkos::Max<Real> norm_max(max_norm);
        pmb0->par_reduce("max_norm", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA_MESH_3D_REDUCE {
                if (norm_all(b, k, j, i) > local_result) local_result = norm_all(b, k, j, i);
            }
        , norm_max);
        max_norm = MPIMax(max_norm);
        if (MPIRank0()) fprintf(stdout, "Nonlinear iter %d. Max L2 norm: %g\n", iter, max_norm);
    }

    return TaskStatus::complete;

}

} // namespace Implicit
