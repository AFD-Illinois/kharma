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
#include "mhd_functions.hpp"
#include "pack.hpp"

#include <batched/dense/KokkosBatched_LU_Decl.hpp>
#include <batched/dense/impl/KokkosBatched_LU_Serial_Impl.hpp>
#include <batched/dense/KokkosBatched_Trsv_Decl.hpp>
using namespace KokkosBatched;

namespace Implicit
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin)
{
    // TODO can I just build/add/use a Prescription here, rather than building one
    // before each call?
    auto pkg = std::make_shared<StateDescriptor>("Implicit");
    Params &params = pkg->AllParams();

    // Implicit solver parameters
    bool jacobian_eps = pin->GetOrAddReal("implicit", "jacobian_eps", 4.e-8);
    params.Add("jacobian_eps", jacobian_eps);
    bool rootfind_tol = pin->GetOrAddReal("implicit", "rootfind_tol", 1.e-3);
    params.Add("rootfind_tol", rootfind_tol);
    bool max_nonlinear_iter = pin->GetOrAddInteger("implicit", "max_nonlinear_iter", 1);
    params.Add("max_nonlinear_iter", max_nonlinear_iter);

    // Any fields particular to the implicit solver (NOT EGRMHD IN GENERAL)
    // Likely none...
    // see viscosity/viscosity.cpp for EGRMHD/auxiliary fields

    // Anything we need to run from this package on callbacks
    // None of this will be crucial for the step
    // pkg->PostFillDerivedBlock = Implicit::PostFillDerivedBlock;
    // pkg->PostStepDiagnosticsMesh = Implicit::PostStepDiagnostics;

    return pkg;
}

void UtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "UtoP electrons");
    auto pmb = rc->GetBlockPointer();

    MetadataFlag isNonideal = pmb->packages.Get("Viscosity")->Param<MetadataFlag>("NonidealFlag");
    MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    // No need for a "map" here, we just want everything that fits these
    auto& e_P = rc->PackVariables({isNonideal, isPrimitive});
    auto& e_U = rc->PackVariables({isNonideal, Metadata::Conserved});
    // And then the local density
    GridScalar rho_U = rc->Get("cons.rho").data;

    const auto& G = pmb->coords;

    // Get array bounds from Parthenon
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    IndexRange ib = bounds.GetBoundsI(domain);
    IndexRange jb = bounds.GetBoundsJ(domain);
    IndexRange kb = bounds.GetBoundsK(domain);

    // For speed, we will need need need to copy & reorder indices before running this

    // Begin the funky kokkos bit
    // Let's do a batched LU and Trsv!
    const Real alpha = 1, tiny = 0;
    const int ni = bounds.ncellsi(domain), nj = bounds.ncellsj(domain), nk = bounds.ncellsk(domain);
    ParArray5D<Real> AA("AA", nk, nj, ni, 7, 7);
    ParArray4D<Real> B("B", nk, nj, ni, 7);

    // Simulating some iterations
    for (int iter=0; iter < 5; iter++) {
        // Normally, when doing multiple batched operations,
        // we would need either a general solve function,
        // or two reads through the full array. Not so in Kokkos!
        // This could be faster I think -- there are versions of the inner portion
        // that cover rows at a time, by taking member objects on a Team
        // see e.g. fluxes.hpp for usage of teams
        pmb->par_for("implicit_solve", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA_3D {
                // This code lightly adapted from 
                auto A = Kokkos::subview(AA, k, j, i, Kokkos::ALL(), Kokkos::ALL());
                auto b = Kokkos::subview(B, k, j, i, Kokkos::ALL());
                /// [template]AlgoType: Unblocked, Blocked, CompatMKL
                /// [in/out]A: 2d view
                /// [in]tiny: a magnitude scalar value to avoid div/0
                KokkosBatched::SerialLU<Algo::LU::Blocked>::invoke(A, tiny);
                /// [template]UploType: indicates either upper triangular or lower triangular; Uplo::Upper, Uplo::Lower
                /// [template]TransType: transpose of A; Trans::NoTranspose, Trans::Transpose
                /// [template]DiagType: diagonals; Diag::Unit or Diag::NonUnit
                /// [template]AlgoType: Unblocked, Blocked, CompatMKL
                /// [in]alpha: scalar value
                /// [in]A: 2d view
                /// [in]b: 1d view
                KokkosBatched::SerialTrsv<Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsv::Unblocked>::invoke(alpha, A, b);
            }
        );
    }

}

} // namespace Implicit
