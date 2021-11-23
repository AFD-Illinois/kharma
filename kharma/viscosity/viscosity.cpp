/* 
 *  File: viscosity.cpp
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
#include "viscosity.hpp"

#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

#include <parthenon/parthenon.hpp>

#include <batched/dense/KokkosBatched_LU_Decl.hpp>
#include <batched/dense/impl/KokkosBatched_LU_Serial_Impl.hpp>
#include <batched/dense/KokkosBatched_Trsv_Decl.hpp>
using namespace KokkosBatched;

using namespace parthenon;

// Used only in Howes model
#define ME (9.1093826e-28  ) // Electron mass
#define MP (1.67262171e-24 ) // Proton mass

// Do I really want to reintroduce this?
#define SMALL 1.e-20

namespace Viscosity
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    auto pkg = std::make_shared<StateDescriptor>("Viscosity");
    Params &params = pkg->AllParams();

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // Floors & fluid gamma
    // Any parameters, like above

    MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    MetadataFlag isNonideal = Metadata::AllocateNewFlag("Nonideal");
    params.Add("NonidealFlag", isNonideal);

    // General options for primitive and conserved scalar variables in KHARMA
    Metadata m_con  = Metadata({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                 Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes, isNonideal});
    Metadata m_prim = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
                  isPrimitive, isNonideal});

    // Heat conduction
    pkg->AddField("cons.q", m_con);
    pkg->AddField("prims.q", m_prim);
    // Pressure anisotropy
    pkg->AddField("cons.dP", m_con);
    pkg->AddField("prims.dP", m_prim);

    // This ensures that UtoP is called (by way of viscosity.hpp definitions)
    pkg->FillDerivedBlock = Viscosity::FillDerived;
    pkg->PostFillDerivedBlock = Viscosity::PostFillDerived;
    return pkg;
}

void UtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    FLAG("UtoP electrons");
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

    // We will need need need to copy & reorder indices before running this

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

void PostUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    // Any fixing after that... whole thing
}

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc)
{
    FLAG("Printing electron diagnostics");

    // Output any diagnostics after a step completes

    FLAG("Printed")
    return TaskStatus::complete;
}

void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // Any variables or diagnostics that should be computed especially for output to a file,
    // but which are not otherwise updated.
}

} // namespace B_FluxCT
