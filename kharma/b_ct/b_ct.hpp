/* 
 *  File: b_flux_ct.hpp
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
#pragma once

#include "decs.hpp"
#include "grmhd_functions.hpp"
#include "reductions.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

#include <memory>

/**
 * This physics package implements Constrained Transport of a split face-centered B field.
 * Any CT implementations should probably go here.
 */
namespace B_CT {
/**
 * Declare fields, initialize (few) parameters
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Seed a divergence-free magnetic field of user's choice, optionally
 * proportional to existing fluid density.
 * Updates primitive and conserved variables.
 */
TaskStatus SeedBField(MeshBlockData<Real> *rc, ParameterInput *pin);

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived".
 * Also applies floors to the calculated primitives, and fixes up any inversion errors
 * 
 * Defaults to entire domain, as the KHARMA algorithm relies on applying UtoP over ghost zones.
 * 
 * input: Conserved B = sqrt(-gdet) * B^i
 * output: Primitive B = B^i
 */
void BlockUtoP(MeshBlockData<Real> *md, IndexDomain domain, bool coarse=false);
TaskStatus MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse=false);

/**
 * Reverse of the above.  Only used alone during initialization.
 * Generally, use Flux::BlockPtoU or Flux::BlockPtoUExceptMHD.
 */
void BlockPtoU(MeshBlockData<Real> *md, IndexDomain domain, bool coarse=false);

/**
 * Replace conserved face B field components with versions calculated
 * by constrained transport.
 */
void AddSource(MeshData<Real> *md, MeshData<Real> *mdudt);

// TODO UNIFY ALL THE FOLLOWING

/**
 * Calculate maximum corner-centered divergence of magnetic field,
 * to check it is being preserved ~=0
 * Used as a Parthenon History function, so must take exactly the
 * listed arguments
 */
double MaxDivB(MeshData<Real> *md);
double BlockMaxDivB(MeshBlockData<Real> *rc);

/**
 * Returns the global maximum value, rather than the maximum over this rank's MeshData
 */
double GlobalMaxDivB(MeshData<Real> *md);

/**
 * Diagnostics printed/computed after each step
 * Currently just max divB
 */
TaskStatus PrintGlobalMaxDivB(MeshData<Real> *md, bool kill_on_large_divb=false);

/**
 * Diagnostics function should print divB, and optionally stop execution if it's large
 */
inline TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    auto& params = md->GetMeshPointer()->block_list[0]->packages.Get("B_CT")->AllParams();
    return PrintGlobalMaxDivB(md, params.Get<bool>("kill_on_large_divb"));
}

/**
 * Fill fields which are calculated only for output to file, i.e., divB
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);
/**
 * Fill field "name" with divB
 */
void CalcDivB(MeshData<Real> *md, std::string divb_field_name="divB");

// Reductions: FOR LATER
// KOKKOS_INLINE_FUNCTION Real phi(REDUCE_FUNCTION_ARGS_EH)
// {
//     // \Phi == \int |*F^1^0| * gdet * dx2 * dx3 == \int |B1| * gdet * dx2 * dx3
//     return 0.5 * m::abs(U(m_u.B1, k, j, i)); // factor of gdet already in cons.B
// }

// inline Real ReducePhi0(MeshData<Real> *md)
// {
//     return Reductions::EHReduction(md, UserHistoryOperation::sum, phi, 0);
// }
// inline Real ReducePhi5(MeshData<Real> *md)
// {
//     return Reductions::EHReduction(md, UserHistoryOperation::sum, phi, 5);
// }

// Device functions
template<typename Global>
KOKKOS_INLINE_FUNCTION Real face_div(const GRCoordinates &G, Global &v, const int &ndim, const int &k, const int &j, const int &i)
{
    Real du = (v(F1, 0, k, j, i + 1) - v(F1, 0, k, j, i));
    if (ndim > 1) {
        du += (v(F2, 0, k, j + 1, i) - v(F2, 0, k, j, i));
    }
    if (ndim > 2) {
        du += (v(F3, 0, k + 1, j, i) - v(F3, 0, k, j, i));
    }
    return du / G.CellVolume(k, j, i);
}

// KOKKOS_INLINE_FUNCTION void curl_2D(const GRCoordinates& G, const GridVector& A, const VariablePack<Real>& B_U,
//                                              const int& k, const int& j, const int& i)
// {
//     B_U(F1, 0, k, j, i) = (A(V3, k, j + 1, i) - A(V3, k, j, i)) / G.Dxc<2>(j);// A3,2 derivative
//     B_U(F2, 0, k, j, i) =-(A(V3, k, j, i + 1) - A(V3, k, j, i)) / G.Dxc<1>(i);// A3,1 derivative
//     B_U(F3, 0, k, j, i) = 0.;
// }

KOKKOS_INLINE_FUNCTION void curl_3D(const GRCoordinates& G, const GridVector& A, const VariablePack<Real>& B_U,
                                             const int& k, const int& j, const int& i)
{
    // "CT" to faces from a cell-centered potential

    B_U(F1, 0, k, j, i) = (A(V3, k, j + 1, i) - A(V3, k, j, i)) / G.Dxc<2>(j) // A3,2 derivative
                        - (A(V2, k + 1, j, i) - A(V2, k, j, i)) / G.Dxc<3>(k);// A2,3 derivative

    B_U(F2, 0, k, j, i) = (A(V1, k + 1, j, i) - A(V1, k, j, i)) / G.Dxc<3>(k) // A1,3 derivative
                        - (A(V3, k, j, i + 1) - A(V3, k, j, i)) / G.Dxc<1>(i);// A3,1 derivative

    B_U(F3, 0, k, j, i) = (A(V2, k, j, i + 1) - A(V2, k, j, i)) / G.Dxc<1>(i) // A2,1 derivative
                        - (A(V1, k, j + 1, i) - A(V1, k, j, i)) / G.Dxc<2>(j);// A1,2 derivative
}

KOKKOS_INLINE_FUNCTION void curl_2D(const GRCoordinates& G, const GridVector& A, const VariablePack<Real>& B_U,
                                             const int& k, const int& j, const int& i)
{
    B_U(F1, 0, k, j, i) = (A(V3, k, j + 1, i) - A(V3, k, j, i)) / G.Dxc<2>(j);// A3,2 derivative
    B_U(F2, 0, k, j, i) =-(A(V3, k, j, i + 1) - A(V3, k, j, i)) / G.Dxc<1>(i);// A3,1 derivative
    B_U(F3, 0, k, j, i) = 0.;
}

}
