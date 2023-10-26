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

#include <memory>

/**
 * 
 * This physics package implements B field transport with Flux-CT (Toth 2000)
 *
 * This requires only the values at cell centers
 * 
 * This implementation includes conversion from "primitive" to "conserved" B and back
 */
namespace B_FluxCT {
/**
 * Declare fields, initialize (few) parameters
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

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
void MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse=false);

/**
 * Reverse of the above.  Only used alone during initialization.
 * Generally, use Flux::BlockPtoU or Flux::BlockPtoUExceptMHD.
 */
void BlockPtoU(MeshBlockData<Real> *md, IndexDomain domain, bool coarse=false);

/**
 * All flux corrections required by this package
 */
void FixFlux(MeshData<Real> *md);
/**
 * Modify the B field fluxes to take a constrained-transport step as in Toth (2000)
 */
void FluxCT(MeshData<Real> *md);
/**
 * Modify the B field fluxes just beyond the polar (or radial) boundary so as to
 * ensure no flux through the boundary after applying FluxCT
 */
void FixBoundaryFlux(MeshData<Real> *md, IndexDomain domain, bool coarse);

/**
 * Alternate B field fix for X1 boundary, keeps zero divergence while permitting flux
 * through the boundary, at the cost of a short non-local solve.
 */
// added by Hyerin
TaskStatus FixX1Flux(MeshData<Real> *md);

/**
 * Calculate maximum corner-centered divergence of magnetic field,
 * to check it is being preserved ~=0
 * Used as a Parthenon History function, so must take exactly the
 * listed arguments
 */
double MaxDivB(MeshData<Real> *md);

/**
 * Returns the global maximum value, rather than the maximum over this rank's MeshData
 * 
 * By default, only returns the correct value on rank 0 for printing
 */
double GlobalMaxDivB(MeshData<Real> *md, bool all_reduce=false);

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
    auto& params = md->GetMeshPointer()->packages.Get("B_FluxCT")->AllParams();
    return PrintGlobalMaxDivB(md, params.Get<bool>("kill_on_large_divb"));
}

/**
 * Fill fields which are calculated only for output to file, i.e., divB
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);
/**
 * Fill the field 'divb_field_name' with divB
 */
void CalcDivB(MeshData<Real> *md, std::string divb_field_name="divB");

inline Real ReducePhi0(MeshData<Real> *md)
{
    return Reductions::EHReduction<Reductions::Var::phi, Real>(md, UserHistoryOperation::sum, 0);
}
inline Real ReducePhi5(MeshData<Real> *md)
{
    return Reductions::EHReduction<Reductions::Var::phi, Real>(md, UserHistoryOperation::sum, 5);
}

/**
 * ND divergence, averaging to cell corners
 * TODO likely better templated, as with all ND stuff
 */
template<typename Global>
KOKKOS_INLINE_FUNCTION double corner_div(const GRCoordinates& G, const Global& B_U, const int& b,
                                         const int& k, const int& j, const int& i, const bool& do_3D)
{
    const double norm = (do_3D) ? 0.25 : 0.5;
    // 2D divergence, averaging to corners
    double term1 = B_U(b, V1, k, j, i)   - B_U(b, V1, k, j, i-1) +
                   B_U(b, V1, k, j-1, i) - B_U(b, V1, k, j-1, i-1);
    double term2 = B_U(b, V2, k, j, i)   - B_U(b, V2, k, j-1, i) +
                   B_U(b, V2, k, j, i-1) - B_U(b, V2, k, j-1, i-1);
    double term3 = 0.;
    if (do_3D) {
        // Average to corners in 3D, add 3rd flux
        term1 +=  B_U(b, V1, k-1, j, i)   + B_U(b, V1, k-1, j-1, i)
                - B_U(b, V1, k-1, j, i-1) - B_U(b, V1, k-1, j-1, i-1);
        term2 +=  B_U(b, V2, k-1, j, i)   + B_U(b, V2, k-1, j, i-1)
                - B_U(b, V2, k-1, j-1, i) - B_U(b, V2, k-1, j-1, i-1);
        term3 =   B_U(b, V3, k, j, i)     + B_U(b, V3, k, j-1, i)
                + B_U(b, V3, k, j, i-1)   + B_U(b, V3, k, j-1, i-1)
                - B_U(b, V3, k-1, j, i)   - B_U(b, V3, k-1, j-1, i)
                - B_U(b, V3, k-1, j, i-1) - B_U(b, V3, k-1, j-1, i-1);
    }
    return norm*term1/G.Dxc<1>(i) + norm*term2/G.Dxc<2>(j) + norm*term3/G.Dxc<3>(k);
}
template<typename Global>
KOKKOS_INLINE_FUNCTION double corner_div(const GRCoordinates& G, const Global& P, const VarMap& m_p, 
                                         const int& b, const int& k, const int& j, const int& i,
                                         const bool& do_3D)
{
    const double norm = (do_3D) ? 0.25 : 0.5;
    // 2D divergence, averaging to corners
    double term1 = P(b, m_p.B1, k, j, i)   - P(b, m_p.B1, k, j, i-1) +
                   P(b, m_p.B1, k, j-1, i) - P(b, m_p.B1, k, j-1, i-1);
    double term2 = P(b, m_p.B2, k, j, i)   - P(b, m_p.B2, k, j-1, i) +
                   P(b, m_p.B2, k, j, i-1) - P(b, m_p.B2, k, j-1, i-1);
    double term3 = 0.;
    if (do_3D) {
        // Average to corners in 3D, add 3rd flux
        term1 +=  P(b, m_p.B1, k-1, j, i)   + P(b, m_p.B1, k-1, j-1, i)
                - P(b, m_p.B1, k-1, j, i-1) - P(b, m_p.B1, k-1, j-1, i-1);
        term2 +=  P(b, m_p.B2, k-1, j, i)   + P(b, m_p.B2, k-1, j, i-1)
                - P(b, m_p.B2, k-1, j-1, i) - P(b, m_p.B2, k-1, j-1, i-1);
        term3 =   P(b, m_p.B3, k, j, i)     + P(b, m_p.B3, k, j-1, i)
                + P(b, m_p.B3, k, j, i-1)   + P(b, m_p.B3, k, j-1, i-1)
                - P(b, m_p.B3, k-1, j, i)   - P(b, m_p.B3, k-1, j-1, i)
                - P(b, m_p.B3, k-1, j, i-1) - P(b, m_p.B3, k-1, j-1, i-1);
    }
    return norm*term1/G.Dxc<1>(i) + norm*term2/G.Dxc<2>(j) + norm*term3/G.Dxc<3>(k);
}

/**
 * 2D or 3D gradient, averaging to cell centers from corners.
 * Note this is forward-difference, while previous def is backward
 */
template<typename Global>
KOKKOS_INLINE_FUNCTION void center_grad(const GRCoordinates& G, const Global& P, const int& b,
                                          const int& k, const int& j, const int& i, const bool& do_3D,
                                          double& B1, double& B2, double& B3)
{
    const double norm = (do_3D) ? 0.25 : 0.5;
    // 2D divergence, averaging to corners
    double term1 =  P(b, 0, k, j+1, i+1) + P(b, 0, k, j, i+1)
                  - P(b, 0, k, j+1, i)   - P(b, 0, k, j, i);
    double term2 =  P(b, 0, k, j+1, i+1) + P(b, 0, k, j+1, i)
                  - P(b, 0, k, j, i+1)   - P(b, 0, k, j, i);
    double term3 = 0.;
    if (do_3D) {
        // Average to corners in 3D, add 3rd flux
        term1 += P(b, 0, k+1, j+1, i+1) + P(b, 0, k+1, j, i+1)
               - P(b, 0, k+1, j+1, i)   - P(b, 0, k+1, j, i);
        term2 += P(b, 0, k+1, j+1, i+1) + P(b, 0, k+1, j+1, i)
               - P(b, 0, k+1, j, i+1)   - P(b, 0, k+1, j, i);
        term3 =  P(b, 0, k+1, j+1, i+1) + P(b, 0, k+1, j, i+1)
               + P(b, 0, k+1, j+1, i)   + P(b, 0, k+1, j, i)
               - P(b, 0, k, j+1, i+1)   - P(b, 0, k, j, i+1)
               - P(b, 0, k, j+1, i)     - P(b, 0, k, j, i);
    }
    B1 = norm*term1/G.Dxc<1>(i);
    B2 = norm*term2/G.Dxc<2>(j);
    B3 = norm*term3/G.Dxc<3>(k);
}

KOKKOS_INLINE_FUNCTION void averaged_curl_3D(const GRCoordinates& G, const GridVector& A, const GridVector& B_U,
                                             const int& k, const int& j, const int& i)
{
    // Take a flux-ct step from the corner potentials.
    // This needs to be 3D because post-tilt A may not point in the phi direction only

    // A3,2 derivative
    const Real A3c2f = (A(V3, k, j + 1, i)     + A(V3, k, j + 1, i + 1) + 
                        A(V3, k + 1, j + 1, i) + A(V3, k + 1, j + 1, i + 1)) / 4;
    const Real A3c2b = (A(V3, k, j, i)     + A(V3, k, j, i + 1) +
                        A(V3, k + 1, j, i) + A(V3, k + 1, j, i + 1)) / 4;
    // A2,3 derivative
    const Real A2c3f = (A(V2, k + 1, j, i)     + A(V2, k + 1, j, i + 1) +
                        A(V2, k + 1, j + 1, i) + A(V2, k + 1, j + 1, i + 1)) / 4;
    const Real A2c3b = (A(V2, k, j, i)     + A(V2, k, j, i + 1) +
                        A(V2, k, j + 1, i) + A(V2, k, j + 1, i + 1)) / 4;
    B_U(V1, k, j, i) = (A3c2f - A3c2b) / G.Dxc<2>(j) - (A2c3f - A2c3b) / G.Dxc<3>(k);

    // A1,3 derivative
    const Real A1c3f = (A(V1, k + 1, j, i)     + A(V1, k + 1, j, i + 1) + 
                        A(V1, k + 1, j + 1, i) + A(V1, k + 1, j + 1, i + 1)) / 4;
    const Real A1c3b = (A(V1, k, j, i)     + A(V1, k, j, i + 1) +
                        A(V1, k, j + 1, i) + A(V1, k, j + 1, i + 1)) / 4;
    // A3,1 derivative
    const Real A3c1f = (A(V3, k, j, i + 1)     + A(V3, k + 1, j, i + 1) +
                        A(V3, k, j + 1, i + 1) + A(V3, k + 1, j + 1, i + 1)) / 4;
    const Real A3c1b = (A(V3, k, j, i)     + A(V3, k + 1, j, i) +
                        A(V3, k, j + 1, i) + A(V3, k + 1, j + 1, i)) / 4;
    B_U(V2, k, j, i) = (A1c3f - A1c3b) / G.Dxc<3>(k) - (A3c1f - A3c1b) / G.Dxc<1>(i);

    // A2,1 derivative
    const Real A2c1f = (A(V2, k, j, i + 1)     + A(V2, k, j + 1, i + 1) + 
                        A(V2, k + 1, j, i + 1) + A(V2, k + 1, j + 1, i + 1)) / 4;
    const Real A2c1b = (A(V2, k, j, i)     + A(V2, k, j + 1, i) +
                        A(V2, k + 1, j, i) + A(V2, k + 1, j + 1, i)) / 4;
    // A1,2 derivative
    const Real A1c2f = (A(V1, k, j + 1, i)     + A(V1, k, j + 1, i + 1) +
                        A(V1, k + 1, j + 1, i) + A(V1, k + 1, j + 1, i + 1)) / 4;
    const Real A1c2b = (A(V1, k, j, i)     + A(V1, k, j, i + 1) +
                        A(V1, k + 1, j, i) + A(V1, k + 1, j, i + 1)) / 4;
    B_U(V3, k, j, i) = (A2c1f - A2c1b) / G.Dxc<1>(i) - (A1c2f - A1c2b) / G.Dxc<2>(j);
}

KOKKOS_INLINE_FUNCTION void averaged_curl_2D(const GRCoordinates& G, const GridVector& A, const GridVector& B_U,
                                             const int& k, const int& j, const int& i)
{
    // A3,2 derivative
    const Real A3c2f = (A(V3, k, j + 1, i) + A(V3, k, j + 1, i + 1)) / 2;
    const Real A3c2b = (A(V3, k, j, i)     + A(V3, k, j, i + 1)) / 2;
    B_U(V1, k, j, i) = (A3c2f - A3c2b) / G.Dxc<2>(j);

    // A3,1 derivative
    const Real A3c1f = (A(V3, k, j, i + 1) + A(V3, k, j + 1, i + 1)) / 2;
    const Real A3c1b = (A(V3, k, j, i)     + A(V3, k, j + 1, i)) / 2;
    B_U(V2, k, j, i) = - (A3c1f - A3c1b) / G.Dxc<1>(i);

    B_U(V3, k, j, i) = 0;
}

}
