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
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived".
 * Also applies floors to the calculated primitives, and fixes up any inversion errors
 * 
 * Defaults to entire domain, as the KHARMA algorithm relies on applying UtoP over ghost zones.
 * 
 * input: Conserved B = sqrt(-gdet) * B^i
 * output: Primitive B = B^i
 */
void UtoP(MeshData<Real> *md, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerivedMesh(MeshData<Real> *md) { UtoP(md); }
inline TaskStatus FillDerivedMeshTask(MeshData<Real> *md) { UtoP(md); return TaskStatus::complete; }
void UtoP(MeshBlockData<Real> *md, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerivedBlock(MeshBlockData<Real> *rc) { UtoP(rc); }
inline TaskStatus FillDerivedBlockTask(MeshBlockData<Real> *rc) { UtoP(rc); return TaskStatus::complete; }

/**
 * Inverse of above. Generally only for initialization.
 */
void PtoU(MeshBlockData<Real> *md, IndexDomain domain=IndexDomain::interior, bool coarse=false);

/**
 * Modify the B field fluxes to take a constrained-transport step as in Toth (2000)
 */
TaskStatus FluxCT(MeshData<Real> *md);

/**
 * Modify the B field fluxes just beyond the polar boundary so as to ensure no flux through it,
 * after applying FluxCT
 */
TaskStatus FixPolarFlux(MeshData<Real> *md);

/**
 * Task combining the above two (polar fix and FluxCT) for simplicity
 */
TaskStatus TransportB(MeshData<Real> *md);

/**
 * Calculate maximum corner-centered divergence of magnetic field,
 * to check it is being preserved ~=0
 * Used as a Parthenon History function, so must take exactly the
 * listed arguments
 */
double MaxDivB(MeshData<Real> *md);

/**
 * Returns the global maximum value, rather than the maximum over this rank's MeshData
 */
double GlobalMaxDivB(MeshData<Real> *md);

/**
 * Clean the magnetic field divergence via successive over-relaxation
 * Currently only used when resizing inputs.
 * TODO option to sprinkle into updates every N steps
 */
void CleanupDivergence(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::interior, bool coarse=false);

/**
 * Diagnostics printed/computed after each step
 * Currently just max divB
 */
TaskStatus PrintGlobalMaxDivB(MeshData<Real> *md);
inline TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
    { return PrintGlobalMaxDivB(md); }
// Block version; unused now, kept for future fiascos
TaskStatus PrintMaxBlockDivB(MeshBlockData<Real> *rc, bool prims, std::string tag);

/**
 * Fill fields which are calculated only for output to file
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);
/**
 * Fill field "name" with divB
 */
void CalcDivB(MeshData<Real> *md, std::string divb_field_name="divB");

/**
 * ND divergence, averaging to cell corners
 * TODO likely better templated, as with all ND stuff
 */
template<typename Global>
KOKKOS_INLINE_FUNCTION double corner_div(const GRCoordinates& G, const Global& B_U, const int& b,
                                         const int& k, const int& j, const int& i, const bool& do_3D, const bool& do_2D=true)
{
    const double norm = (do_2D) ? ((do_3D) ? 0.25 : 0.5) : 1.;
    // 1D divergence
    double term1 = B_U(b, V1, k, j, i) - B_U(b, V1, k, j, i-1);
    double term2 = 0.;
    double term3 = 0.;
    if (do_2D) {
        // 2D divergence, averaging to corners
        term1 +=   B_U(b, V1, k, j-1, i) - B_U(b, V1, k, j-1, i-1);
        term2 +=   B_U(b, V2, k, j, i)   + B_U(b, V2, k, j, i-1)
                        - B_U(b, V2, k, j-1, i) - B_U(b, V2, k, j-1, i-1);
        term3 += 0.;
    }
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
    return norm*term1/G.dx1v(i) + norm*term2/G.dx2v(j) + norm*term3/G.dx3v(k);
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
    B1 = norm*term1/G.dx1v(i);
    B2 = norm*term2/G.dx2v(j);
    B3 = norm*term3/G.dx3v(k);
}

}
