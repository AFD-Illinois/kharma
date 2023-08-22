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
#include "matrix.hpp"
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
TaskStatus UpdateFaces(std::shared_ptr<MeshData<Real>>& md, std::shared_ptr<MeshData<Real>>& mdudt);

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

KOKKOS_INLINE_FUNCTION Real upwind_diff(const VariableFluxPack<Real>& B_U, const VariablePack<Real>& emfc, const VariablePack<Real>& uvec,
                                        const int& comp, const int& dir, const int& vdir,
                                        const int& k, const int& j, const int& i, const bool& left_deriv)
{
    // See SG09 eq 23
    // Upwind based on vel(vdir) at the left face in vdir (contact mode)
    TopologicalElement face = FaceOf(vdir);
    const Real contact_vel = uvec(face, vdir-1, k, j, i);
    // Upwind by one zone in dir
    const int i_up = (vdir == 1) ? i - 1 : i;
    const int j_up = (vdir == 2) ? j - 1 : j;
    const int k_up = (vdir == 3) ? k - 1 : k;
    // Sign for transforming the flux to EMF, based on directions
    const int emf_sign = antisym(comp-1, dir-1, vdir-1);

    // If we're actually taking the derivative at -3/4, back up which center we use,
    // and reverse the overall sign
    const int i_cent = (left_deriv && dir == 1) ? i - 1 : i;
    const int j_cent = (left_deriv && dir == 2) ? j - 1 : j;
    const int k_cent = (left_deriv && dir == 3) ? k - 1 : k;
    const int i_cent_up = (left_deriv && dir == 1) ? i_up - 1 : i_up;
    const int j_cent_up = (left_deriv && dir == 2) ? j_up - 1 : j_up;
    const int k_cent_up = (left_deriv && dir == 3) ? k_up - 1 : k_up;
    const int return_sign = (left_deriv) ? -1 : 1;


    // TODO calculate offsets once somehow?

    if (contact_vel > 0) {
        // Forward: difference at i
        return return_sign * (emfc(0, k_cent, j_cent, i_cent) - emf_sign * B_U.flux(dir, vdir-1, k, j, i));
    } else if (contact_vel < 0) {
        // Back: twice difference at i-1
        return return_sign * (emfc(0, k_cent_up, j_cent_up, i_cent_up) - emf_sign * B_U.flux(dir, vdir-1, k_up, j_up, i_up));
    } else {
        // Half and half
        return return_sign*0.5*(emfc(0, k_cent, j_cent, i_cent) - emf_sign * B_U.flux(dir, vdir-1, k, j, i) +
                    emfc(0, k_cent_up, j_cent_up, i_cent_up) - emf_sign * B_U.flux(dir, vdir-1, k_up, j_up, i_up));
    }
}

}
