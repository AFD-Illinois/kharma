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

#include "kharma_driver.hpp"

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
 * Get the primitive variables, which in Parthenon's nomenclature are "derived".
 * Also applies floors to the calculated primitives, and fixes up any inversion errors
 * 
 * Defaults to entire domain, as the KHARMA algorithm relies on applying UtoP over ghost zones.
 * 
 * input: Conserved B = sqrt(-gdet) * B^i
 * output: Primitive B = B^i
 */
void BlockUtoP(MeshBlockData<Real> *mbd, IndexDomain domain, bool coarse=false);
TaskStatus MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse=false);

/**
 * Calculate the EMF around edges of faces caused by the flux of B field
 * through each face.
 */
TaskStatus CalculateEMF(MeshData<Real> *md);

/**
 * Calculate the change in magnetic field on faces for this step,
 * from the EMFs at edges.
 */
TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt);

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

KOKKOS_INLINE_FUNCTION void curl_3D(const GRCoordinates& G, const GridVector& A, const VariablePack<Real>& B_U,
                                             const int& k, const int& j, const int& i)
{
    // Take a face-ct step from the corner potentials.
    // This needs to be 3D because post-tilt A may not point in the phi direction only

    // A3,2 derivative
    const Real A3c2f = (A(V3, k, j + 1, i) + A(V3, k + 1, j + 1, i)) / 2;
    const Real A3c2b = (A(V3, k, j, i)     + A(V3, k + 1, j, i)) / 2;
    // A2,3 derivative
    const Real A2c3f = (A(V2, k + 1, j, i) + A(V2, k + 1, j + 1, i)) / 2;
    const Real A2c3b = (A(V2, k, j, i)     + A(V2, k, j + 1, i)) / 2;
    B_U(F1, 0, k, j, i) = (A3c2f - A3c2b) - (A2c3f - A2c3b);

    // A1,3 derivative
    const Real A1c3f = (A(V1, k + 1, j, i)     + A(V1, k + 1, j, i + 1)) / 2;
    const Real A1c3b = (A(V1, k, j, i)         + A(V1, k, j, i + 1)) / 2;
    // A3,1 derivative
    const Real A3c1f = (A(V3, k, j, i + 1)     + A(V3, k + 1, j, i + 1)) / 2;
    const Real A3c1b = (A(V3, k, j, i)         + A(V3, k + 1, j, i)) / 2;
    B_U(F2, 0, k, j, i) = (A1c3f - A1c3b) - (A3c1f - A3c1b);

    // A2,1 derivative
    const Real A2c1f = (A(V2, k, j, i + 1)     + A(V2, k, j + 1, i + 1)) / 2;
    const Real A2c1b = (A(V2, k, j, i)     + A(V2, k, j + 1, i)) / 2;
    // A1,2 derivative
    const Real A1c2f = (A(V1, k, j + 1, i)     + A(V1, k, j + 1, i + 1)) / 2;
    const Real A1c2b = (A(V1, k, j, i)     + A(V1, k, j, i + 1)) / 2;
    B_U(F3, 0, k, j, i) = (A2c1f - A2c1b) - (A1c2f - A1c2b);
}

KOKKOS_INLINE_FUNCTION void curl_2D(const GRCoordinates& G, const GridVector& A, const VariablePack<Real>& B_U,
                                    const int& k, const int& j, const int& i)
{
    // TODO why do these not need 
    // A3,2 derivative
    B_U(F1, 0, k, j, i) = (A(V3, k, j + 1, i) - A(V3, k, j, i));
    // A3,1 derivative
    B_U(F2, 0, k, j, i) = - (A(V3, k, j, i + 1) - A(V3, k, j, i));
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

// Only by formatting may the following be made even a little comprehensible.

template<int diff_face, int diff_side, int offset, int DIM>
KOKKOS_FORCEINLINE_FUNCTION Real F(const ParArrayND<Real, VariableState> &fine, int l, int m, int n, int fk, int fj, int fi)
{
    // TODO compile-time error on misuse?
    constexpr int df_is_k = 2*(diff_face == V3 && DIM > 2);
    constexpr int df_is_j = 2*(diff_face == V2 && DIM > 1);
    constexpr int df_is_i = 2*(diff_face == V1 && DIM > 0);
    constexpr int ds_is_k = (diff_side == V3 && DIM > 2);
    constexpr int ds_is_j = (diff_side == V2 && DIM > 1);
    constexpr int ds_is_i = (diff_side == V1 && DIM > 0);
    constexpr int of_is_k = (offset == V3 && DIM > 2);
    constexpr int of_is_j = (offset == V2 && DIM > 1);
    constexpr int of_is_i = (offset == V1 && DIM > 0);
    // if(fi == 10 && fj == 10 && fk == 0) {
    //     fprintf(stderr, "F facediff dir %d sidediff dirr %d off dir %d\nadding terms %d %d %d, -%d %d %d, -%d %d %d, %d %d %d\n",
    //             diff_face, diff_side, offset,
    //             df_is_i+ds_is_i+of_is_i, df_is_j+ds_is_j+of_is_j, df_is_k+ds_is_k+of_is_k,
    //             ds_is_i+of_is_i,         ds_is_j+of_is_j,         ds_is_k+of_is_k,
    //             df_is_i+of_is_i,         df_is_j+of_is_j,         df_is_k+of_is_k,
    //             of_is_i                , of_is_j                , of_is_k);
    // }
    return fine(diff_face, l, m, n, fk+df_is_k+ds_is_k+of_is_k, fj+df_is_j+ds_is_j+of_is_j, fi+df_is_i+ds_is_i+of_is_i)
         - fine(diff_face, l, m, n, fk+ds_is_k+of_is_k        , fj+ds_is_j+of_is_j        , fi+ds_is_i+of_is_i)
         - fine(diff_face, l, m, n, fk+df_is_k+of_is_k        , fj+df_is_j+of_is_j        , fi+df_is_i+of_is_i)
         + fine(diff_face, l, m, n, fk+of_is_k                , fj+of_is_j                , fi+of_is_i);
}

struct ProlongateInternalOlivares {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel && (fel == F1 || fel == F2 || fel == F3);
  }

  template <int DIM, TopologicalElement el = TopologicalElement::CC,
            TopologicalElement cel = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *,
     const ParArrayND<Real, VariableState> *pfine) {

        // Definitely exit on what we can't handle
        if constexpr (el != TE::F1 && el != TE::F2 && el != TE::F3)
            return;

        // Handle permutations "naturally."
        // Olivares et al. is fond of listing x1 versions which permute,
        // this makes translating/checking those easier
        constexpr int me = static_cast<int>(el) % 3;
        constexpr int next = (me+1) % 3;
        constexpr int third = (me+2) % 3;

        // Exit if we're computing a trivial direction
        if constexpr ((me == V3 && !(DIM > 2)) || (me == V2 && !(DIM > 1)) || (me == V1 && !(DIM > 0)))
            return;

        // Fine array, indices
        auto &fine = *pfine;
        const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
        const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
        const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

        // TODO can we handle this in Parthenon instead?
        if ((el == TE::F1 && fi+2 > ib.s) || (el == TE::F2 && fj+2 > jb.s) || (el == TE::F3 && fk+2 > kb.s))
            return;

        // Coefficients selecting a particular formula (see Olivares et al. 2019)
        // TODO options here. This corresponds to Cunningham, but we could have:
        // 1. differences of squares of zone dimesnions (Toth)
        // 2. heuristic based on flux difference of top vs bottom halves (Olivares)
        //constexpr Real a[3] = {0., 0., 0.};
        const Real a[3] = {(SQR(coords.Dxc<2>(j)) - SQR(coords.Dxc<3>(k)))/(SQR(coords.Dxc<2>(j)) + SQR(coords.Dxc<3>(k))),
                        (SQR(coords.Dxc<3>(k)) - SQR(coords.Dxc<1>(i)))/(SQR(coords.Dxc<3>(k)) + SQR(coords.Dxc<1>(i))),
                        (SQR(coords.Dxc<1>(i)) - SQR(coords.Dxc<2>(j)))/(SQR(coords.Dxc<1>(i)) + SQR(coords.Dxc<2>(j)))};

        // Coefficients for each term evaluating the four sub-faces
        const Real coeff[4][4] = {{3 + a[next], 1 - a[next], 3 - a[third], 1 + a[third]},
                                {3 + a[next], 1 - a[next], 1 + a[third], 3 - a[third]},
                                {1 - a[next], 3 + a[next], 3 - a[third], 1 + a[third]},
                                {1 - a[next], 3 + a[next], 1 + a[third], 3 - a[third]}};

        constexpr int diff_k = (me == V3), diff_j = (me == V2), diff_i = (me == V1);
        // if(fi == 10 && fj == 10 && fk == 0) {
        //     fprintf(stderr, "Prolongating %d %d %d EL %d, DIM %d\n", fi, fj, fk, static_cast<int>(el), DIM);
        //     fprintf(stderr, "Differencing %d %d %d\n", diff_i, diff_j, diff_k);
        // }

        // Iterate through the 4 sub-faces
        for (int elem=0; elem < 4; elem++) {
            // Make sure we can offset in other directions before doing so, though
            // TODO eliminate redundant work or template these so the compiler can?
            const int off_i = (DIM > 0) ? elem%2*(me == V2) + elem/2*(me == V3) + (me == V1) : 0;
            const int off_j = (DIM > 1) ? elem%2*(me == V3) + elem/2*(me == V1) + (me == V2) : 0;
            const int off_k = (DIM > 2) ? elem%2*(me == V1) + elem/2*(me == V2) + (me == V3) : 0;

            fine(me, l, m, n, fk+off_k, fj+off_j, fi+off_i) =
                // Average faces on either side of us in selected direction (diff), on each of the 4 sub-faces (off)
                0.5*(fine(me, l, m, n, fk+off_k-diff_k, fj+off_j-diff_j, fi+off_i-diff_i) +
                    fine(me, l, m, n, fk+off_k+diff_k, fj+off_j+diff_j, fi+off_i+diff_i)) +
                1./16*(coeff[elem][0]*F<next ,me,-1,DIM>(fine, l, m, n, fk, fj, fi) + coeff[elem][1]*F<next,me,third,DIM>(fine, l, m, n, fk, fj, fi)
                    + coeff[elem][2]*F<third,me,-1,DIM>(fine, l, m, n, fk, fj, fi) + coeff[elem][3]*F<third,me,next,DIM>(fine, l, m, n, fk, fj, fi));

            // if(fi == 10 && fj == 10 && fk == 0 && me == V1) {
            //     fprintf(stderr, "Elem %d Offset %d %d %d set %g\n", elem, off_i, off_j, off_k, fine(me, l, m, n, fk+off_k, fj+off_j, fi+off_i));
            //     fprintf(stderr, "Averaging faces %d %d %d and %d %d %d (%g & %g)\n", fi+off_i-diff_i, fj+off_j-diff_j, fk+off_k-diff_k, 
            //         fi+off_i+diff_i, fj+off_j+diff_j, fk+off_k+diff_k, 
            //         fine(me, l, m, n, fk+off_k-diff_k, fj+off_j-diff_j, fi+off_i-diff_i),
            //         fine(me, l, m, n, fk+off_k+diff_k, fj+off_j+diff_j, fi+off_i+diff_i));
            //     fprintf(stderr, "Coeffs %g %g %g %g\n", coeff[elem][0]*F<next,me,-1,DIM>(fine, l, m, n, fk, fj, fi),
            //                                             coeff[elem][1]*F<next,me,third,DIM>(fine, l, m, n, fk, fj, fi),
            //                                             coeff[elem][2]*F<third,me,-1,DIM>(fine, l, m, n, fk, fj, fi),
            //                                             coeff[elem][3]*F<third,me,next,DIM>(fine, l, m, n, fk, fj, fi));
            // }
        }
    }
};

}
