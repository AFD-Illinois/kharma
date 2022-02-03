/* 
 *  File: types.hpp
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

#include <parthenon/parthenon.hpp>

/**
 * Types and convenience functions
 * 
 * Anything potentially useful throughout KHARMA, but specific to it
 * (general copy/pastes from StackOverflow go in kharma_utils.hpp)
 */

// Denote reconstruction algorithms
// See reconstruction.hpp for implementations
enum ReconstructionType{donor_cell=0, linear_mc, linear_vl, ppm, mp5, weno5, weno5_lower_poles};

// Denote inversion failures (pflags). See U_to_P for status explanations
// Only thrown from function in U_to_P.hpp, see that file for meanings
enum InversionStatus{success=0, neg_input, max_iter, bad_ut, bad_gamma, neg_rho, neg_u, neg_rhou};

// Struct for derived 4-vectors at a point, usually calculated and needed together
typedef struct {
    parthenon::Real ucon[GR_DIM];
    parthenon::Real ucov[GR_DIM];
    parthenon::Real bcon[GR_DIM];
    parthenon::Real bcov[GR_DIM];
} FourVectors;

/**
 * Map of the locations of particular variables in a VariablePack
 * Used for operations conducted over all vars which must still
 * distinguish between them, e.g. fluxes.hpp
 *
 * We use this instead of the PackIndexMap, because comparing strings
 * on the device every time we need the index of a variable is slow.
 *
 * Think of this a bit like the macros in iharm2d or 3d, except
 * that since we're packing on the fly, they can't be globally
 * constant anymore.
 *
 * Note the values of any variables not present in the pack will be -1
 */
class VarMap {
    public:
        // 127 values ought to be enough for anybody
        int8_t RHO, UU, U1, U2, U3, B1, B2, B3, PSI;
        int8_t RHO_ADDED, UU_ADDED, PASSIVE;
        int8_t KTOT, K_CONSTANT, K_HOWES, K_KAWAZURA, K_WERNER, K_ROWAN, K_SHARMA;
        // Total struct size 20 bytes, < 1 vector of 4 doubles

        VarMap(parthenon::PackIndexMap& name_map, bool is_cons)
        {
            if (is_cons) {
                // HD
                RHO = name_map["cons.rho"].first;
                UU = name_map["cons.u"].first;
                U1 = name_map["cons.uvec"].first;
                // B
                B1 = name_map["cons.B"].first;
                PSI = name_map["cons.psi_cd"].first;
                // Floors
                RHO_ADDED = name_map["cons.rho_added"].first;
                UU_ADDED = name_map["cons.u_added"].first;
                // Electrons
                KTOT = name_map["cons.Ktot"].first;
                K_CONSTANT = name_map["cons.Kel_Constant"].first;
                K_HOWES = name_map["cons.Kel_Howes"].first;
                K_KAWAZURA = name_map["cons.Kel_Kawazura"].first;
                K_WERNER = name_map["cons.Kel_Werner"].first;
                K_ROWAN = name_map["cons.Kel_Rowan"].first;
                K_SHARMA = name_map["cons.Kel_Sharma"].first;
            } else {
                // HD
                RHO = name_map["prims.rho"].first;
                UU = name_map["prims.u"].first;
                U1 = name_map["prims.uvec"].first;
                // B
                B1 = name_map["prims.B"].first;
                PSI = name_map["prims.psi_cd"].first;
                // Floors (TODO cons only?)
                RHO_ADDED = name_map["prims.rho_added"].first;
                UU_ADDED = name_map["prims.u_added"].first;
                // Electrons
                KTOT = name_map["prims.Ktot"].first;
                K_CONSTANT = name_map["prims.Kel_Constant"].first;
                K_HOWES = name_map["prims.Kel_Howes"].first;
                K_KAWAZURA = name_map["prims.Kel_Kawazura"].first;
                K_WERNER = name_map["prims.Kel_Werner"].first;
                K_ROWAN = name_map["prims.Kel_Rowan"].first;
                K_SHARMA = name_map["prims.Kel_Sharma"].first;
            }
            U2 = U1 + 1;
            U3 = U1 + 2;
            B2 = B1 + 1;
            B3 = B1 + 2;
        }
};

/**
 * Functions for checking boundaries in 3D
 */
KOKKOS_INLINE_FUNCTION bool inside(const int& k, const int& j, const int& i,
                                   const IndexRange& kb, const IndexRange& jb, const IndexRange& ib)
{
    return (i >= ib.s) && (i <= ib.e) && (j >= jb.s) && (j <= jb.e) && (k >= kb.s) && (k <= kb.e);
}
KOKKOS_INLINE_FUNCTION bool outside(const int& k, const int& j, const int& i,
                                    const IndexRange& kb, const IndexRange& jb, const IndexRange& ib)
{
    return (i < ib.s) || (i > ib.e) || (j < jb.s) || (j > jb.e) || (k < kb.s) || (k > kb.e);
}

#if TRACE
inline void Flag(std::string label)
{
#pragma omp critical
    if(MPIRank0()) std::cerr << label << std::endl;
}
inline void Flag(MeshBlockData<Real> *rc, std::string label)
{
#pragma omp critical
{
    if(MPIRank0()) std::cerr << label << std::endl;
    if(0) {
        auto rhop = rc->Get("prims.rho").data.GetHostMirrorAndCopy();
        auto up = rc->Get("prims.u").data.GetHostMirrorAndCopy();
        auto uvecp = rc->Get("prims.uvec").data.GetHostMirrorAndCopy();
        auto Bp = rc->Get("prims.B").data.GetHostMirrorAndCopy();
        auto rhoc = rc->Get("cons.rho").data.GetHostMirrorAndCopy();
        auto uc = rc->Get("cons.u").data.GetHostMirrorAndCopy();
        auto uvecc = rc->Get("cons.uvec").data.GetHostMirrorAndCopy();
        auto Bu = rc->Get("cons.B").data.GetHostMirrorAndCopy();
        cerr << "P:";
        for (int j=0; j<8; j++) {
            cout << endl;
            for (int i=0; i<8; i++) {
                fprintf(stderr, "%.5g\t", uvecp(2, 0, j, i));
            }
        }
        cerr << endl << "U:";
        for (int j=0; j<8; j++) {
            cerr << endl;
            for (int i=0; i<8; i++) {
                fprintf(stderr, "%.5g\t", uvecc(2, 0, j, i));
            }
        }
        cerr << endl << endl;
    }
}
}
inline void Flag(MeshData<Real> *md, std::string label)
{
#pragma omp critical
{
    if(MPIRank0()) std::cerr << label << std::endl;
    if(0) {
        cerr << label << ":" << std::endl;
        auto rc = md->GetBlockData(0);
        auto rhop = rc->Get("prims.rho").data.GetHostMirrorAndCopy();
        auto up = rc->Get("prims.u").data.GetHostMirrorAndCopy();
        auto uvecp = rc->Get("prims.uvec").data.GetHostMirrorAndCopy();
        auto Bp = rc->Get("prims.B").data.GetHostMirrorAndCopy();
        auto rhoc = rc->Get("cons.rho").data.GetHostMirrorAndCopy();
        auto uc = rc->Get("cons.u").data.GetHostMirrorAndCopy();
        auto uvecc = rc->Get("cons.uvec").data.GetHostMirrorAndCopy();
        auto Bu = rc->Get("cons.B").data.GetHostMirrorAndCopy();
        cerr << "P:";
        for (int j=0; j<8; j++) {
            cout << endl;
            for (int i=0; i<8; i++) {
                fprintf(stderr, "%.5g\t", uvecp(2, 0, j, i));
            }
        }
        cerr << endl;
        cerr << "U:";
        for (int j=0; j<8; j++) {
            cerr << endl;
            for (int i=0; i<8; i++) {
                fprintf(stderr, "%.5g\t", uvecc(2, 0, j, i));
            }
        }
        cerr << endl << endl;
    }
}
}
#else
inline void Flag(std::string label) {}
inline void Flag(MeshBlockData<Real> *rc, std::string label) {}
inline void Flag(MeshData<Real> *md, std::string label) {}
#endif
