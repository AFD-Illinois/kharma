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

#include "kharma_package.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

using parthenon::MeshBlockData;

/**
 * Types, macros, and convenience functions
 * 
 * Anything potentially useful throughout KHARMA, but specific to it
 * (general copy/pastes from StackOverflow go in kharma_utils.hpp)
 */

// This provides a way of addressing vectors that matches
// directions, to make derivatives etc more readable
// TODO Spammy to namespace. Keep?
#define V1 0
#define V2 1
#define V3 2

// Struct for derived 4-vectors at a point, usually calculated and needed together
typedef struct {
    Real ucon[GR_DIM];
    Real ucov[GR_DIM];
    Real bcon[GR_DIM];
    Real bcov[GR_DIM];
} FourVectors;

typedef struct {
    IndexRange ib;
    IndexRange jb;
    IndexRange kb;
} IndexRange3;

/**
 * Map of the locations of particular variables in a VariablePack
 * Used for operations conducted over all vars which must still
 * distinguish between them, e.g. flux.hpp
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
        // Use int8. 127 values ought to be enough for anybody, right?
        // Basic primitive variables
        int8_t RHO, UU, U1, U2, U3, B1, B2, B3;
        // Tracker variables
        int8_t RHO_ADDED, UU_ADDED, PASSIVE;
        // Electron entropy/energy tracking
        int8_t KTOT, K_CONSTANT, K_HOWES, K_KAWAZURA, K_WERNER, K_ROWAN, K_SHARMA;
        // Implicit-solver variables: constraint damping, EGRMHD
        int8_t PSI, Q, DP;
        // Total struct size ~20 bytes, < 1 vector of 4 doubles

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
                // Extended MHD
                Q = name_map["cons.q"].first;
                DP = name_map["cons.dP"].first;
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
                // Extended MHD
                Q = name_map["prims.q"].first;
                DP = name_map["prims.dP"].first;
            }
            U2 = U1 + 1;
            U3 = U1 + 2;
            B2 = B1 + 1;
            B3 = B1 + 2;
        }
};

/**
 * Functions for checking boundaries in 3D.
 * Uses IndexRange objects, or this would be in kharma_utils.hpp
 */
KOKKOS_INLINE_FUNCTION bool outside(const int& k, const int& j, const int& i,
                                    const IndexRange& kb, const IndexRange& jb, const IndexRange& ib)
{
    return (i < ib.s) || (i > ib.e) || (j < jb.s) || (j > jb.e) || (k < kb.s) || (k > kb.e);
}
KOKKOS_INLINE_FUNCTION bool inside(const int& k, const int& j, const int& i,
                                   const IndexRange& kb, const IndexRange& jb, const IndexRange& ib)
{
    // This is faster in the case that the point is outside
    return !outside(k, j, i, kb, jb, ib);
}

/**
 * Function for checking boundary flags: is this a domain or internal bound?
 */
inline bool IsDomainBound(std::shared_ptr<MeshBlock> pmb, BoundaryFace face)
{
    return !(pmb->boundary_flag[face] == BoundaryFlag::block ||
             pmb->boundary_flag[face] == BoundaryFlag::periodic);
}

inline bool BoundaryIsInner(IndexDomain domain)
{
    return domain == IndexDomain::inner_x1 ||
           domain == IndexDomain::inner_x2 ||
           domain == IndexDomain::inner_x3;
}

inline int BoundarySide(IndexDomain domain)
{
    switch (domain) {
        case IndexDomain::inner_x1:
        case IndexDomain::outer_x1:
            return 1;
        case IndexDomain::inner_x2:
        case IndexDomain::outer_x2:
            return 2;
        case IndexDomain::inner_x3:
        case IndexDomain::outer_x3:
            return 3;
        default:
            return 0;
    }
}

inline std::string BoundaryName(IndexDomain domain)
{
    switch (domain) {
        case IndexDomain::inner_x1:
            return "inner_x1";
        case IndexDomain::outer_x1:
            return "outer_x1";
        case IndexDomain::inner_x2:
            return "inner_x2";
        case IndexDomain::outer_x2:
            return "outer_x2";
        case IndexDomain::inner_x3:
            return "inner_x3";
        case IndexDomain::outer_x3:
            return "outer_x3";
        case IndexDomain::interior:
            return "interior";
        case IndexDomain::entire:
            return "entire";
        default:
            return "unknown";
    }
}

/**
 * Get zones in the domain interior
 */

/**
 * Get the 
 */
inline IndexRange3 GetPhysicalZones(std::shared_ptr<MeshBlock> pmb, IndexShape& bounds)
{
    return IndexRange3{IndexRange{IsDomainBound(pmb, BoundaryFace::inner_x1)
                                    ? bounds.is(IndexDomain::interior)
                                    : bounds.is(IndexDomain::entire),
                                  IsDomainBound(pmb, BoundaryFace::outer_x1)
                                    ? bounds.ie(IndexDomain::interior)
                                    : bounds.ie(IndexDomain::entire)},
                       IndexRange{IsDomainBound(pmb, BoundaryFace::inner_x2)
                                    ? bounds.js(IndexDomain::interior)
                                    : bounds.js(IndexDomain::entire),
                                  IsDomainBound(pmb, BoundaryFace::outer_x2)
                                    ? bounds.je(IndexDomain::interior)
                                    : bounds.je(IndexDomain::entire)},
                       IndexRange{IsDomainBound(pmb, BoundaryFace::inner_x3)
                                    ? bounds.ks(IndexDomain::interior)
                                    : bounds.ks(IndexDomain::entire),
                                  IsDomainBound(pmb, BoundaryFace::outer_x3)
                                    ? bounds.ke(IndexDomain::interior)
                                    : bounds.ke(IndexDomain::entire)}};
}

/**
 * Functions for "tracing" execution by printing strings (and optionally state of zones)
 * at each important function entry/exit
 */
#if TRACE
#define PRINTCORNERS 0
#define PRINTZONE 0
inline void PrintCorner(MeshBlockData<Real> *rc)
{
    auto rhop = rc->Get("prims.rho").data.GetHostMirrorAndCopy();
    auto up = rc->Get("prims.u").data.GetHostMirrorAndCopy();
    auto uvecp = rc->Get("prims.uvec").data.GetHostMirrorAndCopy();
    auto Bp = rc->Get("prims.B").data.GetHostMirrorAndCopy();
    auto rhoc = rc->Get("cons.rho").data.GetHostMirrorAndCopy();
    auto uc = rc->Get("cons.u").data.GetHostMirrorAndCopy();
    auto uvecc = rc->Get("cons.uvec").data.GetHostMirrorAndCopy();
    auto Bu = rc->Get("cons.B").data.GetHostMirrorAndCopy();
    //auto p = rc->Get("p").data.GetHostMirrorAndCopy();
    auto pflag = rc->Get("pflag").data.GetHostMirrorAndCopy();
    //auto q = rc->Get("prims.q").data.GetHostMirrorAndCopy();
    //auto dP = rc->Get("prims.dP").data.GetHostMirrorAndCopy();
    const IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
    std::cerr << "p:";
    for (int j=0; j<8; j++) {
        std::cerr << std::endl;
        for (int i=0; i<8; i++) {
            fprintf(stderr, "%.5g\t", pflag(kb.s, j, i));
        }
    }
    // std::cerr << std::endl << "B1:";
    // for (int j=0; j<8; j++) {
    //     std::cerr << std::endl;
    //     for (int i=0; i<8; i++) {
    //         fprintf(stderr, "%.5g\t", Bu(V1, kb.s, j, i));
    //     }
    // }
    std::cerr << std::endl << std::endl;
}

inline void PrintZone(MeshBlockData<Real> *rc)
{
    auto rhop = rc->Get("prims.rho").data.GetHostMirrorAndCopy();
    auto up = rc->Get("prims.u").data.GetHostMirrorAndCopy();
    auto uvecp = rc->Get("prims.uvec").data.GetHostMirrorAndCopy();
    auto Bp = rc->Get("prims.B").data.GetHostMirrorAndCopy();
    auto q = rc->Get("prims.q").data.GetHostMirrorAndCopy();
    auto dP = rc->Get("prims.dP").data.GetHostMirrorAndCopy();

    auto rhoU = rc->Get("cons.rho").data.GetHostMirrorAndCopy();
    auto uU = rc->Get("cons.u").data.GetHostMirrorAndCopy();
    auto uvecU = rc->Get("cons.uvec").data.GetHostMirrorAndCopy();
    auto BU = rc->Get("cons.B").data.GetHostMirrorAndCopy();
    auto qU = rc->Get("cons.q").data.GetHostMirrorAndCopy();
    auto dPU = rc->Get("cons.dP").data.GetHostMirrorAndCopy();

    std::cerr << "RHO: " << rhop(0,108,63)
         << " UU: "  << up(0,108,63)
         << " U: "   << uvecp(0,0,108,63) << " " << uvecp(1,0,108,63)<< " " << uvecp(2,0,108,63)
         << " B: "   << Bp(0,0,108,63) << " " << Bp(1,0,108,63) << " " << Bp(2,0,108,63)
         << " q: "   << q(0,108,63) 
         << " dP: "  << dP(0,108,63) << std::endl;
    std::cerr << "RHO: " << rhoU(0,108,63)
         << " UU: "  << uU(0,108,63)
         << " U: "   << uvecU(0,0,108,63) << " " << uvecU(1,0,108,63)<< " " << uvecU(2,0,108,63)
         << " B: "   << BU(0,0,108,63) << " " << BU(1,0,108,63) << " " << BU(2,0,108,63)
         << " q: "   << qU(0,108,63) 
         << " dP: "  << dPU(0,108,63) << std::endl;
}

inline void Flag(std::string label)
{
    if(MPIRank0()) std::cerr << label << std::endl;
}

inline void Flag(MeshBlockData<Real> *rc, std::string label)
{
    if(MPIRank0()) {
        std::cerr << label << std::endl;
        if(PRINTCORNERS) PrintCorner(rc);
        if(PRINTZONE) PrintZone(rc);
    }
}

inline void Flag(MeshData<Real> *md, std::string label)
{
    if(MPIRank0()) {
        std::cerr << label << std::endl;
        if(PRINTCORNERS || PRINTZONE) {
            auto rc = md->GetBlockData(0).get();
            if(PRINTCORNERS) PrintCorner(rc);
            if(PRINTZONE) PrintZone(rc);
        }
    }
}

#else
inline void Flag(std::string label) {}
inline void Flag(MeshBlockData<Real> *rc, std::string label) {}
inline void Flag(MeshData<Real> *md, std::string label) {}
#endif
/**
 * Versions of Flag() that take shared_ptr objects and call through with get()
 * Avoids having to pay attention to shared_ptr vs * pointers in adding Flag() calls
 * when diagnosing a problem.
 */
inline void Flag(std::shared_ptr<MeshBlockData<Real>>& rc, std::string label) { Flag(rc.get(), label); }
inline void Flag(std::shared_ptr<MeshData<Real>>& md, std::string label) { Flag(md.get(), label); }
