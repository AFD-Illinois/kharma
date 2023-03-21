/* 
 *  File: decs.hpp
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

/**
 * General definitions and imports we'll need in all parts of KHARMA.
 * 
 * Note that this file *cannot* import all of Parthenon: it is itself
 * imported (indirectly) in several Parthenon headers, through
 * gr_coordinates.hpp, which provides Parthenon's coordinates object
 * 
 * Thus it is mostly geometry-related and Kokkos-related definitions.
 * Convenience functions and most KHARMA-specific datatypes are in types.h
 */

// KHARMA INCLUDES
// Standard libs we absolutely need everywhere
#include <map>
#include <memory>
#include <stdexcept>

// Libraries I need directly
#include "Kokkos_Core.hpp"

#if 0
// Resolve math functions to new Kokkos versions. Faster, maybe
namespace m = Kokkos::Experimental;
#else
// Resolve to standard library
namespace m = std;
#endif
// TODO CUDA library explicitly?

// Bare Parthenon defs
// Anything more leads to circular deps from gr_coordinates.hpp
// TODO update (carefully), this was from very early Parthenon
#include "parthenon_arrays.hpp"
#include "parthenon_mpi.hpp"
#include "globals.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "mesh/domain.hpp"

// KHARMA DEFINITIONS

// Parthenon stole our type names
// Lots of work will need to be done for Real != double
using parthenon::Real;
using GReal = double;

// A small number, compared to the grid or problem scale
#define SMALL 1e-20

// GEOMETRY
// This stuff needs to be in decs.h as it's used by functions in coordinates/,
// which must be imported *inside Parthenon* in order to use GRCoodrdinates
// in there
// TODO version DLOOP(mu,nu)?
#define GR_DIM 4
#define DLOOP1 for(int mu = 0; mu < GR_DIM; ++mu)
#define DLOOP2 DLOOP1 for(int nu = 0; nu < GR_DIM; ++nu)
#define DLOOP3 DLOOP2 for(int lam = 0; lam < GR_DIM; ++lam)
#define DLOOP4 DLOOP3 for(int kap = 0; kap < GR_DIM; ++kap)

#define NVEC 3
#define VLOOP for(int v = 0; v < NVEC; ++v)

// Useful Enums to avoid lots of #defines
// See following functions and coord() in gr_coordinates.hpp to
// get an idea of these locations.  All faces/corner are *left* of center
#define NLOC 5
enum class Loci{face1=0, face2, face3, center, corner};

// Return the face location corresponding to the direction 'dir'
KOKKOS_INLINE_FUNCTION Loci loc_of(const int& dir)
{
    switch (dir) {
    case 0:
        return Loci::center;
    case parthenon::X1DIR:
        return Loci::face1;
    case parthenon::X2DIR:
        return Loci::face2;
    case parthenon::X3DIR:
        return Loci::face3;
    default:
        return Loci::corner;
    }
}
KOKKOS_INLINE_FUNCTION int dir_of(const Loci loc)
{
    switch (loc) {
    case Loci::center:
        return 0;
    case Loci::face1:
        return parthenon::X1DIR;
    case Loci::face2:
        return parthenon::X2DIR;
    case Loci::face3:
        return parthenon::X3DIR;
    default:
        return -1;
    }
}

#ifdef MPI_PARALLEL
/**
 * Am I rank 0?  Saves typing vs comparing the global every time
 */
inline bool MPIRank0()
{
    return (parthenon::Globals::my_rank == 0 ? true : false);
}
#else
/**
 * Am I rank 0?  Saves typing vs comparing the global every time.
 * DUMMY function for no-MPI case: constexpr return for slight optimizations.
 */
inline bool MPIRank0() { return true; }
#endif // MPI_PARALLEL

// A few generic "NDArray" overloads for readability.
// TODO torn on futures of these, as they're used inconsistently
// Shape+3D ("Grid") arrays
using GridScalar = parthenon::ParArrayND<parthenon::Real>;
using GridVector = parthenon::ParArrayND<parthenon::Real>;
// Shape+2D ("Geom") versions for symmetric geometry
using GeomScalar = parthenon::ParArrayND<parthenon::Real>;
using GeomTensor2 = parthenon::ParArrayND<parthenon::Real>;
using GeomTensor3 = parthenon::ParArrayND<parthenon::Real>;
