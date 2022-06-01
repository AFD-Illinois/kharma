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

// Bare Parthenon defs
// Anything more leads to circular deps from gr_coordinates.hpp
// TODO update, this was from very early Parthenon
#include "parthenon_arrays.hpp"
#include "parthenon_mpi.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "mesh/domain.hpp"

// My set of MPI wrappers, stubbed out when MPI is not present
#include "mpi.hpp"

// KHARMA DEFINITIONS

// Parthenon stole our type names
// Lots of work will need to be done for Real != double
using Real = parthenon::Real;
using GReal = double;

// A small number, compared to the grid or problem scale
#define SMALL 1e-20

// GEOMETRY
#define GR_DIM 4
#define DLOOP1 for(int mu = 0; mu < GR_DIM; ++mu)
#define DLOOP2 DLOOP1 for(int nu = 0; nu < GR_DIM; ++nu)
#define DLOOP3 DLOOP2 for(int lam = 0; lam < GR_DIM; ++lam)
#define DLOOP4 DLOOP3 for(int kap = 0; kap < GR_DIM; ++kap)

#define NVEC 3
#define VLOOP for(int v = 0; v < NVEC; ++v)
#define VLOOP2 VLOOP for(int w = 0; w < NVEC; ++w)
// This provides a way of addressing vectors that matches
// directions, to make derivatives etc more readable
#define V1 0
#define V2 1
#define V3 2

// And an odd but useful loop for ex-iharm3d code
// This requires nvar to be defined in caller!
// It is not a const/global anymore.  So, use this loop carefully
#define PLOOP for(int ip=0; ip < nvar; ++ip)

// Useful Enums to avoid lots of #defines
// See following functions and coord() in gr_coordinates.hpp to
// get an idea of these locations.  All faces/corner are *left* of center
#define NLOC 5
enum Loci{face1=0, face2, face3, center, corner};

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

// Emulate old names for possible stronger typing later,
// and for readability
// TODO specify ParArrayXD instead of generic?
using GridScalar = parthenon::ParArrayND<Real>;
using GridVector = parthenon::ParArrayND<Real>;
using GridVars = parthenon::ParArrayND<Real>;  // TODO ELIM
using GridInt = parthenon::ParArrayND<int>;

using GeomScalar = parthenon::ParArrayND<Real>;
using GeomVector = parthenon::ParArrayND<Real>;
using GeomTensor2 = parthenon::ParArrayND<Real>;
using GeomTensor3 = parthenon::ParArrayND<Real>;

// Specific lambdas for our array shapes
#define KOKKOS_LAMBDA_1D KOKKOS_LAMBDA (const int& i)
#define KOKKOS_LAMBDA_2D KOKKOS_LAMBDA (const int& j, const int& i)
#define KOKKOS_LAMBDA_3D KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_4D KOKKOS_LAMBDA (const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_5D KOKKOS_LAMBDA (const int& m, const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_VARS KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_VEC KOKKOS_LAMBDA (const int &mu, const int &k, const int &j, const int &i)
// Same things for mesh-wide ops
#define KOKKOS_LAMBDA_MESH_1D KOKKOS_LAMBDA (const int& b, const int& i)
#define KOKKOS_LAMBDA_MESH_2D KOKKOS_LAMBDA (const int& b, const int& j, const int& i)
#define KOKKOS_LAMBDA_MESH_3D KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_MESH_4D KOKKOS_LAMBDA (const int& b, const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_MESH_5D KOKKOS_LAMBDA (const int& b, const int& m, const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_MESH_VARS KOKKOS_LAMBDA (const int& b, const int &p, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_MESH_VEC KOKKOS_LAMBDA (const int& b, const int &mu, const int &k, const int &j, const int &i)

// TODO separate macros for return type if this becomes a thing?  Or don't macro at all
#define KOKKOS_LAMBDA_1D_REDUCE KOKKOS_LAMBDA (const int &i, Real &local_result)
// This is used for timestep and divB, which are explicitly double
#define KOKKOS_LAMBDA_2D_REDUCE KOKKOS_LAMBDA (const int &j, const int &i, double &local_result)
#define KOKKOS_LAMBDA_3D_REDUCE KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result)
#define KOKKOS_LAMBDA_3D_REDUCE_INT KOKKOS_LAMBDA (const int &k, const int &j, const int &i, int &local_result)
// Versions for full mesh
#define KOKKOS_LAMBDA_MESH_3D_REDUCE KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, double &local_result)
#define KOKKOS_LAMBDA_MESH_3D_REDUCE_INT KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result)
#define KOKKOS_LAMBDA_MESH_4D_REDUCE KOKKOS_LAMBDA (const int &b, const int &v, const int &k, const int &j, const int &i, double &local_result)
