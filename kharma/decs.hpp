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

// General definitions for KHARMA the code, applying 
// Most compile-time options are in kharma/CMakeLists.txt
// Some can be set through make.sh, some require editing that file

#pragma once

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

// Accuracy for numerical derivatives of the metric
#define DELTA 1.e-8
// Accuracy required for U to P
#define UTOP_ERRTOL 1.e-8

// GEOMETRY
#define GR_DIM 4
#define DLOOP1 for(int mu = 0; mu < GR_DIM; ++mu)
#define DLOOP2 DLOOP1 for(int nu = 0; nu < GR_DIM; ++nu)
#define DLOOP3 DLOOP2 for(int lam = 0; lam < GR_DIM; ++lam)
#define DLOOP4 DLOOP3 for(int kap = 0; kap < GR_DIM; ++kap)

#define NVEC 3
#define VLOOP for(int v = 0; v < NVEC; ++v)
#define VLOOP2 VLOOP for(int w = 0; w < NVEC; ++w)

// Useful Enums to avoid lots of #defines
#define NLOC 5
enum Loci{face1=0, face2, face3, center, corner};

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
// Versions for full mesh (TODO use only these in KHARMA)
#define KOKKOS_LAMBDA_MESH_3D_REDUCE KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, double &local_result)
#define KOKKOS_LAMBDA_MESH_3D_REDUCE_INT KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, int &local_result)
// KHARMA FUNCTIONS

/**
 * Return whether a boundary is physical (i.e. border of the simulation) -- that is, not internal or periodic
 * Ironically, the zones in non-physical boundaries are "physical" i.e. bulk, non-ghost zones
 * 
 * Defined because UtoP needs to calculate primitives for real, domain zones -- that is, where this function returns false
 */
inline bool IsPhysicalBound(parthenon::BoundaryFlag bflag) {
    //if (bflag == parthenon::BoundaryFlag::undef) throw std::invalid_argument("Undefined boundary flag!");
    //return bflag != parthenon::BoundaryFlag::block && bflag != parthenon::BoundaryFlag::periodic;
    return false;
}

inline parthenon::IndexRange GetPhysicalZonesI(parthenon::BoundaryFlag boundary_flags[6], parthenon::IndexShape cellbounds)
{
    parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
    parthenon::IndexDomain entire = parthenon::IndexDomain::entire;
    int is = IsPhysicalBound(boundary_flags[parthenon::BoundaryFace::inner_x1]) ?
                cellbounds.is(interior) : cellbounds.is(entire);
    int ie = IsPhysicalBound(boundary_flags[parthenon::BoundaryFace::outer_x1]) ?
                cellbounds.ie(interior) : cellbounds.ie(entire);
    return parthenon::IndexRange{is, ie};
}
inline parthenon::IndexRange GetPhysicalZonesJ(parthenon::BoundaryFlag boundary_flags[6], parthenon::IndexShape cellbounds)
{
    parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
    parthenon::IndexDomain entire = parthenon::IndexDomain::entire;
    int js = IsPhysicalBound(boundary_flags[parthenon::BoundaryFace::inner_x2]) ?
                cellbounds.js(interior) : cellbounds.js(entire);
    int je = IsPhysicalBound(boundary_flags[parthenon::BoundaryFace::outer_x2]) ?
                cellbounds.je(interior) : cellbounds.je(entire);
    return parthenon::IndexRange{js, je};
}
inline parthenon::IndexRange GetPhysicalZonesK(parthenon::BoundaryFlag boundary_flags[6], parthenon::IndexShape cellbounds)
{
    parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
    parthenon::IndexDomain entire = parthenon::IndexDomain::entire;
    int ks = IsPhysicalBound(boundary_flags[parthenon::BoundaryFace::inner_x3]) ?
                cellbounds.ks(interior) : cellbounds.ks(entire);
    int ke = IsPhysicalBound(boundary_flags[parthenon::BoundaryFace::outer_x3]) ?
                cellbounds.ke(interior) : cellbounds.ke(entire);
    return parthenon::IndexRange{ks, ke};
}

// This is a macro and not a function for the sole reason that it still compiles if I forget the semicolon
#if TRACE
#define FLAG(x) if(MPIRank0()) std::cout << x << std::endl;
#else
#define FLAG(x)
#endif