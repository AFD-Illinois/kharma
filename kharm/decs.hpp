/*
 * decs.h
 *
 *  Created on: Mar 11, 2018
 *      Author: bprather
 */
#pragma once

#include <Kokkos_Core.hpp>
#include <stdexcept>

// Classic Macros^(TM)
#define NDIM 4
#define DLOOP for(index_t dim = 0; dim < NDIM; ++dim)

// Precision flexibility:
// Real is used for arrays & temps of physical variables & metric values
// GReal is used for arrays & temps of grid locations
typedef double Real;
//typedef float Real;
typedef double GReal;
//typedef fload GReal;

#if USE_MPI
#else
static const auto global_start = {0.0, 0.0, 0.0};
#endif

// Useful Enums to avoid lots of #defines
enum prims{rho, u, u1, u2, u3, B1, B2, B3};
enum Loci{face1, face2, face3, center, corner};

// Data structures common to all k-harm
#if defined( Kokkos_ENABLE_CUDA )
// TODO MemSpace, HostSpace
typedef Kokkos::View<Real***> GridScalar;
typedef Kokkos::View<Real***[NDIM]> GridVector;
typedef Kokkos::View<Real****> GridVars;
typedef Kokkos::View<Real****, Kokkos::HostSpace> GridVarsHost;
#warning "Compiling with CUDA"
#else
typedef Kokkos::View<Real***> GridScalar;
typedef Kokkos::View<Real***[NDIM]> GridVector;
typedef Kokkos::View<Real****> GridVars;
typedef GridVars GridVarsHost;
#warning "Compiling with OpenMP Only"
#endif
