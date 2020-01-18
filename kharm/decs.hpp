/*
 * decs.h
 *
 *  Created on: Mar 11, 2018
 *      Author: bprather
 */

#pragma once

#include <Kokkos_Core.hpp>

// Classic Macros^(TM)
#define NDIM 4
#define DLOOP for(index_t dim = 0; dim < NDIM; ++dim)

// Data structures common to all k-harm
#if defined( Kokkos_ENABLE_CUDA )
// TODO MemSpace, HostSpace
typedef Kokkos::View<double***> GridScalar;
typedef Kokkos::View<double***[NDIM]> GridVector;
typedef Kokkos::View<double****> GridPrims;
typedef Kokkos::View<double****, Kokkos::HostSpace> GridPrimsHost;
#warning "Compiling with CUDA"
#else
typedef Kokkos::View<double***> GridScalar;
typedef Kokkos::View<double***[NDIM]> GridVector;
typedef Kokkos::View<double****> GridPrims;
typedef GridPrims GridPrimsHost;
#warning "Compiling with OpenMP Only"
#endif
