/*
 * decs.h
 *
 *  Created on: Mar 11, 2018
 *      Author: bprather
 */

#pragma once

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>

// One step at a time
#if defined( KOKKOS_ENABLE_MPI )
#error "MPI not supported yet"
//#include <mpi.h>
#endif

// Check for CUDA and lambda dispatch, which I assume
#if defined( KOKKOS_ENABLE_CUDA )
// Declare our spaces rather than using defaults
typedef Kokkos::Cuda ExecSpace;
typedef Kokkos::CudaSpace MemSpace;
typedef Kokkos::LayoutRight Layout;
#else
// Declare our spaces rather than using defaults
typedef Kokkos::OpenMP ExecSpace;
typedef Kokkos::HostSpace MemSpace;
typedef Kokkos::LayoutRight Layout;
#endif

#if !defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA )
#error "C++11 lambda must be supported by NVCC (CUDA > 8.0)"
#endif

#define NDIM 3

// Macros
typedef unsigned int index_t;
#define FOR_NDIM for(index_t dim = 0; dim < NDIM; ++dim)

// Struct for reducing NDIM vectors
typedef struct VecReduce {
  double vector[NDIM];
  void operator+=(VecReduce const& other) {
    FOR_NDIM vector[dim] += other.vector[dim];
  }
  void operator+=(VecReduce const volatile& other) volatile {
    FOR_NDIM vector[dim] += other.vector[dim];
  }
} VecReduce;

typedef Kokkos::View<double*, Layout, MemSpace> KScalar;
typedef Kokkos::View<double*[NDIM], Layout, MemSpace> KVector;

typedef Kokkos::TeamPolicy<>               team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;
