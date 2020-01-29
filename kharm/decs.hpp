/*
 * decs.h
 *
 *  Created on: Mar 11, 2018
 *      Author: bprather
 */
#pragma once

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <map>

// Classic Macros^(TM)
#define VERSION "kharm-alpha-0.1"
#define NDIM 4
#define DLOOP1 for(int mu = 0; mu < NDIM; ++mu)
#define DLOOP2 for(int mu = 0; mu < NDIM; ++mu) for(int nu = 0; nu < NDIM; ++nu)
// TODO PLOOP?  Rely on np being defined or get it from G?

// Precision flexibility:
// Real is used for arrays & temps of physical variables & metric values
// GReal is used for arrays & temps of grid locations
typedef double Real;
//typedef float Real;
typedef double GReal;
//typedef fload GReal;

typedef std::map<std::string, double> Parameters;

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

// Derived 4-vectors
typedef struct {
    Real ucon[NDIM];
    Real ucov[NDIM];
    Real bcon[NDIM];
    Real bcov[NDIM];
} Derived;
typedef struct {
    GridVector ucon;
    GridVector ucov;
    GridVector bcon;
    GridVector bcov;
} GridDerived;

#if DEBUG
#warning "Compiling with debug"
#endif