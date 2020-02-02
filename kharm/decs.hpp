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
// There may be more than 8 vars!!!!!
// But only 8 are primitives. Sometimes we just need those together
#define NPRIM 8
#define PLOOP for(int mu = 0; mu < NPRIM; ++mu)

// Accuracy for numerical derivatives
#define DELTA 1.e-5
// Arbitrary small number >0
#define SMALL 1.e-40

// Precision flexibility:
// Real is used for arrays & temps of physical variables & metric values
// GReal is used for arrays & temps of grid locations
typedef double Real;
//typedef float Real;
typedef double GReal;
//typedef float GReal;
// TODO float Reals crash

typedef std::map<std::string, double> Parameters;

// Useful Enums to avoid lots of #defines
enum prims{rho, u, u1, u2, u3, B1, B2, B3};
#define NLOC 5
enum Loci{face1, face2, face3, center, corner};

// Data structures common to all k-harm
// TODO something cute with the type checker to distinguish prims from cons?  Names seem fine.
typedef Kokkos::View<Real***, Kokkos::LayoutLeft> GridScalar;
typedef Kokkos::View<int***> GridInt;
typedef Kokkos::View<Real***[NDIM]> GridVector;
typedef Kokkos::View<Real****> GridVars;

// TODO these all start with NLOC but C++ is mean
typedef Kokkos::View<Real***> GeomScalar;
typedef Kokkos::View<Real***[NDIM][NDIM]> GeomTensor;
// Connection coeffs are only recorded at zone center
typedef Kokkos::View<Real**[NDIM][NDIM][NDIM]> GeomConn;

#define KOKKOS_LAMBDA_3D KOKKOS_LAMBDA (const int &i, const int &j, const int &k)
#define KOKKOS_LAMBDA_VARS KOKKOS_LAMBDA (const int &i, const int &j, const int &k, const int &p)

#if defined( Kokkos_ENABLE_CUDA )
typedef Kokkos::View<Real****, Kokkos::HostSpace> GridVarsHost;
#warning "Compiling with CUDA"
#else
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

// TODO fix debug flag on CMake Debug target
#if DEBUG
#warning "Compiling with debug"
#endif

#if DEBUG
#define FLAG(x) cout << x << endl;
#else
#define FLAG(x)
#endif