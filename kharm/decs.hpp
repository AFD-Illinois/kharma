/*
 * decs.h
 *
 *  Created on: Mar 11, 2018
 *      Author: bprather
 */
#pragma once

#include <Kokkos_Core.hpp>
#include <mpark/variant.hpp>

#include <stdexcept>
#include <map>

// Classic Macros^(TM)
#define VERSION "kharm-alpha-0.1"
#define NDIM 4
#define DLOOP1 for(int mu = 0; mu < NDIM; ++mu)
#define DLOOP2 for(int mu = 0; mu < NDIM; ++mu) for(int nu = 0; nu < NDIM; ++nu)
// There may be more than 8 variables, so don't use these
// unless you *only* want the primitives
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
// TODO float Reals crash. Hybridize w/double B?

// TODO take an arbitrary list + command line map and shove it all in here
typedef struct std::map<std::string, mpark::variant<int, double, std::string>> Parameters;

// Useful Enums to avoid lots of #defines
enum prims{rho=0, u, u1, u2, u3, B1, B2, B3};
#define NLOC 5
enum Loci{face1=0, face2, face3, center, corner};

// Data structures common to all k-harm
// TODO something cute with the type checker to distinguish prims from cons?  Names seem fine.
typedef Kokkos::View<Real***> GridScalar;
typedef Kokkos::View<int***> GridInt;
typedef Kokkos::View<Real***[NDIM]> GridVector;
typedef Kokkos::View<Real****> GridVars;

// TODO these start with NLOC but C++ is mean
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

#if DEBUG
#warning "Compiling with debug"
#define FLAG(x) cout << x << endl;
#else
#define FLAG(x)
#endif

// pflag codes, for indicating causes of inversion failures
// TODO enum?
#define ERR_NEG_INPUT -100
#define ERR_MAX_ITER 1
#define ERR_UTSQ 2
#define ERR_GAMMA 3
#define ERR_RHO_NEGATIVE 6
#define ERR_U_NEGATIVE 7
#define ERR_BOTH_NEGATIVE 8