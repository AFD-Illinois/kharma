// General definitions for KHARMA the code

#pragma once

#include <stdexcept>
#include <map>

// Libraries I need directly
#include <mpark/variant.hpp>
#include "Kokkos_Core.hpp"

// Parthenon defs: Real, etc.  (srsly, athena?)
#include "athena.hpp"

// Classic Macros^(TM)
#define VERSION "kharma-alpha-0.1"
#define NDIM 4
#define DLOOP1 for(int mu = 0; mu < NDIM; ++mu)
#define DLOOP2 DLOOP1 for(int nu = 0; nu < NDIM; ++nu)
#define DLOOP3 DLOOP2 for(int lam = 0; lam < NDIM; ++lam)
#define DLOOP4 DLOOP3 for(int kap = 0; kap < NDIM; ++kap)

// Parthenon stole our type names
using Real = parthenon::Real;
using GReal = double;

// TODO split out GRMHD-specific header

// Struct for derived 4-vectors at a point, for readability
typedef struct {
    parthenon::Real ucon[NDIM];
    parthenon::Real ucov[NDIM];
    parthenon::Real bcon[NDIM];
    parthenon::Real bcov[NDIM];
} FourVectors;

// The standard HARMDriver object will evolve the 8 primitives/conserved for GRMHD.
// Anything extra should be handled by new physics packages
#define NPRIM 8
#define PLOOP for(int mu = 0; mu < NPRIM; ++mu)
enum prims{rho=0, u, u1, u2, u3, B1, B2, B3};

// Accuracy for numerical derivatives of the metric
#define DELTA 1.e-5

// Useful Enums to avoid lots of #defines. TODO move to suitable headers
#define NLOC 5
enum Loci{face1=0, face2, face3, center, corner};
// TODO explain these
enum InversionStatus{success=0, neg_input, max_iter, bad_ut, bad_gamma, neg_rho, neg_u, neg_rhou};

// These are to declare lambdas optionally host or device-side. Always use these with Kokkos_range or par_for
#define KOKKOS_LAMBDA_3D KOKKOS_LAMBDA (const int &i, const int &j, const int &k)
#define KOKKOS_LAMBDA_VARS KOKKOS_LAMBDA (const int &i, const int &j, const int &k, const int &p)

#if defined( Kokkos_ENABLE_CUDA )
#warning "Compiling with CUDA"
#else
#warning "Compiling with OpenMP Only"
#endif

// TODO MPI for flags, 
#if DEBUG
#warning "Compiling with debug"
#define FLAG(x) std::cout << x << std::endl;
#else
#define FLAG(x)
#endif
