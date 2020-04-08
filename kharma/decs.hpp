
#pragma once
#include <mpark/variant.hpp>

#include <stdexcept>
#include <map>

// Classic Macros^(TM)
#define VERSION "kharm-alpha-0.1"
#define NDIM 4
#define DLOOP1 for(int mu = 0; mu < NDIM; ++mu)
#define DLOOP2 DLOOP1 for(int nu = 0; nu < NDIM; ++nu)
#define DLOOP3 DLOOP2 for(int lam = 0; lam < NDIM; ++lam)
#define DLOOP4 DLOOP3 for(int kap = 0; kap < NDIM; ++kap)

// The standard HARMDriver object will evolve the 8 primitives/conserved for GRMHD.
// Anything extra should be handled by new physics packages
#define NPRIM 8
#define PLOOP for(int mu = 0; mu < NPRIM; ++mu)

// Accuracy for numerical derivatives
#define DELTA 1.e-5
// Arbitrary small number >0
#define SMALL 1.e-40

// Useful Enums to avoid lots of #defines. TODO move to suitable headers
#define NLOC 5
enum Loci{face1=0, face2, face3, center, corner};
// TODO explain these
enum InversionStatus{neg_input=0, max_iter, bad_ut, bad_gamma, neg_rho, neg_u, neg_rhou};


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
