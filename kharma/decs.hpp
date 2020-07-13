// General definitions for KHARMA the code

#pragma once

// Standard libs we absolutely need everywhere
#include <map>
#include <memory>
#include <stdexcept>

// Libraries I need directly
#include "Kokkos_Core.hpp"

// Bare Parthenon defs
// Anything more leads to circular deps from gr_coordinates.hpp
#include "parthenon_arrays.hpp"
#include "parthenon_mpi.hpp"

// My set of MPI wrappers, stubbed out when MPI not present
#include "mpi.hpp"

// Parthenon stole our type names
using Real = parthenon::Real;
using GReal = double;

// TODO add make.sh/CMake option for tracing vs just debug
#if DEBUG_TRACE
#define FLAG(x) if(MPIRank0()) std::cout << x << std::endl;
#else
#define FLAG(x)
#endif

// TODO split out GR header with these?
#define GR_DIM 4
#define DLOOP1 for(int mu = 0; mu < GR_DIM; ++mu)
#define DLOOP2 DLOOP1 for(int nu = 0; nu < GR_DIM; ++nu)
#define DLOOP3 DLOOP2 for(int lam = 0; lam < GR_DIM; ++lam)
#define DLOOP4 DLOOP3 for(int kap = 0; kap < GR_DIM; ++kap)

// Useful Enums to avoid lots of #defines. TODO move to suitable headers
#define NLOC 5
enum Loci{face1=0, face2, face3, center, corner};

// Accuracy for numerical derivatives of the metric
#define DELTA 1.e-5

// TODO move to grmhd.hpp or similar?
// The standard HARMDriver object will evolve the 8 primitives/conserved for GRMHD.
// Anything extra should be handled by new physics packages
#define NPRIM 8
#define PLOOP for(int p = 0; p < NPRIM; ++p)
enum prims{rho=0, u, u1, u2, u3, B1, B2, B3};
// Just the fluid variables, among the primitives
#define NFLUID 5
#define FLOOP for(int p=0; p < NFLUID; p++)

// Emulate old names for possible stronger typing...
using GridScalar = parthenon::ParArrayND<Real>;
using GridVector = parthenon::ParArrayND<Real>;
using GridVars = parthenon::ParArrayND<Real>;
using GridInt = parthenon::ParArrayND<int>;

using GeomScalar = parthenon::ParArrayND<Real>;
using GeomVector = parthenon::ParArrayND<Real>;
using GeomTensor2 = parthenon::ParArrayND<Real>;
using GeomTensor3 = parthenon::ParArrayND<Real>;

// Specific lambdas for our array shapes
#define KOKKOS_LAMBDA_1D KOKKOS_LAMBDA (const int& i)
#define KOKKOS_LAMBDA_2D KOKKOS_LAMBDA (const int& j, const int& i)
#define KOKKOS_LAMBDA_3D KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_VARS KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i)
// TODO separate macros for return type if this becomes a thing?  Or don't macro at all
#define KOKKOS_LAMBDA_1D_REDUCE KOKKOS_LAMBDA (const int &i, Real &local_result)
// This is used for timestep and divB, which are explicitly double.  Lots of work would need to be done to Parthenon if Real != double though
#define KOKKOS_LAMBDA_3D_REDUCE KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result)
#define KOKKOS_LAMBDA_3D_REDUCE_INT KOKKOS_LAMBDA (const int &k, const int &j, const int &i, int &local_result)

// KHARMA TYPES

// Struct for derived 4-vectors at a point, usually calculated and needed together
typedef struct {
    parthenon::Real ucon[GR_DIM];
    parthenon::Real ucov[GR_DIM];
    parthenon::Real bcon[GR_DIM];
    parthenon::Real bcov[GR_DIM];
} FourVectors;

// Denote inversion failures (pflags). See U_to_P for status explanations
enum InversionStatus{success=0, neg_input, max_iter, bad_ut, bad_gamma, neg_rho, neg_u, neg_rhou};

// Denote reconstruction algorithms
enum ReconstructionType{linear_mc=0, ppm, weno5, mp5};

// Floor codes are non-exclusive, so it makes a lot less sense to use an enum
// Instead, we start them high enough that we can stick the enum in the bottom 5 bits
// See ApplyFloors for code explanations
#define HIT_FLOOR_GEOM_RHO 32
#define HIT_FLOOR_GEOM_U 64
#define HIT_FLOOR_B_RHO 128
#define HIT_FLOOR_B_U 256
#define HIT_FLOOR_TEMP 512
#define HIT_FLOOR_GAMMA 1024
#define HIT_FLOOR_KTOT 2048

// KHARMA OPTIONS

// GAMMA FLOOR
// Maximum gamma factor allowed for fluid velocity
// Defined in decs.hpp since it's also needed by U_to_P
#define GAMMAMAX 200.

// GEOMETRY FLOORS
// Limiting values for density and internal energy
// These are scaled with radius for spherical sims,
// and multiplied by an additional 0.01 for cartesian sims
#define RHOMIN 1.e-6
#define UUMIN  1.e-8
// Radius in M, around which to steepen floor prescription from r^-2 to r^-3
#define FLOOR_R_CHAR 10.

// RATIO CEILINGS
// Maximum ratio of internal energy to density (i.e. Temperature)
#define UORHOMAX   100.
// Same for magnetic field (i.e. magnetization sigma)
#define BSQORHOMAX 500.
#define BSQOUMAX   (BSQORHOMAX * UORHOMAX)