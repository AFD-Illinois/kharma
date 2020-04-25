// General definitions for KHARMA the code

#pragma once

#include <stdexcept>
#include <map>

// Libraries I need directly
#include "Kokkos_Core.hpp"

// Parthenon defs
#include "athena.hpp"

// Stuff that's useful across the whole code
#define VERSION "kharma-alpha-0.1"

// Parthenon stole our type names
using Real = parthenon::Real;
using GReal = double;

// TODO make this MPI-aware
// TODO add make.sh/CMake option for tracing directly
#if DEBUG
#define FLAG(x) std::cout << x << std::endl;
#else
#define FLAG(x)
#endif

// TODO split out GR header with these?
#define NDIM 4
#define DLOOP1 for(int mu = 0; mu < NDIM; ++mu)
#define DLOOP2 DLOOP1 for(int nu = 0; nu < NDIM; ++nu)
#define DLOOP3 DLOOP2 for(int lam = 0; lam < NDIM; ++lam)
#define DLOOP4 DLOOP3 for(int kap = 0; kap < NDIM; ++kap)

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

// Emulate old names, for 2 reasons:
// 1. Compat with files from K/HARM
// 2. May be possible to make these more strongly typed in future
using GridScalar = parthenon::ParArrayND<Real>;
using GridVector = parthenon::ParArrayND<Real>;
using GridVars = parthenon::ParArrayND<Real>;
using GridInt = parthenon::ParArrayND<int>;

using GeomScalar = parthenon::ParArrayND<Real>;
using GeomVector = parthenon::ParArrayND<Real>;
using GeomTensor2 = parthenon::ParArrayND<Real>;
using GeomTensor3 = parthenon::ParArrayND<Real>;

// Specific lambdas for our array shapes
#define KOKKOS_LAMBDA_3D KOKKOS_LAMBDA (const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_VARS KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i)

// Struct for derived 4-vectors at a point, usually calculated and needed together
typedef struct {
    parthenon::Real ucon[NDIM];
    parthenon::Real ucov[NDIM];
    parthenon::Real bcon[NDIM];
    parthenon::Real bcov[NDIM];
} FourVectors;

// Denote inversion failures (pflags). See U_to_P for status explanations
enum InversionStatus{success=0, neg_input, max_iter, bad_ut, bad_gamma, neg_rho, neg_u, neg_rhou};

// Floor codes are bit-masks, so it makes a lot less sense to use an enum
// See ApplyFloors for code explanations
#define HIT_FLOOR_GEOM_RHO 1
#define HIT_FLOOR_GEOM_U 2
#define HIT_FLOOR_B_RHO 4
#define HIT_FLOOR_B_U 8
#define HIT_FLOOR_TEMP 16
#define HIT_FLOOR_GAMMA 32
#define HIT_FLOOR_KTOT 64

// Most floors are in fixup.hpp only, but this one is needed by U_to_P
// TODO of course they should be runtime prefs
#define GAMMAMAX 200.