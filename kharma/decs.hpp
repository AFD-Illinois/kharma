// General definitions for KHARMA the code

#pragma once

// KHARMA UNIVERSAL INCLUDES
// Standard libs we absolutely need everywhere
#include <map>
#include <memory>
#include <stdexcept>

// Libraries I need directly
#include "Kokkos_Core.hpp"

// Bare Parthenon defs
// Anything more leads to circular deps from gr_coordinates.hpp
// TODO update, this was from very early Parthenon
#include "parthenon_arrays.hpp"
#include "parthenon_mpi.hpp"
#include "bvals/bvals_interfaces.hpp"

// My set of MPI wrappers, stubbed out when MPI is not present
#include "mpi.hpp"

// KHARMA OPTIONS

// Parthenon stole our type names
// Lots of work will need to be done for Real != double
using Real = parthenon::Real;
using GReal = double;

// TODO make these CMAKE flags
#define TRACE 0
// Bundle the usual four flux calculation kernels (floors,R,L,apply)
// into one. Worth experimenting
#define FUSE_FLUX_KERNELS 1
// Bundle the three emf direction kernels into one
// Only affects much in *very* strong scaling regime
#define FUSE_EMF_KERNELS 0
// Bundle the two floor kernels into one
#define FUSE_FLOOR_KERNELS 1

// KHARMA DEFINITIONS

// TODO split out a GR header with these?
#define GR_DIM 4
#define DLOOP1 for(int mu = 0; mu < GR_DIM; ++mu)
#define DLOOP2 DLOOP1 for(int nu = 0; nu < GR_DIM; ++nu)
#define DLOOP3 DLOOP2 for(int lam = 0; lam < GR_DIM; ++lam)
#define DLOOP4 DLOOP3 for(int kap = 0; kap < GR_DIM; ++kap)

// Useful Enums to avoid lots of #defines
#define NLOC 5
enum Loci{face1=0, face2, face3, center, corner};

// Accuracy for numerical derivatives of the metric
#define DELTA 1.e-8
// Accuracy required for U to P
#define UTOP_ERRTOL 1.e-8

// The 5 fluid primitives for HD/GRHD
#define NPRIM 5
#define PLOOP for(int p = 0; p < NPRIM; ++p)
enum prims{rho=0, u, u1, u2, u3};

// B fields etc etc are split into packages
#define NVEC 3
#define VLOOP for(int v = 0; v < NVEC; ++v)

// Map of the locations of particular variables in a VariablePack
// Used for operations conducted over all vars which must still
// distinguish them, e.g. fluxes.hpp
struct varmap {
    int p, u; // GRMHD variables
    int Bp, Bu; // Magnetic fields
    int psip, psiu; // Psi field for CD
    int ep, eu; // electrons
    int psp, psu; // passives
};

// Emulate old names for possible stronger typing later,
// and for readability
// TODO specify ParArrayXD instead of generic?
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
#define KOKKOS_LAMBDA_4D KOKKOS_LAMBDA (const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_5D KOKKOS_LAMBDA (const int& m, const int& l, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_VARS KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i)
#define KOKKOS_LAMBDA_VEC KOKKOS_LAMBDA (const int &mu, const int &k, const int &j, const int &i)
// TODO separate macros for return type if this becomes a thing?  Or don't macro at all
#define KOKKOS_LAMBDA_1D_REDUCE KOKKOS_LAMBDA (const int &i, Real &local_result)
// This is used for timestep and divB, which are explicitly double.
#define KOKKOS_LAMBDA_2D_REDUCE KOKKOS_LAMBDA (const int &j, const int &i, double &local_result)
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
enum ReconstructionType{donor_cell=0, linear_mc, linear_vl, ppm, mp5, weno5, weno5_lower_poles};

// Floor codes are non-exclusive, so it makes little sense to use an enum
// Instead, we use bitflags, starting high enough that we can stick the enum in the bottom 5 bits
// See floors.hpp for explanations of the flags
#define HIT_FLOOR_GEOM_RHO 32
#define HIT_FLOOR_GEOM_U 64
#define HIT_FLOOR_B_RHO 128
#define HIT_FLOOR_B_U 256
#define HIT_FLOOR_TEMP 512
#define HIT_FLOOR_GAMMA 1024
#define HIT_FLOOR_KTOT 2048

// KHARMA UNIVERSAL FUNCTIONS

/**
 * Return whether a boundary is physical (i.e. border of the simulation) -- that is, not internal or periodic
 * Ironically, the zones in non-physical boundaries are "physical" i.e. bulk, non-ghost zones
 * 
 * Defined because UtoP needs to calculate primitives for real, domain zones -- that is, where this function returns false
 * TODO should this be a part of BoundaryFlag?
 */
KOKKOS_INLINE_FUNCTION bool is_physical_bound(parthenon::BoundaryFlag bflag) {
    // TODO error on undef?
    return bflag != parthenon::BoundaryFlag::block && bflag != parthenon::BoundaryFlag::periodic;
    //return false;
}

// TODO make into static inline like above?  Meh.
#if TRACE
#define FLAG(x) if(MPIRank0()) std::cout << x << std::endl;
#else
#define FLAG(x)
#endif