/* 
 *  File: reductions.hpp
 *  
 *  BSD 3-Clause License
 *  
 *  Copyright (c) 2020, AFD Group at UIUC
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *  
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *  
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include "debug.hpp"

#include "mhd_functions.hpp"

namespace Reductions {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// Okay so this requires a little explaining.
// The point is to share all the code we can between calculations of
// all of the EH or inner bound fluxes: mdot, edot, ldot, etc, etc, etc

// We start with a template, which will be used for all reductions
// The "typename" here is basically a flag to distinguish implementations
template<typename T>
Real AccretionRate(MeshData<Real> *md, const int& i);
template<typename T>
Real DomainSum(MeshData<Real> *md);

// Define the macro which will generate all of our accretion rate calculations.
// This is a general (dangerous) macro for an implementation of
// AccretionRate<Something>, which allows us to specify an ordered pair
// "Something", "Function" specifying a variable name, and what to use
// as the Kokkos lambda function in the template
#define MAKE_SUM2D_FN(name, fn) template<> inline Real AccretionRate<name>(MeshData<Real> *md, const int& i) { \
    FLAG("Performing accretion reduction"); \
    auto pmesh = md->GetMeshPointer(); \
\
    Real result = 0.; \
    for (auto &pmb : pmesh->block_list) { \
        auto& rc = pmb->meshblock_data.Get(); \
        if (pmb->boundary_flag[parthenon::BoundaryFace::inner_x1] == BoundaryFlag::user) { \
            GridScalar rho_U = rc->Get("cons.rho").data; \
            GridScalar u_U = rc->Get("cons.u").data; \
            GridScalar uvec_U = rc->Get("cons.uvec").data; \
            GridScalar B_U = rc->Get("cons.B").data; \
            GridScalar rho_P = rc->Get("prims.rho").data; \
            GridScalar u_P = rc->Get("prims.u").data; \
            GridScalar uvec_P = rc->Get("prims.uvec").data; \
            GridScalar B_P = rc->Get("prims.B").data; \
            GridScalar rho_F = rc->Get("cons.rho").flux[1]; \
            GridScalar u_F = rc->Get("cons.u").flux[1]; \
            GridScalar uvec_F = rc->Get("cons.uvec").flux[1]; \
            const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma"); \
\
            IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior); \
            IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior); \
            IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior); \
            const auto& G = pmb->coords; \
\
            Real result_local; \
            Kokkos::Sum<Real> sum_reducer(result_local); \
            pmb->par_reduce("accretion_sum", kb.s, kb.e, jb.s, jb.e, ib.s+i, ib.s+i, \
            fn, sum_reducer); \
            result += result_local; \
        } \
    } \
\
    FLAG("Reduced"); \
\
    return result; \
}

// Now we need some type names. These just serve to tell our implementations apart with
// keywords that the compiler can understand.
// We also provide some implementations.  Each of these expands to a definition of
// AccretionRate<Type> using the loop body listed in the macro
enum class Mdot : int;
MAKE_SUM2D_FN(Mdot, KOKKOS_LAMBDA_3D_REDUCE { local_result += -rho_P(k, j, i) * uvec_P(0, k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i) * G.gdet(Loci::center, j, i); })
enum class Edot : int;
MAKE_SUM2D_FN(Edot, KOKKOS_LAMBDA_3D_REDUCE { local_result += -uvec_U(0, k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i); })
enum class Ldot : int;
MAKE_SUM2D_FN(Ldot, KOKKOS_LAMBDA_3D_REDUCE { local_result += uvec_U(2, k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i); })
enum class Phi : int;
MAKE_SUM2D_FN(Phi, KOKKOS_LAMBDA_3D_REDUCE { local_result += 0.5 * fabs(B_U(0, k, j, i)) * G.dx3v(k) * G.dx2v(j); })

// Versions with fluxes.  Note we pulled out pointers to both the conserved varibles and the fluxes,
// so that we can define these using the same macro
enum class Mdot_Flux : int;
MAKE_SUM2D_FN(Mdot_Flux, KOKKOS_LAMBDA_3D_REDUCE { local_result += -rho_F(k, j, i) * G.dx3v(k) * G.dx2v(j); })
enum class Edot_Flux : int;
MAKE_SUM2D_FN(Edot_Flux, KOKKOS_LAMBDA_3D_REDUCE { local_result += (u_F(k, j, i) - rho_F(k, j, i)) * G.dx3v(k) * G.dx2v(j); })
enum class Ldot_Flux : int;
MAKE_SUM2D_FN(Ldot_Flux, KOKKOS_LAMBDA_3D_REDUCE { local_result += uvec_F(2, k, j, i) * G.dx3v(k) * G.dx2v(j); })

// Finally, we can specialize to particular zones and name our functions
inline Real MdotBound(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 0);}
inline Real MdotEH(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 5);}
inline Real EdotBound(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 0);}
inline Real EdotEH(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 5);}
inline Real LdotBound(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 0);}
inline Real LdotEH(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 5);}

inline Real MdotBoundFlux(MeshData<Real> *md) {return AccretionRate<Mdot_Flux>(md, 0);}
inline Real MdotEHFlux(MeshData<Real> *md) {return AccretionRate<Mdot_Flux>(md, 5);}
inline Real EdotBoundFlux(MeshData<Real> *md) {return AccretionRate<Ldot_Flux>(md, 0);}
inline Real EdotEHFlux(MeshData<Real> *md) {return AccretionRate<Ldot_Flux>(md, 5);}
inline Real LdotBoundFlux(MeshData<Real> *md) {return AccretionRate<Edot_Flux>(md, 0);}
inline Real LdotEHFlux(MeshData<Real> *md) {return AccretionRate<Edot_Flux>(md, 5);}

// Same as above, but for the whole domain
#define MAKE_SUM3D_FN(name, fn) template<> inline Real DomainSum<name>(MeshData<Real> *md) { \
    FLAG("Performing domain reduction"); \
    auto pmesh = md->GetMeshPointer(); \
\
    Real result = 0.; \
    for (auto &pmb : pmesh->block_list) { \
        auto& rc = pmb->meshblock_data.Get(); \
        GridScalar rho_U = rc->Get("cons.rho").data; \
        GridScalar u_U = rc->Get("cons.u").data; \
        GridScalar uvec_U = rc->Get("cons.uvec").data; \
        GridScalar B_U = rc->Get("cons.B").data; \
        GridScalar rho_P = rc->Get("prims.rho").data; \
        GridScalar u_P = rc->Get("prims.u").data; \
        GridScalar uvec_P = rc->Get("prims.uvec").data; \
        GridScalar B_P = rc->Get("prims.B").data; \
        const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma"); \
\
        IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior); \
        IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior); \
        IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior); \
        const auto& G = pmb->coords; \
\
        Real result_local; \
        Kokkos::Sum<Real> sum_reducer(result_local); \
        pmb->par_reduce("domain_sum", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, \
        fn, sum_reducer); \
        result += result_local; \
    } \
\
    FLAG("Reduced"); \
\
    return result; \
}
enum class Mtot : int;
MAKE_SUM3D_FN(Mtot, KOKKOS_LAMBDA_3D_REDUCE { local_result += rho_U(k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i); })
enum class Ltot : int;
MAKE_SUM3D_FN(Ltot, KOKKOS_LAMBDA_3D_REDUCE { local_result += uvec_U(2, k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i); })
enum class Etot : int;
MAKE_SUM3D_FN(Etot, KOKKOS_LAMBDA_3D_REDUCE { local_result += u_U(k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i); })

enum class EHTLum : int;
MAKE_SUM3D_FN(EHTLum, (KOKKOS_LAMBDA_3D_REDUCE {
    Real rho = rho_P(k, j, i);
    Real Pg = (gam - 1.) * u_P(k, j, i);
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, uvec_P, B_P, k, j, i, Loci::center, Dtmp);
    Real Bmag = sqrt(dot(Dtmp.bcon, Dtmp.bcov));
    Real j_eht = pow(rho, 3.) * pow(Pg, -2.) * exp(-0.2 * pow(rho * rho / (Bmag * Pg * Pg), 1./3.));
    local_result += j_eht * G.dx3v(k) * G.dx2v(j) * G.dx1v(i) * G.gdet(Loci::center, j, i);
}))
enum class JetLum : int;
MAKE_SUM3D_FN(JetLum, (KOKKOS_LAMBDA_3D_REDUCE {
    Real rho = rho_P(k, j, i);
    Real Pg = (gam - 1.) * u_P(k, j, i);
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, uvec_P, B_P, k, j, i, Loci::center, Dtmp);
    Real bsq = dot(Dtmp.bcon, Dtmp.bcov);
    double sig = bsq / rho_P(k, j, i);
    if (sig > 1.) {
        Real uvec_loc[NVEC] = {uvec_P(0, k, j, i), uvec_P(1, k, j, i), uvec_P(2, k, j, i)};
        Real B_loc[NVEC] = {B_P(0, k, j, i), B_P(1, k, j, i), B_P(2, k, j, i)};
        Real rho_ut, T[GR_DIM];
        GRMHD::p_to_u_loc(G, 0., 0., uvec_loc, B_loc, gam, k, j, i, rho_ut, T);
        local_result += -T[1] * G.dx3v(k) * G.dx2v(j);
    }
}))


inline Real TotalM(MeshData<Real> *md) {return DomainSum<Mtot>(md);}
inline Real TotalL(MeshData<Real> *md) {return DomainSum<Ltot>(md);}
inline Real TotalE(MeshData<Real> *md) {return DomainSum<Etot>(md);}

inline Real TotalEHTLum(MeshData<Real> *md) {return DomainSum<Etot>(md);}
inline Real TotalJetLum(MeshData<Real> *md) {return DomainSum<Etot>(md);}

inline int NPFlags(MeshData<Real> *md) {return CountPFlags(md, IndexDomain::entire, 0);}
inline int NFFlags(MeshData<Real> *md) {return CountFFlags(md, IndexDomain::interior, 0);}

} // namespace Reductions