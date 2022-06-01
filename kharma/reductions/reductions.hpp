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

#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "types.hpp"

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
Real DomainSum(MeshData<Real> *md, const Real& radius);

// Then we define the macro which will generate all of our accretion rate calculations.
// This is a general (dangerous) macro which will generate an implementation of
// AccretionRate<Something>, given the arguments
// "Something" and "Function", which together specify a variable name, and the function
// to run inside the reduction

// And no, this can't just be a template: "Function" must be first defined within "AccretionRate",
// so that it can inherit the variable names (U, P, etc.) from the function context.
// That is, if we try to define "Function" outside and pass it as a template argument,
// the compiler has no idea what "U" means
// TODO this function needs a version/to be able to sum at any particular radius: test per-block X1<X<X2, find corresponding i.
#define MAKE_SUM2D_FN(name, fn) template<> inline Real AccretionRate<name>(MeshData<Real> *md, const int& i) { \
    Flag("Performing accretion reduction"); \
    auto pmesh = md->GetMeshPointer(); \
\
    Real result = 0.; \
    for (auto &pmb : pmesh->block_list) { \
        auto& rc = pmb->meshblock_data.Get(); \
        if (pmb->boundary_flag[parthenon::BoundaryFace::inner_x1] == BoundaryFlag::user) { \
            const auto& pars = pmb->packages.Get("GRMHD")->AllParams(); \
            const MetadataFlag isPrimitive = pars.Get<MetadataFlag>("PrimitiveFlag"); \
            PackIndexMap prims_map, cons_map; \
            const auto& P = rc->PackVariables(std::vector<MetadataFlag>{isPrimitive}, prims_map); \
            const auto& U = rc->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map); \
            const VarMap m_u(cons_map, true), m_p(prims_map, false); \
\
            const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma"); \
\
            IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior); \
            IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior); \
            IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior); \
            const auto& G = pmb->coords; \
\
            Real block_result; \
            Kokkos::Sum<Real> sum_reducer(block_result); \
            pmb->par_reduce("accretion_sum", kb.s, kb.e, jb.s, jb.e, ib.s+i, ib.s+i, \
                KOKKOS_LAMBDA_3D_REDUCE { \
                    FourVectors Dtmp; \
                    Real T[GR_DIM][GR_DIM]; \
                    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp); \
                    DLOOP1 Flux::calc_tensor(G, P, m_p, Dtmp, gam, k, j, i, mu, T[mu]); \
                    GReal gdA = G.dx3v(k) * G.dx2v(j) * G.gdet(Loci::center, j, i); \
                    GReal dA = G.dx3v(k) * G.dx2v(j); \
                    fn \
                } \
            , sum_reducer); \
            result += block_result; \
        } \
    } \
\
    Flag("Reduced"); \
\
    return result; \
}
// Re: B_P and B_U above, they need to not crash but can return nonsense:
// hence, just use an equivalent-size replacement.
// There may be more elegant solutions...

// Now we need some valid type names to use in distinguishing functions.
// The 'enum class' lines just serve to define an arbitrary name as some valid type,
// so that it can be used to distinguish between implementations of AccretionRate<X>.
// We could also have used different int values here, but type names seemed more elegant.

// We also provide some implementations.
// Each of the MAKE_ETC "calls" expands into an implementation of
// AccretionRate<Type> using the macro we just defined above.
// TODO These are GRHD/GRMHD *only* for now, they will require a generic Flux::calc_tensor,
// plus packing & passing
enum class Mdot : int;
MAKE_SUM2D_FN(Mdot,
    // TODO document values here. e.g.:
    // \dot{M} == \int rho * u^1 * gdet * dx2 * dx3
    local_result += -P(m_p.RHO, k, j, i) * Dtmp.ucon[1] * gdA;
)
enum class Edot : int;
MAKE_SUM2D_FN(Edot,
    // Edot == \int - T^1_0 * gdet * dx2 * dx3
    local_result += -T[X1DIR][X0DIR] * gdA;
)
enum class Ldot : int;
MAKE_SUM2D_FN(Ldot,
    // Ldot == \int T^1_3 * gdet * dx2 * dx3
    local_result += T[X1DIR][X3DIR] * gdA;
)
enum class Phi : int;
MAKE_SUM2D_FN(Phi,
    // phi == \int |*F^1^0| * gdet * dx2 * dx3 == \int |B1| * gdet * dx2 * dx3
    // Can also sum the hemispheres independently to be fancy (TODO?)
    local_result += 0.5 * fabs(U(m_u.B1, k, j, i)) * dA; // gdet is included in cons.B
)

// Then we can define the same with fluxes.
// The MAKE_SUM2D_FN macro pulls out pretty much any variable we could need here
enum class Mdot_Flux : int;
MAKE_SUM2D_FN(Mdot_Flux, local_result += -U.flux(X1DIR, m_u.RHO, k, j, i) * dA;)
enum class Edot_Flux : int;
MAKE_SUM2D_FN(Edot_Flux, local_result += (U.flux(X1DIR, m_u.UU, k, j, i) - U.flux(X1DIR, m_u.RHO, k, j, i)) * dA;)
enum class Ldot_Flux : int;
MAKE_SUM2D_FN(Ldot_Flux, local_result += U.flux(X1DIR, m_u.U3, k, j, i) * dA;)

// Finally, we define the reductions in the form Parthenon needs, picking particular
// variables and zones so that the resulting functions take only MeshData as an argument
inline Real MdotBound(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 0);}
inline Real MdotEH(MeshData<Real> *md) {return AccretionRate<Mdot>(md, 5);}
inline Real EdotBound(MeshData<Real> *md) {return AccretionRate<Edot>(md, 0);}
inline Real EdotEH(MeshData<Real> *md) {return AccretionRate<Edot>(md, 5);}
inline Real LdotBound(MeshData<Real> *md) {return AccretionRate<Ldot>(md, 0);}
inline Real LdotEH(MeshData<Real> *md) {return AccretionRate<Ldot>(md, 5);}
inline Real PhiBound(MeshData<Real> *md) {return AccretionRate<Phi>(md, 0);}
inline Real PhiEH(MeshData<Real> *md) {return AccretionRate<Phi>(md, 5);}

inline Real MdotBoundFlux(MeshData<Real> *md) {return AccretionRate<Mdot_Flux>(md, 0);}
inline Real MdotEHFlux(MeshData<Real> *md) {return AccretionRate<Mdot_Flux>(md, 5);}
inline Real EdotBoundFlux(MeshData<Real> *md) {return AccretionRate<Edot_Flux>(md, 0);}
inline Real EdotEHFlux(MeshData<Real> *md) {return AccretionRate<Edot_Flux>(md, 5);}
inline Real LdotBoundFlux(MeshData<Real> *md) {return AccretionRate<Ldot_Flux>(md, 0);}
inline Real LdotEHFlux(MeshData<Real> *md) {return AccretionRate<Ldot_Flux>(md, 5);}

// Now we repeat the whole process for reductions across the entire domain
// TODO could probably check blocks for containing/being within radius
// TODO could at least avoid calculating T in all zones

#define MAKE_SUM3D_FN(name, fn) template<> inline Real DomainSum<name>(MeshData<Real> *md, const Real& radius) { \
    Flag("Performing domain reduction"); \
    auto pmesh = md->GetMeshPointer(); \
\
    Real result = 0.; \
    for (auto &pmb : pmesh->block_list) { \
        auto& rc = pmb->meshblock_data.Get(); \
        if (pmb->boundary_flag[parthenon::BoundaryFace::inner_x1] == BoundaryFlag::user) { \
            const auto& pars = pmb->packages.Get("GRMHD")->AllParams(); \
            const MetadataFlag isPrimitive = pars.Get<MetadataFlag>("PrimitiveFlag"); \
            PackIndexMap prims_map, cons_map; \
            const auto& P = rc->PackVariables(std::vector<MetadataFlag>{isPrimitive}, prims_map); \
            const auto& U = rc->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map); \
            const VarMap m_u(cons_map, true), m_p(prims_map, false); \
\
            const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma"); \
\
            IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior); \
            IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior); \
            IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior); \
            const auto& G = pmb->coords; \
\
            Real block_result; \
            Kokkos::Sum<Real> sum_reducer(block_result); \
            pmb->par_reduce("domain_sum", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, \
                KOKKOS_LAMBDA_3D_REDUCE { \
                    FourVectors Dtmp; \
                    Real T[GR_DIM][GR_DIM]; \
                    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp); \
                    DLOOP1 Flux::calc_tensor(G, P, m_p, Dtmp, gam, k, j, i, mu, T[mu]); \
                    GReal gdV = G.dx3v(k) * G.dx2v(j) * G.dx1v(i) * G.gdet(Loci::center, j, i); \
                    GReal dV = G.dx3v(k) * G.dx2v(j) * G.dx1v(i); \
                    fn \
                } \
            , sum_reducer); \
            result += block_result; \
        } \
    } \
\
    Flag("Reduced"); \
\
    return result; \
}
enum class Mtot : int;
MAKE_SUM3D_FN(Mtot,
    // Within radius...
    GReal X[GR_DIM];
    G.coord_embed(k, j, i, Loci::face1, X);
    if (X[1] < radius) {
        local_result += U(m_u.RHO, k, j, i) * dV;
    }
)
enum class Ltot : int;
MAKE_SUM3D_FN(Ltot,
    GReal X[GR_DIM];
    G.coord_embed(k, j, i, Loci::face1, X);
    if (X[1] < radius) {
        local_result += U(m_u.U3, k, j, i) * dV;
    }
)
enum class Etot : int;
MAKE_SUM3D_FN(Etot,
    GReal X[GR_DIM];
    G.coord_embed(k, j, i, Loci::face1, X);
    if (X[1] < radius) {
        local_result += U(m_u.UU, k, j, i) * dV;
    }
)

// Luminosity proxy from (for example) Porth et al 2019.
// Notice that this will be totaled for *all zones*,
// but one could define a variable which checks sigma, G.coord_embed(), etc
enum class EHTLum : int;
MAKE_SUM3D_FN(EHTLum,
    // Within radius...
    GReal X[GR_DIM];
    G.coord_embed(k, j, i, Loci::face1, X);
    if (X[1] > radius) {
        Real rho = P(m_p.RHO, k, j, i);
        Real Pg = (gam - 1.) * P(m_p.UU, k, j, i);
        Real Bmag = sqrt(dot(Dtmp.bcon, Dtmp.bcov));
        Real j_eht = pow(rho, 3.) * pow(Pg, -2.) * exp(-0.2 * pow(rho * rho / (Bmag * Pg * Pg), 1./3.));
        local_result += j_eht * gdV;
    }
)

// Example of checking extra conditions before adding local results:
// sums total jet power only at exactly r=radius, for areas with sig > 1
// Split versions for e.g. E&M power only should calculate T manually for their case
enum class JetLum : int;
MAKE_SUM3D_FN(JetLum,
    // At r = radius, i.e. if our faces span acreoss it...
    GReal X_f[GR_DIM]; GReal X_b[GR_DIM];
    G.coord_embed(k, j, i, Loci::face1, X_b);
    G.coord_embed(k, j, i+1, Loci::face1, X_f);
    if (X_f[1] > radius && X_b[1] < radius) {
        // If sigma > 1...
        if ((dot(Dtmp.bcon, Dtmp.bcov) / P(m_p.RHO, k, j, i)) > 1.) {
            // Energy flux, like at EH. 2D integral jacobian.
            local_result += -T[X1DIR][X0DIR] * G.dx3v(k) * G.dx2v(j) * G.gdet(Loci::center, j, i);;
        }
    }
)

inline Real TotalM(MeshData<Real> *md) {return DomainSum<Mtot>(md, 50.);}
inline Real TotalE(MeshData<Real> *md) {return DomainSum<Etot>(md, 50.);}
inline Real TotalL(MeshData<Real> *md) {return DomainSum<Ltot>(md, 50.);}

inline Real TotalEHTLum(MeshData<Real> *md) {return DomainSum<EHTLum>(md, 50.);}
inline Real JetLum_50(MeshData<Real> *md) {return DomainSum<JetLum>(md, 50.);} // Recall this is *at* not *within*

inline int NPFlags(MeshData<Real> *md) {return CountPFlags(md, IndexDomain::entire, 0);}
inline int NFFlags(MeshData<Real> *md) {return CountFFlags(md, IndexDomain::interior, 0);}

} // namespace Reductions
