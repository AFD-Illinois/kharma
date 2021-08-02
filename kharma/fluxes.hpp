/* 
 *  File: fluxes.hpp
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

#include "decs.hpp"

#include <parthenon/parthenon.hpp>

// Package functions
#include "mhd_functions.hpp"
#include "b_flux_ct.hpp"
#include "b_cd.hpp"

#include "debug.hpp"
#include "floors.hpp"
#include "reconstruction.hpp"
#include "source.hpp"

#define INLINE_FLUXCALC 1
#define INLINE_RL_CALC 1

extern double ctop_max;

namespace Flux {
/**
 * Calculate dU/dt from a set of fluxes.
 * This combines Parthenon's "FluxDivergence" operation with the GRMHD source term
 * It also allows adding an arbitrary "wind" source term for stability
 *
 * @param rc is the current stage's container
 * @param dudt is the base container containing the global dUdt term
 * @param dt the timestep *by which to advance*, i.e. sometimes dt/2 or less for RK2+ time integration
 */
TaskStatus ApplyFluxes(MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt, const Real& dt);

// Fluxes a.k.a. "Approximate Riemann Solvers"
// More complex solvers require speed estimates not calculable completely from
// invariants, necessitating frame transformations and related madness.
// These have identical signatures, so that we could runtime relink w/variant like coordinate_embedding

// Local Lax-Friedrichs flux (usual, more stable)
KOKKOS_INLINE_FUNCTION Real llf(const Real& fluxL, const Real& fluxR, const Real& cmax, 
                                const Real& cmin, const Real& Ul, const Real& Ur)
{
    Real ctop = max(cmax, cmin);
    return 0.5 * (fluxL + fluxR - ctop * (Ur - Ul));
}
// Harten, Lax, van Leer, & Einfeldt flux (early problems but not extensively studied since)
KOKKOS_INLINE_FUNCTION Real hlle(const Real& fluxL, const Real& fluxR, const Real& cmax,
                                const Real& cmin, const Real& Ul, const Real& Ur)
{
    return (cmax*fluxL + cmin*fluxR - cmax*cmin*(Ur - Ul)) / (cmax + cmin);
}

/**
 * Reconstruct the values of primitive variables at left and right zone faces,
 * find the corresponding conserved variables and their fluxes through the zone faces
 *
 * @param rc the current stage container, holding pointers to all variable data
 * @param dt the timestep *by which to advance*, i.e. sometimes dt/2 or less for RK2+ time integration
 * 
 * Memory-wise, this fills the "flux" portions of the "conserved" fields.  All fluxes are applied
 * together "ApplyFluxes," and the final fields are calculated by Parthenon in 
 * Also fills the "ctop" vector with the signal speed mhd_vchar -- used to estimate timestep later.
 */
template <ReconstructionType Recon, int dir>
inline TaskStatus GetFlux(MeshBlockData<Real> *rc, const Real& dt)
{
    FLAG(string_format("Recon and flux X%d", dir));
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    // 1-zone halo in nontrivial dimensions. Don't bother with fluxes in trivial dimensions
    // We leave is/ie, js/je, ks/ke with their usual definitions for consistency, and define
    // the loop bounds separately to include the appropriate halo
    const int ndim = pmb->pmy_mesh->ndim;
    int halo = 1;
    const int ks_l = (ndim > 2) ? ks - halo : ks;
    const int ke_l = (ndim > 2) ? ke + halo : ke;
    const int js_l = (ndim > 1) ? js - halo : js;
    const int je_l = (ndim > 1) ? je + halo : je;
    const int is_l = is - halo;
    const int ie_l = ie + halo;
    // Don't calculate fluxes we won't use
    if (ndim < 3 && dir == X3DIR) return TaskStatus::complete;
    if (ndim < 2 && dir == X2DIR) return TaskStatus::complete;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    if (0) { // No amount of verbosity warrants this abuse,
            // but if the code segfaults these numbers are a likely culprit
        int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
        int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
        cout << string_format("Domain: %d-%d %d-%d %d-%d", is_l, ie_l, js_l, je_l, ks_l, ke_l) << endl;
        cout << string_format("Total: %dx%dx%d", n1, n2, n3) << endl;
    }

    // OPTIONS
    bool use_hlle = pmb->packages.Get("GRMHD")->Param<bool>("use_hlle");
    // Pull out a struct of just the actual floor values for speed
    FloorPrescription floors = FloorPrescription(pmb->packages.Get("GRMHD")->AllParams());
    const ReconstructionType& recon = pmb->packages.Get("GRMHD")->Param<ReconstructionType>("recon");
    // B field package options
    const bool use_b_flux_ct = pmb->packages.AllPackages().count("B_FluxCT");
    const bool use_b_cd = pmb->packages.AllPackages().count("B_CD");

    // If we're reducing the order of reconstruction at the poles, determine whether this
    // block is on a pole (otherwise BoundaryFlag will be "block")
    bool is_inner_x2 = pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect;
    bool is_outer_x2 = pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect;

    auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // Fluxes in direction X1 should be calculated at face 1, likewise others
    // TODO adapt if we ever go non-Cartesian
    Loci loc;
    switch (dir) {
    case X1DIR:
        loc = Loci::face1;
        break;
    case X2DIR:
        loc = Loci::face2;
        break;
    case X3DIR:
        loc = Loci::face3;
        break;
    }

    // VARIABLES
    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;
    // Pack all primitive and conserved variables, 
    MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables({isPrimitive}, prims_map);
    const auto& U = rc->PackVariablesAndFluxes({Metadata::Conserved}, cons_map);
    FLAG("Packed variables");
    // Indices into these packs, for addressing specific variables in the loop
    // Parthenon returns -1 index for keys not found in a pack
    struct varmap mf;
    mf.u = cons_map["c.c.bulk.cons"].first;
    mf.p = prims_map["c.c.bulk.prims"].first;
    mf.Bu = cons_map["c.c.bulk.B_con"].first;
    mf.Bp = prims_map["c.c.bulk.B_prim"].first;
    mf.psiu = cons_map["c.c.bulk.psi_cd_con"].first;
    mf.psip = prims_map["c.c.bulk.psi_cd_prim"].first;
    const struct varmap m = mf;
    const int nvar = P.GetDim(4);

    // TODO something other than this
    const double ctop_max_l = ctop_max;

    // SCRATCH SPACE
    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    const size_t var_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);
    const size_t speed_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(1, n1);
    // Allocate enough to cache prims, conserved, and fluxes, for left and right faces,
    // plus temporaries inside reconstruction (linear_vl uses a bunch)
    // Then add cmax and cmin!
    const size_t total_scratch_bytes = (6 + 1 + 4*(Recon == ReconstructionType::linear_vl)) * var_size_in_bytes
                                        + 2 * speed_size_in_bytes;

    FLAG("Flux kernel");
    pmb->par_for_outer(string_format("flux_x%d", dir), total_scratch_bytes, scratch_level,
        ks_l, ke_l, js_l, je_l,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
            ScratchPad2D<Real> Pl(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Pr(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Ul(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Ur(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Fl(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Fr(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad1D<Real> cmax(member.team_scratch(scratch_level), n1);
            ScratchPad1D<Real> cmin(member.team_scratch(scratch_level), n1);

            // Wrapper for a big switch statement between reconstruction schemes. Possibly slow.
            // This function is generally a lot of if statements
            KReconstruction::reconstruct<Recon, dir>(member, G, P, k, j, is_l, ie_l, Pl, Pr);

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            // Calculate conserved fluxes at centers & faces
            parthenon::par_for_inner(member, is_l, ie_l,
                [&](const int& i) {
                    // Apply floors to the *reconstructed* primitives, because without TVD
                    // we have no guarantee they remotely resemble the *centered* primitives
                    if (Recon == ReconstructionType::weno5) {
                        apply_geo_floors(G, Pl, m, gam, k, j, i, floors, loc);
                        apply_geo_floors(G, Pr, m, gam, k, j, i, floors, loc);
                    }

                    // LR -> flux
                    // Declare temporary vectors
                    FourVectors Dtmp;

                    // Left
                    GRMHD::calc_4vecs(G, Pl, m, k, j, i, loc, Dtmp);
                    GRMHD::prim_to_flux(G, Pl, m, Dtmp, gam, k, j, i, loc, 0, Ul);
                    GRMHD::prim_to_flux(G, Pl, m, Dtmp, gam, k, j, i, loc, dir, Fl);

                    Real cmaxL, cminL;
                    GRMHD::vchar(G, Pl, m, Dtmp, gam, k, j, i, loc, dir, cmaxL, cminL);
                    // Record speeds
                    cmax(i) = max(0., cmaxL);
                    cmin(i) = max(0., -cminL);

                    if (use_b_flux_ct) {
                        B_FluxCT::prim_to_u(G, Pl, m, Dtmp, j, i, loc, Ul);
                        B_FluxCT::prim_to_flux(G, Pl, m, Dtmp, j, i, loc, dir, Fl);
                    } else if (use_b_cd) {
                        B_CD::prim_to_u(G, Pl, m, Dtmp, j, i, loc, Ul);
                        B_CD::prim_to_flux(G, Pl, m, Dtmp, j, i, loc, dir, Fl);
                    }
#if !INLINE_RL_CALC
                }
            );
            member.team_barrier();

            // RIGHT FACES, final ctop
            parthenon::par_for_inner(member, is_l, ie_l,
                [&](const int& i) {
                    // LR -> flux
                    // Declare temporary vectors
                    FourVectors Dtmp;
#endif
                    // Right
                    GRMHD::calc_4vecs(G, Pr, m, k, j, i, loc, Dtmp);
                    GRMHD::prim_to_flux(G, Pr, m, Dtmp, gam, k, j, i, loc, 0, Ur);
                    GRMHD::prim_to_flux(G, Pr, m, Dtmp, gam, k, j, i, loc, dir, Fr);

                    // Calculate the max and replace as we go
                    Real cmaxR, cminR;
                    GRMHD::vchar(G, Pr, m, Dtmp, gam, k, j, i, loc, dir, cmaxR, cminR);
                    cmax(i) = fabs(max(cmax(i),  cmaxR));
                    cmin(i) = fabs(max(cmin(i), -cminR));
                    ctop(dir, k, j, i) = max(cmax(i), cmin(i));

                    if (use_b_flux_ct) { // TODO Moooore templating?
                        B_FluxCT::prim_to_u(G, Pr, m, Dtmp, j, i, loc, Ur);
                        B_FluxCT::prim_to_flux(G, Pr, m, Dtmp, j, i, loc, dir, Fr);
                    } else if (use_b_cd) {
                        B_CD::prim_to_u(G, Pr, m, Dtmp, j, i, loc, Ur);
                        B_CD::prim_to_flux(G, Pr, m, Dtmp, j, i, loc, dir, Fr);
                    }
#if INLINE_FLUXCALC
                    if (use_hlle) {
                        for (int p=0; p < nvar; ++p)
                            U.flux(dir, p, k, j, i) = hlle(Fl(p,i), Fr(p,i), cmax(i), cmin(i), Ul(p,i), Ur(p,i));
                    } else {
                        for (int p=0; p < nvar; ++p)
                            U.flux(dir, p, k, j, i) = llf(Fl(p,i), Fr(p,i), cmax(i), cmin(i), Ul(p,i), Ur(p,i));
                    }
                    if (use_b_cd) {
                        // The unphysical variable psi and its corrections can propagate at the max speed
                        // for the stepsize, rather than the sound speed
                        // Since the speeds are the same it will always correspond to the LLF flux
                        U.flux(dir, m.psiu, k, j, i) = llf(Fl(m.psiu,i), Fr(m.psiu,i), ctop_max_l, ctop_max_l, Ul(m.psiu,i), Ur(m.psiu,i));
                        U.flux(dir, m.Bu+dir-1, k, j, i) = llf(Fl(m.Bu+dir-1,i), Fr(m.Bu+dir-1,i), ctop_max_l, ctop_max_l, Ul(m.Bu+dir-1,i), Ur(m.Bu+dir-1,i));
                    }
#endif
                }
            );
            // OTHERS HERE
            member.team_barrier();

#if !INLINE_FLUXCALC
            // Apply what we've calculated
            for (int p=0; p < nvar; ++p) {
                if (use_b_cd && (p == m.psiu || p == m.Bu+dir-1)) {
                    // The unphysical variable psi and its corrections can propagate at the max speed for the stepsize, rather than the sound speed
                    // Since the speeds are the same it will always correspond to the LLF flux
                    parthenon::par_for_inner(member, is_l, ie_l,
                        [&](const int& i) {
                            U.flux(dir, p, k, j, i) = llf(Fl(p,i), Fr(p,i), ctop_max_l, ctop_max_l, Ul(p,i), Ur(p,i));
                        }
                    );
                } else if (use_hlle) {
                    // Option to try HLLE fluxes for everything else
                    parthenon::par_for_inner(member, is_l, ie_l,
                        [&](const int& i) {
                            U.flux(dir, p, k, j, i) = hlle(Fl(p,i), Fr(p,i), cmax(i), cmin(i), Ul(p,i), Ur(p,i));
                        }
                    );
                } else {
                    // Or LLF, probably safest option
                    parthenon::par_for_inner(member, is_l, ie_l,
                        [&](const int& i) {
                                U.flux(dir, p, k, j, i) = llf(Fl(p,i), Fr(p,i), cmax(i), cmin(i), Ul(p,i), Ur(p,i));
                        }
                    );
                }
            }
#endif
        }
    );

    if (pmb->packages.Get("GRMHD")->Param<int>("extra_checks") > 0) {
        CheckNaN(rc, dir);
    }

    FLAG(string_format("Finished recon and flux X%d", dir));
    return TaskStatus::complete;
}
}
