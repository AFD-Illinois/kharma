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

#include "debug.hpp"
#include "floors.hpp"
#include "pack.hpp"
#include "reconstruction.hpp"

// Package functions
#include "mhd_functions.hpp"
#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "electrons.hpp"

namespace Flux {
/**
 * Calculate dU/dt from a set of fluxes.
 * This combines Parthenon's "FluxDivergence" operation with the GRMHD source term
 * It also allows adding an arbitrary "wind" source term for stability
 *
 * @param rc is the current stage's container
 * @param dudt is the base container containing the global dUdt term
 */
TaskStatus ApplyFluxes(MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt);

/**
 * Fill all conserved variables (U) from primitive variables (P), over the whole grid.
 */
TaskStatus PrimToFlux(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire);

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
 * 
 * Memory-wise, this fills the "flux" portions of the "conserved" fields.  All fluxes are applied
 * together "ApplyFluxes," and the final fields are calculated by Parthenon in 
 * Also fills the "ctop" vector with the signal speed mhd_vchar -- used to estimate timestep later.
 * 
 * This function is defined in the header because it is templated on the reconstruction scheme and
 * direction.  Since there are only a few reconstruction schemes supported, and we will only ever
 * need fluxes in three directions, we can recompile the function for every combination.
 * This allows some extra optimization from knowing that dir != 0 in parcticular, and inlining
 * the particular reconstruction call we need.
 */
template <ReconstructionType Recon, int dir>
inline TaskStatus GetFlux(MeshBlockData<Real> *rc)
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
    const FloorPrescription floors = FloorPrescription(pmb->packages.Get("GRMHD")->AllParams());
    // B field package options
    const bool use_b_flux_ct = pmb->packages.AllPackages().count("B_FluxCT");
    const bool use_b_cd = pmb->packages.AllPackages().count("B_CD");
    const bool use_electrons = pmb->packages.AllPackages().count("Electrons");

    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const double ctop_max = pmb->packages.Get("Globals")->Param<Real>("ctop_max_last");

    // Fluxes in direction X1 should be calculated at face 1, likewise others
    Loci loc_tmp;
    switch (dir) {
    case X1DIR:
        loc_tmp = Loci::face1;
        break;
    case X2DIR:
        loc_tmp = Loci::face2;
        break;
    case X3DIR:
        loc_tmp = Loci::face3;
        break;
    }
    const Loci loc = loc_tmp;

    // VARIABLES
    auto& ctop = rc->GetFace("ctop").data;
    // Pack all primitive and conserved variables,
    MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables({isPrimitive}, prims_map);
    const auto& U = rc->PackVariablesAndFluxes({Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    const int nvar = U.GetDim(4);
    FLAG("Packed variables");

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
                        GRMHD::apply_geo_floors(G, Pl, m_p, gam, k, j, i, floors, loc);
                        GRMHD::apply_geo_floors(G, Pr, m_p, gam, k, j, i, floors, loc);
                    }
#if !FUSE_FLUX_KERNELS
                }
            );
            member.team_barrier();

            // LEFT FACES, final ctop
            parthenon::par_for_inner(member, is_l, ie_l,
                [&](const int& i) {
#endif
                    // LR -> flux
                    // Declare temporary vectors
                    FourVectors Dtmp;

                    // Left
                    GRMHD::calc_4vecs(G, Pl, m_p, k, j, i, loc, Dtmp);
                    GRMHD::prim_to_flux(G, Pl, m_p, Dtmp, gam, k, j, i, 0, Ul, m_u, loc);
                    GRMHD::prim_to_flux(G, Pl, m_p, Dtmp, gam, k, j, i, dir, Fl, m_u, loc);
                    if (use_b_flux_ct) {
                        B_FluxCT::prim_to_flux(G, Pl, m_p, Dtmp, k, j, i, 0, Ul, m_u, loc);
                        B_FluxCT::prim_to_flux(G, Pl, m_p, Dtmp, k, j, i, dir, Fl, m_u, loc);
                    } else if (use_b_cd) {
                        B_CD::prim_to_flux(G, Pl, m_p, Dtmp, k, j, i, 0, Ul, m_u, loc);
                        B_CD::prim_to_flux(G, Pl, m_p, Dtmp, k, j, i, dir, Fl, m_u, loc);
                    }
                    if (use_electrons) {
                        Electrons::prim_to_flux(G, Pl, m_p, Dtmp, k, j, i, 0, Ul, m_u, loc);
                        Electrons::prim_to_flux(G, Pl, m_p, Dtmp, k, j, i, dir, Fl, m_u, loc);
                    }

                    // Magnetosonic speeds
                    Real cmaxL, cminL;
                    GRMHD::vchar(G, Pl, m_p, Dtmp, gam, k, j, i, loc, dir, cmaxL, cminL);

#if !FUSE_FLUX_KERNELS
                    // Record speeds
                    cmax(i) = max(0., cmaxL);
                    cmin(i) = max(0., -cminL);
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
                    GRMHD::calc_4vecs(G, Pr, m_p, k, j, i, loc, Dtmp);
                    GRMHD::prim_to_flux(G, Pr, m_p, Dtmp, gam, k, j, i, 0, Ur, m_u, loc);
                    GRMHD::prim_to_flux(G, Pr, m_p, Dtmp, gam, k, j, i, dir, Fr, m_u, loc);
                    if (use_b_flux_ct) {
                        B_FluxCT::prim_to_flux(G, Pr, m_p, Dtmp, k, j, i, 0, Ur, m_u, loc);
                        B_FluxCT::prim_to_flux(G, Pr, m_p, Dtmp, k, j, i, dir, Fr, m_u, loc);
                    } else if (use_b_cd) {
                        B_CD::prim_to_flux(G, Pr, m_p, Dtmp, k, j, i, 0, Ur, m_u, loc);
                        B_CD::prim_to_flux(G, Pr, m_p, Dtmp, k, j, i, dir, Fr, m_u, loc);
                    }
                    if (use_electrons) {
                        Electrons::prim_to_flux(G, Pr, m_p, Dtmp, k, j, i, 0, Ur, m_u, loc);
                        Electrons::prim_to_flux(G, Pr, m_p, Dtmp, k, j, i, dir, Fr, m_u, loc);
                    }

                    // Magnetosonic speeds
                    Real cmaxR, cminR;
                    GRMHD::vchar(G, Pr, m_p, Dtmp, gam, k, j, i, loc, dir, cmaxR, cminR);

#if FUSE_FLUX_KERNELS
                    // Calculate cmax/min from local variables
                    cmax(i) = fabs(max(cmaxL,  cmaxR));
                    cmin(i) = fabs(max(-cminL, -cminR));

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
                        U.flux(dir, m_u.PSI, k, j, i) = llf(Fl(m_u.PSI,i), Fr(m_u.PSI,i), ctop_max, ctop_max, Ul(m_u.PSI,i), Ur(m_u.PSI,i));
                        U.flux(dir, m_u.B1+dir-1, k, j, i) = llf(Fl(m_u.B1+dir-1,i), Fr(m_u.B1+dir-1,i), ctop_max, ctop_max, Ul(m_u.B1+dir-1,i), Ur(m_u.B1+dir-1,i));
                    }
#else
                    // Calculate cmax/min based on comparison with cached values
                    cmax(i) = fabs(max(cmax(i),  cmaxR));
                    cmin(i) = fabs(max(cmin(i), -cminR));
#endif
                    // TODO is it faster to write ctop elsewhere?
                    ctop(dir, k, j, i) = max(cmax(i), cmin(i));
                }
            );
            member.team_barrier();

#if !FUSE_FLUX_KERNELS
            // Apply what we've calculated
            for (int p=0; p < nvar; ++p) {
                if (use_b_cd && (p == m_u.PSI || p == m_u.B1+dir-1)) {
                    // The unphysical variable psi and its corrections can propagate at the max speed for the stepsize, rather than the sound speed
                    // Since the speeds are the same it will always correspond to the LLF flux
                    parthenon::par_for_inner(member, is_l, ie_l,
                        [&](const int& i) {
                            U.flux(dir, p, k, j, i) = llf(Fl(p,i), Fr(p,i), ctop_max, ctop_max, Ul(p,i), Ur(p,i));
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

    FLAG(string_format("Finished recon and flux X%d", dir));
    return TaskStatus::complete;
}
}
