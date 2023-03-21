/* 
 *  File: flux.hpp
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
#include "flux.hpp"

#include "floors_functions.hpp"

namespace Flux {

/**
 * @brief Reconstruct the values of primitive variables at left and right of each zone face,
 * find the corresponding conserved variables and their fluxes through the face
 *
 * @param md the current stage MeshData container, holding pointers to all variable data
 *
 * Memory-wise, this fills the "flux" portions of the "conserved" fields.  These will be used
 * over the course of the step to calculate an update to the zone-centered values.
 * This function also fills the "ctop" vector with the signal speed mhd_vchar,
 * used to estimate the timestep later.
 * 
 * This function is defined in the header because it is templated on the reconstruction scheme and
 * direction.  Since there are only a few reconstruction schemes supported, and we will only ever
 * need fluxes in three directions, we can recompile the function for every combination.
 * This allows some extra optimization from knowing that dir != 0 in parcticular, and inlining
 * the particular reconstruction call we need.
 */
template <KReconstruction::Type Recon, int dir>
inline TaskStatus GetFlux(MeshData<Real> *md)
{
    Flag(md, "Recon and flux");
    // Pointers
    auto pmesh = md->GetMeshPointer();
    auto pmb0  = md->GetBlockData(0)->GetBlockPointer();
    // Exit on trivial operations
    const int ndim = pmesh->ndim;
    if (ndim < 3 && dir == X3DIR) return TaskStatus::complete;
    if (ndim < 2 && dir == X2DIR) return TaskStatus::complete;

    // Options
    const auto& pars       = pmb0->packages.Get("Driver")->AllParams();
    const auto& mhd_pars   = pmb0->packages.Get("GRMHD")->AllParams();
    const auto& globals    = pmb0->packages.Get("Globals")->AllParams();
    const bool use_hlle    = pars.Get<bool>("use_hlle");

    const bool reconstruction_floors = pmb0->packages.AllPackages().count("Floors") &&
                                       (Recon == KReconstruction::Type::weno5);
    Floors::Prescription floors_temp;
    if (reconstruction_floors) {
        // Apply post-reconstruction floors.
        // Only enabled for WENO since it is not TVD, and only when other
        // floors are enabled.
        const auto& floor_pars = pmb0->packages.Get("Floors")->AllParams();
        // Pull out a struct of just the actual floor values for speed
        floors_temp = Floors::Prescription(floor_pars);
    }
    const Floors::Prescription& floors = floors_temp;

    const Real gam = mhd_pars.Get<Real>("gamma");

    // Check whether we're using constraint-damping
    // (which requires that a variable be propagated at ctop_max)
    const bool use_b_cd = pmb0->packages.AllPackages().count("B_CD");
    const double ctop_max = (use_b_cd) ? pmb0->packages.Get("B_CD")->Param<Real>("ctop_max_last") : 0.0;

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb0->packages);

    const Loci loc = loc_of(dir);

    // Pack variables.  Keep ctop separate
    PackIndexMap prims_map, cons_map;
    const auto& ctop  = md->PackVariables(std::vector<std::string>{"ctop"});
    const auto& P_all = md->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const auto& U_all = md->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    //Flag(md, "Packed variables");

    // Get sizes
    const int n1 = pmb0->cellbounds.ncellsi(IndexDomain::entire);
    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, ctop.GetDim(5) - 1};
    const int nvar = U_all.GetDim(4);
    // 1-zone halo in nontrivial dimensions
    // We leave is/ie, js/je, ks/ke with their usual definitions for consistency, and define
    // the loop bounds separately to include the appropriate halo
    // TODO halo 2 "shouldn't" crash but does.  Artifact of switch to faces?
    const IndexRange il = IndexRange{ib.s - 1, ib.e + 1};
    const IndexRange jl = (ndim > 1) ? IndexRange{jb.s - 1, jb.e + 1} : jb;
    const IndexRange kl = (ndim > 2) ? IndexRange{kb.s - 1, kb.e + 1} : kb;

    // Allocate scratch space
    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    const size_t var_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);
    const size_t speed_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(1, n1);
    // Allocate enough to cache prims, conserved, and fluxes, for left and right faces,
    // plus temporaries inside reconstruction (most use 1, WENO5 uses none, linear_vl uses a bunch)
    // Then add cmax and cmin!
    const size_t total_scratch_bytes = (6 + 1*(Recon != KReconstruction::Type::weno5) +
                                            4*(Recon == KReconstruction::Type::linear_vl)) * var_size_in_bytes
                                        + 2 * speed_size_in_bytes;

    Flag(md, "Flux kernel");
    // This isn't a pmb0->par_for_outer because Parthenon's current overloaded definitions
    // do not accept three pairs of bounds, which we need in order to iterate over blocks
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "calc_flux", pmb0->exec_space,
        total_scratch_bytes, scratch_level, block.s, block.e, kl.s, kl.e, jl.s, jl.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
            const auto& G = U_all.GetCoords(b);
            ScratchPad2D<Real> Pl_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Pr_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Ul_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Ur_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Fl_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Fr_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad1D<Real> cmax(member.team_scratch(scratch_level), n1);
            ScratchPad1D<Real> cmin(member.team_scratch(scratch_level), n1);

            // Wrapper for a big switch statement between reconstruction schemes. Possibly slow.
            // This function is generally a lot of if statements
            KReconstruction::reconstruct<Recon, dir>(member, G, P_all(b), k, j, il.s, il.e, Pl_s, Pr_s);

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            // Calculate conserved fluxes at centers & faces
            parthenon::par_for_inner(member, il.s, il.e,
                [&](const int& i) {
                    auto Pl = Kokkos::subview(Pl_s, Kokkos::ALL(), i);
                    auto Pr = Kokkos::subview(Pr_s, Kokkos::ALL(), i);
                    // Apply floors to the *reconstructed* primitives, because without TVD
                    // we have no guarantee they remotely resemble the *centered* primitives
                    if (reconstruction_floors) {
                        Floors::apply_geo_floors(G, Pl, m_p, gam, j, i, floors, loc);
                        Floors::apply_geo_floors(G, Pr, m_p, gam, j, i, floors, loc);
                    }
#if !FUSE_FLUX_KERNELS
                }
            );
            member.team_barrier();

            // LEFT FACES, final ctop
            parthenon::par_for_inner(member, il.s, il.e,
                [&](const int& i) {
                    auto Pl = Kokkos::subview(Pl_s, Kokkos::ALL(), i);
#endif
                    auto Ul = Kokkos::subview(Ul_s, Kokkos::ALL(), i);
                    auto Fl = Kokkos::subview(Fl_s, Kokkos::ALL(), i);
                    // LR -> flux
                    // Declare temporary vectors
                    FourVectors Dtmp;

                    // Left
                    GRMHD::calc_4vecs(G, Pl, m_p, j, i, loc, Dtmp);
                    Flux::prim_to_flux(G, Pl, m_p, Dtmp, emhd_params, gam, j, i, 0, Ul, m_u, loc);
                    Flux::prim_to_flux(G, Pl, m_p, Dtmp, emhd_params, gam, j, i, dir, Fl, m_u, loc);

                    // Magnetosonic speeds
                    Real cmaxL, cminL;
                    Flux::vchar(G, Pl, m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxL, cminL);

#if !FUSE_FLUX_KERNELS
                    // Record speeds
                    cmax(i) = m::max(0., cmaxL);
                    cmin(i) = m::max(0., -cminL);
                }
            );
            member.team_barrier();

            // RIGHT FACES, final ctop
            parthenon::par_for_inner(member, il.s, il.e,
                [&](const int& i) {
                    // LR -> flux
                    // Declare temporary vectors
                    FourVectors Dtmp;
                    auto Pr = Kokkos::subview(Pr_s, Kokkos::ALL(), i);
#endif
                    auto Ur = Kokkos::subview(Ur_s, Kokkos::ALL(), i);
                    auto Fr = Kokkos::subview(Fr_s, Kokkos::ALL(), i);
                    // Right
                    // TODO GRMHD/GRHD versions of this
                    GRMHD::calc_4vecs(G, Pr, m_p, j, i, loc, Dtmp);
                    Flux::prim_to_flux(G, Pr, m_p, Dtmp, emhd_params, gam, j, i, 0, Ur, m_u, loc);
                    Flux::prim_to_flux(G, Pr, m_p, Dtmp, emhd_params, gam, j, i, dir, Fr, m_u, loc);

                    // Magnetosonic speeds
                    Real cmaxR, cminR;
                    Flux::vchar(G, Pr, m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxR, cminR);

#if FUSE_FLUX_KERNELS
                    // Calculate cmax/min from local variables
                    cmax(i) = m::abs(m::max(cmaxL,  cmaxR));
                    cmin(i) = m::abs(m::max(-cminL, -cminR));

                    if (use_hlle) {
                        for (int p=0; p < nvar; ++p)
                            U_all(b).flux(dir, p, k, j, i) = hlle(Fl(p), Fr(p), cmax(i), cmin(i), Ul(p), Ur(p));
                    } else {
                        for (int p=0; p < nvar; ++p)
                            U_all(b).flux(dir, p, k, j, i) = llf(Fl(p), Fr(p), cmax(i), cmin(i), Ul(p), Ur(p));
                    }
                    if (use_b_cd) {
                        // The unphysical variable psi and its corrections can propagate at the max speed
                        // for the stepsize, rather than the sound speed
                        // Since the speeds are the same it will always correspond to the LLF flux
                        U_all(b).flux(dir, m_u.PSI, k, j, i) = llf(Fl(m_u.PSI), Fr(m_u.PSI), ctop_max, ctop_max, Ul(m_u.PSI), Ur(m_u.PSI));
                        U_all(b).flux(dir, m_u.B1+dir-1, k, j, i) = llf(Fl(m_u.B1+dir-1), Fr(m_u.B1+dir-1), ctop_max, ctop_max, Ul(m_u.B1+dir-1), Ur(m_u.B1+dir-1));
                    }
#else
                    // Calculate cmax/min based on comparison with cached values
                    cmax(i) = m::abs(m::max(cmax(i),  cmaxR));
                    cmin(i) = m::abs(m::max(cmin(i), -cminR));
#endif
                    // TODO is it faster to write ctop elsewhere?
                    ctop(b, dir-1, k, j, i) = m::max(cmax(i), cmin(i));
                }
            );
            member.team_barrier();

#if !FUSE_FLUX_KERNELS
            // Apply what we've calculated
            for (int p=0; p < nvar; ++p) {
                if (use_b_cd && (p == m_u.PSI || p == m_u.B1+dir-1)) {
                    // The unphysical variable psi and its corrections can propagate at the max speed for the stepsize, rather than the sound speed
                    // Since the speeds are the same it will always correspond to the LLF flux
                    parthenon::par_for_inner(member, il.s, il.e,
                        [&](const int& i) {
                            U_all(b).flux(dir, p, k, j, i) = llf(Fl_s(p,i), Fr_s(p,i), ctop_max, ctop_max, Ul_s(p,i), Ur_s(p,i));
                        }
                    );
                } else if (use_hlle) {
                    // Option to try HLLE fluxes for everything else
                    parthenon::par_for_inner(member, il.s, il.e,
                        [&](const int& i) {
                            U_all(b).flux(dir, p, k, j, i) = hlle(Fl_s(p,i), Fr_s(p,i), cmax(i), cmin(i), Ul_s(p,i), Ur_s(p,i));
                        }
                    );
                } else {
                    // Or LLF, probably safest option
                    parthenon::par_for_inner(member, il.s, il.e,
                        [&](const int& i) {
                            U_all(b).flux(dir, p, k, j, i) = llf(Fl_s(p,i), Fr_s(p,i), cmax(i), cmin(i), Ul_s(p,i), Ur_s(p,i));
                        }
                    );
                }
            }
#endif
        }
    );

    Flag(md, "Finished recon and flux");
    return TaskStatus::complete;
}

} // Flux
