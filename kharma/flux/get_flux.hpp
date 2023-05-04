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
 * This function also fills the "Flux.cmax" & "Flux.cmin" vectors with the signal speeds,
 * and potentially the "Flux.vl" and "Flux.vr" vectors with the fluid velocities
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
    const auto& cmax  = md->PackVariables(std::vector<std::string>{"Flux.cmax"});
    const auto& cmin  = md->PackVariables(std::vector<std::string>{"Flux.cmin"});
    const auto& P_all = md->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    const auto& U_all = md->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    //Flag(md, "Packed variables");

    const auto& Pl_all = md->PackVariables(std::vector<std::string>{"Flux.Pl"});
    const auto& Pr_all = md->PackVariables(std::vector<std::string>{"Flux.Pr"});
    const auto& Ul_all = md->PackVariables(std::vector<std::string>{"Flux.Ul"});
    const auto& Ur_all = md->PackVariables(std::vector<std::string>{"Flux.Ur"});
    const auto& Fl_all = md->PackVariables(std::vector<std::string>{"Flux.Fl"});
    const auto& Fr_all = md->PackVariables(std::vector<std::string>{"Flux.Fr"});

    // Get sizes
    const int n1 = pmb0->cellbounds.ncellsi(IndexDomain::entire);
    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, cmax.GetDim(5) - 1};
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
    // Allocate enough to cache prims, conserved, and fluxes, for left and right faces,
    // plus temporaries inside reconstruction (most use 1, WENO5 uses none, linear_vl uses a bunch)
    const size_t recon_scratch_bytes = (2 + 1*(Recon != KReconstruction::Type::weno5) +
                                            4*(Recon == KReconstruction::Type::linear_vl)) * var_size_in_bytes;
    const size_t flux_scratch_bytes = 3 * var_size_in_bytes;

    Flag(md, "Recon kernel");
    // This isn't a pmb0->par_for_outer because Parthenon's current overloaded definitions
    // do not accept three pairs of bounds, which we need in order to iterate over blocks
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "calc_flux_recon", pmb0->exec_space,
        recon_scratch_bytes, scratch_level, block.s, block.e, kl.s, kl.e, jl.s, jl.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
            const auto& G = U_all.GetCoords(b);
            ScratchPad2D<Real> Pl_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Pr_s(member.team_scratch(scratch_level), nvar, n1);

            // Wrapper for a big switch statement between reconstruction schemes. Possibly slow.
            // This function is generally a lot of if statements
            KReconstruction::reconstruct<Recon, dir>(member, P_all(b), k, j, il.s, il.e, Pl_s, Pr_s);

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

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
                }
            );
            member.team_barrier();

            // Copy out state (TODO(BSP) eliminate)
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, il.s, il.e,
                    [&](const int& i) {
                        Pl_all(b, p, k, j, i) = Pl_s(p, i);
                        Pr_all(b, p, k, j, i) = Pr_s(p, i);
                    }
                );
            }

        }
    );

    Flag(md, "PtoU Left");
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "calc_flux_left", pmb0->exec_space,
        flux_scratch_bytes, scratch_level, block.s, block.e, kl.s, kl.e, jl.s, jl.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
            const auto& G = U_all.GetCoords(b);
            ScratchPad2D<Real> Pl_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Ul_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Fl_s(member.team_scratch(scratch_level), nvar, n1);

            // Copy in state (TODO(BSP) eliminate)
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, il.s, il.e,
                    [&](const int& i) {
                        Pl_s(p, i) = Pl_all(b, p, k, j, i);
                    }
                );
            }
            member.team_barrier();

            // LEFT FACES
            parthenon::par_for_inner(member, il.s, il.e,
                [&](const int& i) {
                    auto Pl = Kokkos::subview(Pl_s, Kokkos::ALL(), i);
                    auto Ul = Kokkos::subview(Ul_s, Kokkos::ALL(), i);
                    auto Fl = Kokkos::subview(Fl_s, Kokkos::ALL(), i);
                    // Declare temporary vectors
                    FourVectors Dtmp;

                    // Left
                    GRMHD::calc_4vecs(G, Pl, m_p, j, i, loc, Dtmp);
                    Flux::prim_to_flux(G, Pl, m_p, Dtmp, emhd_params, gam, j, i, 0, Ul, m_u, loc);
                    Flux::prim_to_flux(G, Pl, m_p, Dtmp, emhd_params, gam, j, i, dir, Fl, m_u, loc);

                    // Magnetosonic speeds
                    Real cmaxL, cminL;
                    Flux::vchar(G, Pl, m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxL, cminL);

                    // Record speeds
                    cmax(b, dir-1, k, j, i) = m::max(0., cmaxL);
                    cmin(b, dir-1, k, j, i) = m::max(0., -cminL);
                }
            );
            member.team_barrier();

            // Copy out state
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, il.s, il.e,
                    [&](const int& i) {
                        Ul_all(b, p, k, j, i) = Ul_s(p, i);
                        Fl_all(b, p, k, j, i) = Fl_s(p, i);
                    }
                );
            }
        }
    );

    Flag(md, "PtoU Right");
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "calc_flux_right", pmb0->exec_space,
        flux_scratch_bytes, scratch_level, block.s, block.e, kl.s, kl.e, jl.s, jl.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& b, const int& k, const int& j) {
            const auto& G = U_all.GetCoords(b);
            ScratchPad2D<Real> Pr_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Ur_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Fr_s(member.team_scratch(scratch_level), nvar, n1);

            // Copy in state (TODO(BSP) eliminate)
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, il.s, il.e,
                    [&](const int& i) {
                        Pr_s(p, i) = Pr_all(b, p, k, j, i);
                    }
                );
            }
            member.team_barrier();

            // RIGHT FACES, finalize signal speed
            parthenon::par_for_inner(member, il.s, il.e,
                [&](const int& i) {
                    auto Pr = Kokkos::subview(Pr_s, Kokkos::ALL(), i);
                    auto Ur = Kokkos::subview(Ur_s, Kokkos::ALL(), i);
                    auto Fr = Kokkos::subview(Fr_s, Kokkos::ALL(), i);
                    // Declare temporary vectors
                    FourVectors Dtmp;
                    // Right
                    GRMHD::calc_4vecs(G, Pr, m_p, j, i, loc, Dtmp);
                    Flux::prim_to_flux(G, Pr, m_p, Dtmp, emhd_params, gam, j, i, 0, Ur, m_u, loc);
                    Flux::prim_to_flux(G, Pr, m_p, Dtmp, emhd_params, gam, j, i, dir, Fr, m_u, loc);

                    // Magnetosonic speeds
                    Real cmaxR, cminR;
                    Flux::vchar(G, Pr, m_p, Dtmp, gam, emhd_params, k, j, i, loc, dir, cmaxR, cminR);

                    // Calculate cmax/min based on comparison with cached values
                    cmax(b, dir-1, k, j, i) = m::abs(m::max(cmax(b, dir-1, k, j, i),  cmaxR));
                    cmin(b, dir-1, k, j, i) = m::abs(m::max(cmin(b, dir-1, k, j, i), -cminR));
                }
            );
            member.team_barrier();

            // Copy out state
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, il.s, il.e,
                    [&](const int& i) {
                        Ur_all(b, p, k, j, i) = Ur_s(p, i);
                        Fr_all(b, p, k, j, i) = Fr_s(p, i);
                    }
                );
            }

        }
    );

    Flag(md, "Riemann kernel");
    pmb0->par_for("flux_solve", block.s, block.e, 0, nvar-1, kl.s, kl.e, jl.s, jl.e, il.s, il.e,
        KOKKOS_LAMBDA(const int& b, const int& p, const int& k, const int& j, const int& i) {
            // Apply what we've calculated
            // TODO OTHER FLUXES AGAIN
            U_all(b).flux(dir, p, k, j, i) = llf(Fl_all(b, p, k, j, i), Fr_all(b, p, k, j, i),
                                                 cmax(b, dir-1, k, j, i), cmin(b, dir-1, k, j, i),
                                                 Ul_all(b, p, k, j, i), Ur_all(b, p, k, j, i));


        }
    );

    Flag(md, "Finished recon and flux");
    return TaskStatus::complete;
}

} // Flux
