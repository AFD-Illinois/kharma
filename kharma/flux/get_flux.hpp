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
#pragma once

#include "flux.hpp"

#include "domain.hpp"
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
    // Pointers
    auto pmesh = md->GetMeshPointer();
    auto pmb0  = md->GetBlockData(0)->GetBlockPointer();
    auto& packages = pmb0->packages;
    // Exit on trivial operations
    const int ndim = pmesh->ndim;
    if (ndim < 3 && dir == X3DIR) return TaskStatus::complete;
    if (ndim < 2 && dir == X2DIR) return TaskStatus::complete;

    Flag("GetFlux_"+std::to_string(dir));

    // Options
    const auto& pars       = packages.Get("Driver")->AllParams();
    const auto& mhd_pars   = packages.Get("GRMHD")->AllParams();
    const auto& globals    = packages.Get("Globals")->AllParams();
    const bool use_hlle    = pars.Get<bool>("use_hlle");

    // TODO make this an option in Flux package
    const bool reconstruction_floors = packages.AllPackages().count("Floors") &&
                                       (Recon == KReconstruction::Type::weno5);
    Floors::Prescription floors_temp;
    if (reconstruction_floors) {
        // Apply post-reconstruction floors.
        // Only enabled for WENO since it is not TVD, and only when other
        // floors are enabled.
        const auto& floor_pars = packages.Get("Floors")->AllParams();
        // Pull out a struct of just the actual floor values for speed
        floors_temp = Floors::Prescription(floor_pars);
    }
    const Floors::Prescription& floors = floors_temp;

    const Real gam = mhd_pars.Get<Real>("gamma");

    // Check whether we're using constraint-damping
    // (which requires that a variable be propagated at ctop_max)
    const bool use_b_cd = packages.AllPackages().count("B_CD");
    const double ctop_max = (use_b_cd) ? packages.Get("B_CD")->Param<Real>("ctop_max_last") : 0.0;

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(packages);

    const Loci loc = loc_of(dir);

    // Pack variables.  Keep ctop separate
    PackIndexMap prims_map, cons_map;
    const auto& cmax  = md->PackVariables(std::vector<std::string>{"Flux.cmax"});
    const auto& cmin  = md->PackVariables(std::vector<std::string>{"Flux.cmin"});
    // TODO maybe all WithFluxes vars, split into cell & face?
    const auto& P_all = md->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
    const auto& U_all = md->PackVariablesAndFluxes(std::vector<MetadataFlag>{Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    const auto& Pl_all = md->PackVariables(std::vector<std::string>{"Flux.Pl"});
    const auto& Pr_all = md->PackVariables(std::vector<std::string>{"Flux.Pr"});
    const auto& Ul_all = md->PackVariables(std::vector<std::string>{"Flux.Ul"});
    const auto& Ur_all = md->PackVariables(std::vector<std::string>{"Flux.Ur"});
    const auto& Fl_all = md->PackVariables(std::vector<std::string>{"Flux.Fl"});
    const auto& Fr_all = md->PackVariables(std::vector<std::string>{"Flux.Fr"});

    // Get the domain size
    const IndexRange3 b = KDomain::GetRange(md, IndexDomain::interior, -1, 2);
    // Get other sizes we need
    const int n1 = pmb0->cellbounds.ncellsi(IndexDomain::entire);
    const IndexRange block = IndexRange{0, cmax.GetDim(5) - 1};
    const int nvar = U_all.GetDim(4);

    if (globals.Get<int>("verbose") > 2) {
        std::cout << "Calculating fluxes for " << cmax.GetDim(5) << " blocks, "
                << nvar << " variables (" << P_all.GetDim(4) << " primitives)" << std::endl;
        m_u.print(); m_p.print();
        emhd_params.print();
    }

    // Allocate scratch space
    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    const size_t var_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);
    // Allocate enough to cache prims, conserved, and fluxes, for left and right faces,
    // plus temporaries inside reconstruction (most use 1, WENO5 uses none, linear_vl uses a bunch)
    const size_t recon_scratch_bytes = (2 + 1*(Recon != KReconstruction::Type::weno5) +
                                            4*(Recon == KReconstruction::Type::linear_vl)) * var_size_in_bytes;
    const size_t flux_scratch_bytes = 3 * var_size_in_bytes;

    // This isn't a pmb0->par_for_outer because Parthenon's current overloaded definitions
    // do not accept three pairs of bounds, which we need in order to iterate over blocks
    Flag("GetFlux_"+std::to_string(dir)+"_recon");
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "calc_flux_recon", pmb0->exec_space,
        recon_scratch_bytes, scratch_level, block.s, block.e, b.ks, b.ke, b.js, b.je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& bl, const int& k, const int& j) {
            const auto& G = U_all.GetCoords(bl);
            ScratchPad2D<Real> Pl_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Pr_s(member.team_scratch(scratch_level), nvar, n1);

            // We template on reconstruction type to avoid a big switch statement here.
            // Instead, a version of GetFlux() is generated separately for each reconstruction/direction pair.
            // See reconstruction.hpp for all the implementations.
            KReconstruction::ReconstructRow<Recon, dir>(member, P_all(bl), k, j, b.is, b.ie, Pl_s, Pr_s);

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            parthenon::par_for_inner(member, b.is, b.ie,
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
                parthenon::par_for_inner(member, b.is, b.ie,
                    [&](const int& i) {
                        Pl_all(bl, p, k, j, i) = Pl_s(p, i);
                        Pr_all(bl, p, k, j, i) = Pr_s(p, i);
                    }
                );
            }
            member.team_barrier();
        }
    );
    EndFlag();

    // If we have B field on faces, we must replace reconstructed version with that
    if (pmb0->packages.AllPackages().count("B_CT")) {  // TODO if variable "cons.fB"?
        const auto& Bf  = md->PackVariables(std::vector<std::string>{"cons.fB"});
        const TopologicalElement face = (dir == 1) ? F1 : ((dir == 2) ? F2 : F3);
        pmb0->par_for("replace_face", block.s, block.e, b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA(const int& bl, const int& k, const int& j, const int& i) {
                const auto& G = U_all.GetCoords(bl);
                const double bf = Bf(bl, face, 0, k, j, i) / G.gdet(loc, j, i);
                Pl_all(bl, m_p.B1+dir-1, k, j, i) = bf;
                Pr_all(bl, m_p.B1+dir-1, k, j, i) = bf;
            }
        );
    }

    // Now that this is split, we add the biggest TODO in KHARMA
    // TODO per-package prim_to_flux?  Is that slower?
    // At least, we need to template on vchar/stress-energy T type

    Flag("GetFlux_"+std::to_string(dir)+"_left");
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "calc_flux_left", pmb0->exec_space,
        flux_scratch_bytes, scratch_level, block.s, block.e, b.ks, b.ke, b.js, b.je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& bl, const int& k, const int& j) {
            const auto& G = U_all.GetCoords(bl);
            ScratchPad2D<Real> Pl_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Ul_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Fl_s(member.team_scratch(scratch_level), nvar, n1);

            // Copy in state (TODO(BSP) eliminate)
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, b.is, b.ie,
                    [&](const int& i) {
                        Pl_s(p, i) = Pl_all(bl, p, k, j, i);
                    }
                );
            }
            member.team_barrier();

            // LEFT FACES
            parthenon::par_for_inner(member, b.is, b.ie,
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
                    cmax(bl, dir-1, k, j, i) = m::max(0., cmaxL);
                    cmin(bl, dir-1, k, j, i) = m::max(0., -cminL);
                }
            );
            member.team_barrier();

            // Copy out state
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, b.is, b.ie,
                    [&](const int& i) {
                        Ul_all(bl, p, k, j, i) = Ul_s(p, i);
                        Fl_all(bl, p, k, j, i) = Fl_s(p, i);
                    }
                );
            }
        }
    );
    EndFlag();

    Flag("GetFlux_"+std::to_string(dir)+"_right");
    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "calc_flux_right", pmb0->exec_space,
        flux_scratch_bytes, scratch_level, block.s, block.e, b.ks, b.ke, b.js, b.je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& bl, const int& k, const int& j) {
            const auto& G = U_all.GetCoords(bl);
            ScratchPad2D<Real> Pr_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Ur_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> Fr_s(member.team_scratch(scratch_level), nvar, n1);

            // Copy in state (TODO(BSP) eliminate)
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, b.is, b.ie,
                    [&](const int& i) {
                        Pr_s(p, i) = Pr_all(bl, p, k, j, i);
                    }
                );
            }
            member.team_barrier();

            // RIGHT FACES, finalize signal speed
            parthenon::par_for_inner(member, b.is, b.ie,
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
                    cmax(bl, dir-1, k, j, i) = m::abs(m::max(cmax(bl, dir-1, k, j, i),  cmaxR));
                    cmin(bl, dir-1, k, j, i) = m::abs(m::max(cmin(bl, dir-1, k, j, i), -cminR));
                }
            );
            member.team_barrier();

            // Copy out state
            for (int p=0; p < nvar; ++p) {
                parthenon::par_for_inner(member, b.is, b.ie,
                    [&](const int& i) {
                        Ur_all(bl, p, k, j, i) = Ur_s(p, i);
                        Fr_all(bl, p, k, j, i) = Fr_s(p, i);
                    }
                );
            }

        }
    );
    EndFlag();

    // Apply what we've calculated
    Flag("GetFlux_"+std::to_string(dir)+"_riemann");
    if (use_hlle) { // More fluxes would need a template
        pmb0->par_for("flux_hlle", block.s, block.e, 0, nvar-1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA(const int& bl, const int& p, const int& k, const int& j, const int& i) {
                U_all(bl).flux(dir, p, k, j, i) = hlle(Fl_all(bl, p, k, j, i), Fr_all(bl, p, k, j, i),
                                                      cmax(bl, dir-1, k, j, i), cmin(bl, dir-1, k, j, i),
                                                      Ul_all(bl, p, k, j, i), Ur_all(bl, p, k, j, i));


            }
        );
    } else {
        pmb0->par_for("flux_llf", block.s, block.e, 0, nvar-1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA(const int& bl, const int& p, const int& k, const int& j, const int& i) {
                U_all(bl).flux(dir, p, k, j, i) = llf(Fl_all(bl, p, k, j, i), Fr_all(bl, p, k, j, i),
                                                     cmax(bl, dir-1, k, j, i), cmin(bl, dir-1, k, j, i),
                                                     Ul_all(bl, p, k, j, i), Ur_all(bl, p, k, j, i));


            }
        );
    }
    EndFlag();

    // Save the face velocities for upwinding/CT later
    // TODO only for certain GS'05
    if (packages.AllPackages().count("B_CT")) {
        Flag("GetFlux_"+std::to_string(dir)+"_store_vel");
        const auto& vl_all = md->PackVariables(std::vector<std::string>{"Flux.vl"});
        const auto& vr_all = md->PackVariables(std::vector<std::string>{"Flux.vr"});
        TopologicalElement face = (dir == 1) ? F1 : (dir == 2) ? F2 : F3;
        pmb0->par_for("flux_llf", block.s, block.e, 0, NVEC-1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA(const int& bl, const int& v, const int& k, const int& j, const int& i) {
                vl_all(bl, face, v, k, j, i) = Pl_all(bl, m_p.U1+v, k, j, i);
                vr_all(bl, face, v, k, j, i) = Pr_all(bl, m_p.U1+v, k, j, i);
            }
        );
        EndFlag();
    }

    EndFlag();
    return TaskStatus::complete;
}

} // Flux
