/* 
 *  File: fluxes.cpp
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

#include "fluxes.hpp"

#include <parthenon/parthenon.hpp>

// Package functions
#include "mhd_functions.hpp"
#include "b_flux_ct_functions.hpp"
#include "b_cd_glm_functions.hpp"

#include "debug.hpp"
#include "floors.hpp"
#include "reconstruction.hpp"
#include "source.hpp"

using namespace parthenon;

// Look, it seemed like a good idea at the time
extern double ctop_max;

// Identical calls in case relinking proves faster
KOKKOS_INLINE_FUNCTION Real llf(const Real& fluxL, const Real& fluxR, const Real& cmax, 
                                const Real& cmin, const Real& Ul, const Real& Ur)
{
    Real ctop = max(cmax, cmin);
    return 0.5 * (fluxL + fluxR - ctop * (Ur - Ul));
}

KOKKOS_INLINE_FUNCTION Real hlle(const Real& fluxL, const Real& fluxR, const Real& cmax,
                                const Real& cmin, const Real& Ul, const Real& Ur)
{
    return (cmax*fluxL + cmin*fluxR - cmax*cmin*(Ur - Ul)) / (cmax + cmin);
}

TaskStatus Flux::GetFlux(MeshBlockData<Real> *rc, const int& dir, const Real& dt)
{
    FLAG(string_format("Recon and flux X%d", dir));
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    // 1-zone halo in nontrivial dimensions. Don't calculate/allow fluxes in trivial dimensions
    // Leave is/ie, js/je, ks/ke with their usual definitions for consistency, and define the loop
    // bounds separately to include the appropriate halo
    int halo = 1;
    int ks_l = (ks == 0) ? 0 : ks - halo;
    int ke_l = (ke == 0) ? 0 : ke + halo;
    if (ke == 0 && dir == X3DIR) return TaskStatus::complete;
    int js_l = (js == 0) ? 0 : js - halo;
    int je_l = (je == 0) ? 0 : je + halo;
    if (je == 0 && dir == X2DIR) return TaskStatus::complete;
    int is_l = is - halo;
    int ie_l = ie + halo;
    const int ndim = pmb->pmy_mesh->ndim;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    if (0) { // No amount of verbosity warrants this abuse,
             // but if the code crashes these numbers are a likely culprit
        int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
        int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
        cout << string_format("Domain: %d-%d %d-%d %d-%d", is_l, ie_l, js_l, je_l, ks_l, ke_l) << endl;
        cout << string_format("Total: %dx%dx%d", n1, n2, n3) << endl;
    }

    // OPTIONS
    // Pull out a struct of just the actual floor values for speed
    FloorPrescription floors = FloorPrescription(pmb->packages.Get("GRMHD")->AllParams());

    // And cache whether we should reduce reconstruction order on the X2 bound
    bool is_inner_x2 = pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect;
    bool is_outer_x2 = pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect;

    bool use_hlle = pmb->packages.Get("GRMHD")->Param<bool>("use_hlle");

    // B field evolution
    const bool use_b_flux_ct = pmb->packages.AllPackages().count("B_FluxCT") > 0;
    const bool use_b_cd_glm = pmb->packages.AllPackages().count("B_CD_GLM") > 0;
    // TODO should be obvious
    auto& G = pmb->coords;

    // Figure out the correct divB propagation speed c_h
    const Real c_h = !use_b_cd_glm ? 0 :
                            clip(pmb->packages.Get("B_CD_GLM")->Param<Real>("c_h_factor") * ctop_max, 
                                pmb->packages.Get("B_CD_GLM")->Param<Real>("c_h_low"),
                                pmb->packages.Get("B_CD_GLM")->Param<Real>("c_h_high"));
    // if (use_b_cd_glm) {
    //     cout << "c_h is " << c_h << " and ctop is " << ctop_max << endl;
    // }

    EOS* eos = pmb->packages.Get("GRMHD")->Param<EOS*>("eos");
    const ReconstructionType& recon = pmb->packages.Get("GRMHD")->Param<ReconstructionType>("recon");

    // VARIABLES
    MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables({isPrimitive}, prims_map);
    const auto& U = rc->PackVariablesAndFluxes({Metadata::Conserved}, cons_map);
    // Indices into these packs, for addressing specific variables in the loop
    const int cons_start = cons_map["c.c.bulk.cons"].first;
    const int prims_start = prims_map["c.c.bulk.prims"].first;
    const int B_con_start = cons_map["c.c.bulk.B_con"].first;
    const int B_prim_start = prims_map["c.c.bulk.B_prim"].first;
    const int psi_con_start = use_b_cd_glm ? cons_map["c.c.bulk.psi_cd_con"].first : 0;
    const int psi_prim_start = use_b_cd_glm ? prims_map["c.c.bulk.psi_cd_prim"].first : 0;
    const int nvar = P.GetDim(4);
    if (0) {
        cout << "Fluxes: Prims " << prims_start << " B_P " << B_prim_start << " psi_p " << psi_prim_start << endl;
        cout << "Cons " << cons_start << " B_con " << B_con_start << " psi_u " << psi_con_start << " nvar " << nvar << endl;
    }

    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;

    // Calculate fluxes at matching face/direction
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

    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);

    pmb->par_for_outer(string_format("flux_x%d", dir), 7 * scratch_size_in_bytes, scratch_level,
        ks_l, ke_l, js_l, je_l,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
            ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, n1);

            // Wrapper for a big switch statement between reconstruction schemes. Possibly slow.
            // This function is generally a lot of if statements
            KReconstruction::reconstruct(recon, member, G, P, n1, k, j, is_l, ie_l, dir, ql, qr);

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            parthenon::par_for_inner(member, is_l, ie_l,
                [&](const int& i) {
                    // Unpack primitives into immediate values for this zone,
                    // TODO these and below should be scratch-allocated
                    Real p_l[NPRIM], p_r[NPRIM], bp_l[NVEC], bp_r[NVEC], psi_l, psi_r;
                    PLOOP {
                        p_l[p] = ql(prims_start + p, i);
                        p_r[p] = qr(prims_start + p, i);
                    }
                    VLOOP {
                        bp_l[v] = ql(B_prim_start + v, i);
                        bp_r[v] = qr(B_prim_start + v, i);
                    }
                    if (use_b_cd_glm) {
                        psi_l = ql(psi_prim_start, i);
                        psi_r = qr(psi_prim_start, i);
                    }

                    // Apply floors to the *reconstructed* primitives, because we have no
                    // guarantee they remotely resemble the *centered* primitives
                    // TODO can we get away with less?
                    apply_floors(G, p_l, bp_l, eos, k, j, i, floors, loc);
                    //apply_ceilings(G, pl, bp_l, eos, k, j, i, floors, loc);
                    apply_floors(G, p_r, bp_r, eos, k, j, i, floors, loc);
                    //apply_ceilings(G, pr, bp_r, eos, k, j, i, floors, loc);

                    // LR -> flux
                    // Speeds and temporaries
                    FourVectors Dtmp;
                    Real cmaxL, cmaxR, cminL, cminR;
                    Real cmin, cmax;
                    // Hold the flux of all conserved variables locally
                    Real fluxL[16], fluxR[16];
                    Real Ul[16], Ur[16];

                    // TODO Note that the only dependencies here are that the 4vecs be done first.
                    // Otherwise the many prim_to_flux/vchar calls are all independent
                    // One could also perform these as individual par_for_inner calls over i,
                    // might preserve vectorization better

                    // Left
                    GRMHD::calc_4vecs(G, p_l, bp_l, k, j, i, loc, Dtmp);
                    Real rho = p_l[prims::rho];
                    Real u = p_l[prims::u];
                    Real pgas = eos->p(rho, u);
                    GRMHD::prim_to_flux(G, rho, u, pgas, Dtmp, k, j, i, loc, 0, &(Ul[cons_start]));
                    GRMHD::prim_to_flux(G, rho, u, pgas, Dtmp, k, j, i, loc, dir, &(fluxL[cons_start]));
                    GRMHD::vchar(G, rho, u, pgas, Dtmp, eos, k, j, i, loc, dir, cmaxL, cminL);

                    if (use_b_flux_ct) {
                        B_FluxCT::prim_to_flux(G, Dtmp, bp_l, k, j, i, loc, 0, &(Ul[B_con_start]));
                        B_FluxCT::prim_to_flux(G, Dtmp, bp_l, k, j, i, loc, dir, &(fluxL[B_con_start]));
                    } else if (use_b_cd_glm) {
                        B_CD_GLM::prim_to_flux(G, Dtmp, bp_l, psi_l, k, j, i, loc, 0, c_h,
                                                &(Ul[B_con_start]), &(Ul[psi_con_start]));
                        B_CD_GLM::prim_to_flux(G, Dtmp, bp_l, psi_l, k, j, i, loc, dir, c_h,
                                                &(fluxL[B_con_start]), &(fluxL[psi_con_start]));
                    }

                    // Right
                    GRMHD::calc_4vecs(G, p_r, bp_r, k, j, i, loc, Dtmp);
                    rho = p_r[prims::rho];
                    u = p_r[prims::u];
                    pgas = eos->p(rho, u);
                    GRMHD::prim_to_flux(G, rho, u, pgas, Dtmp, k, j, i, loc, 0, &(Ur[cons_start]));
                    GRMHD::prim_to_flux(G, rho, u, pgas, Dtmp, k, j, i, loc, dir, &(fluxR[cons_start]));
                    GRMHD::vchar(G, rho, u, pgas, Dtmp, eos, k, j, i, loc, dir, cmaxR, cminR);

                    if (use_b_flux_ct) {
                        B_FluxCT::prim_to_flux(G, Dtmp, bp_r, k, j, i, loc, 0, &(Ur[B_con_start]));
                        B_FluxCT::prim_to_flux(G, Dtmp, bp_r, k, j, i, loc, dir, &(fluxR[B_con_start]));
                    } else if (use_b_cd_glm) {
                        B_CD_GLM::prim_to_flux(G, Dtmp, bp_r, psi_r, k, j, i, loc, 0, c_h,
                                                &(Ur[B_con_start]), &(Ur[psi_con_start]));
                        B_CD_GLM::prim_to_flux(G, Dtmp, bp_r, psi_r, k, j, i, loc, dir, c_h,
                                                &(fluxR[B_con_start]), &(fluxR[psi_con_start]));
                    }

                    cmax = fabs(max(max(0.,  cmaxL),  cmaxR));
                    cmin = fabs(max(max(0., -cminL), -cminR));
                    ctop(dir, k, j, i) = max(cmax, cmin);

                    // Calculate and apply fluxes to each conserved variable
                    // The unphysical variable psi and its corrections can propagate at the chosen max speed c_h rather than the sound speed
                    if (use_hlle) {
                        for (int p=0; p < nvar; ++p) {
                            if (use_b_cd_glm && (p == psi_con_start || p == B_con_start+dir-1)) {
                                // The unphysical variable psi and its corrections can propagate at the chosen max speed c_h rather than the sound speed
                                U.flux(dir, p, k, j, i) = llf(fluxL[p], fluxR[p], c_h, c_h, Ul[p], Ur[p]);
                            } else {
                                U.flux(dir, p, k, j, i) = hlle(fluxL[p], fluxR[p], cmax, cmin, Ul[p], Ur[p]);
                            }
                        }
                    } else {
                        for (int p=0; p < nvar; ++p) {
                            if (use_b_cd_glm && (p == psi_con_start || p == B_con_start+dir-1)) {
                                U.flux(dir, p, k, j, i) = llf(fluxL[p], fluxR[p], c_h, c_h, Ul[p], Ur[p]);
                            } else {
                                U.flux(dir, p, k, j, i) = llf(fluxL[p], fluxR[p], cmax, cmin, Ul[p], Ur[p]);
                            }
                        }
                    }
                    // Above should correspond to Dedner eq.42 i.e.
                    // Real gdet = G.gdet(loc, j, i);
                    // U.flux(dir, psi_con_start, k, j, i) = 0.5*(c_h*c_h*(Ur[B_con_start+dir-1] + Ul[B_con_start+dir-1]) - c_h*(psi_r*gdet - psi_l*gdet));
                    // U.flux(dir, B_con_start+dir-1, k, j, i = 0.5*(psi_r*gdet + psi_l*gdet - c_h*(Ur[B_con_start+dir-1] - Ul[B_con_start+dir-1]));
                }
            );
        }
    );

    if (pmb->packages.Get("GRMHD")->Param<int>("extra_checks") > 0) {
        CheckNaN(rc, dir);
    }

    FLAG(string_format("Finished recon and flux X%d", dir));
    return TaskStatus::complete;
}

TaskStatus Flux::ApplyFluxes(MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt, const Real& dt)
{
    FLAG("Applying fluxes");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const int ndim = pmb->pmy_mesh->ndim;

    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars B_P = rc->Get("c.c.bulk.B_prim").data;

    PackIndexMap cons_map;
    auto U = rc->PackVariablesAndFluxes({Metadata::Conserved}, cons_map);
    auto dUdt = dudt->PackVariables({Metadata::Conserved});
    int nvar = U.GetDim(4);
    const int cons_start = cons_map["c.c.bulk.cons"].first;

    auto& G = pmb->coords;
    EOS* eos = pmb->packages.Get("GRMHD")->Param<EOS*>("eos");

    // TODO move wind to separate package/function?
    bool wind_term = pmb->packages.Get("GRMHD")->Param<bool>("wind_term");
    Real wind_n = pmb->packages.Get("GRMHD")->Param<Real>("wind_n");
    Real wind_Tp = pmb->packages.Get("GRMHD")->Param<Real>("wind_Tp");
    int wind_pow = pmb->packages.Get("GRMHD")->Param<int>("wind_pow");
    Real wind_ramp_start = pmb->packages.Get("GRMHD")->Param<Real>("wind_ramp_start");
    Real wind_ramp_end = pmb->packages.Get("GRMHD")->Param<Real>("wind_ramp_end");
    Real current_wind_n = wind_n;
    // if (wind_ramp_end > 0.0) {
    //     current_wind_n = min((tm.time - wind_ramp_start) / (wind_ramp_end - wind_ramp_start), 1.0) * wind_n;
    // } else {
    //     current_wind_n = wind_n;
    // }

    pmb->par_for("apply_fluxes", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Calculate the source term and apply it in 1 go (since it's stencil-1)
            FourVectors Dtmp;
            Real dU[NPRIM] = {0};
            GRMHD::calc_4vecs(G, P, B_P, k, j, i, Loci::center, Dtmp);
            GRMHD::get_source(G, P, Dtmp, eos, k, j, i, dU);

            if (wind_term) {
                GRMHD::add_wind(G, eos, k, j, i, current_wind_n, wind_pow, wind_Tp, dU);
            }

            for (int p=0; p < nvar; ++p) {
                dUdt(p, k, j, i) = (U.flux(X1DIR, p, k, j, i) - U.flux(X1DIR, p, k, j, i+1)) / G.dx1v(i);
                if (ndim > 1) dUdt(p, k, j, i) += (U.flux(X2DIR, p, k, j, i) - U.flux(X2DIR, p, k, j+1, i)) / G.dx2v(j);
                if (ndim > 2) dUdt(p, k, j, i) += (U.flux(X3DIR, p, k, j, i) - U.flux(X3DIR, p, k+1, j, i)) / G.dx3v(k);
            }
            PLOOP dUdt(cons_start+p, k, j, i) += dU[p];
        }
    );
    FLAG("Applied");

    return TaskStatus::complete;
}