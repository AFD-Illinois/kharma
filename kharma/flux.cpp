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

#include "flux.hpp"

#include "grmhd.hpp"

using namespace parthenon;

// GetFlux is in the header, as it is templated on reconstruction scheme and flux direction
// That's also why we don't have any extra includes in here

TaskStatus Flux::PtoU(MeshBlockData<Real> *rc, IndexDomain domain)
{
    Flag(rc, "Getting conserved variables");
    // Pointers
    auto pmb = rc->GetBlockPointer();
    // Options
    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam   = pars.Get<Real>("gamma");
    auto pkgs        = pmb->packages.AllPackages();

    const bool flux_ct       = pkgs.count("B_FluxCT");
    const bool b_cd          = pkgs.count("B_CD");
    const bool use_electrons = pkgs.count("Electrons");
    const bool use_emhd      = pkgs.count("EMHD");
    
    MetadataFlag isPrimitive = pars.Get<MetadataFlag>("PrimitiveFlag");

    EMHD::EMHD_parameters emhd_params_tmp;
    if (use_emhd) {
        const auto& emhd_pars = pmb->packages.Get("EMHD")->AllParams();
        emhd_params_tmp = emhd_pars.Get<EMHD::EMHD_parameters>("emhd_params");
    }
    const EMHD::EMHD_parameters& emhd_params = emhd_params_tmp;

    // Pack variables
    PackIndexMap prims_map, cons_map;
    const auto& P_all = rc->PackVariables({isPrimitive}, prims_map);
    const auto& U_all = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    const int nvar = U_all.GetDim(4);

    const IndexRange ib = rc->GetBoundsI(domain);
    const IndexRange jb = rc->GetBoundsJ(domain);
    const IndexRange kb = rc->GetBoundsK(domain);
    const int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

    const auto& G = pmb->coords;

    // This is basically what all kernels look like if I want to stick to
    // single, simple device side functions called over slices
    // See fluxes.hpp or implicit.cpp for explanations of what everything here does
    const int scratch_level = 1;
    const size_t var_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, n1);
    const size_t total_scratch_bytes = (2) * var_size_in_bytes;

    parthenon::par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "PtoU", pmb->exec_space,
        total_scratch_bytes, scratch_level, kb.s, kb.e, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
            ScratchPad2D<Real> P_s(member.team_scratch(scratch_level), nvar, n1);
            ScratchPad2D<Real> U_s(member.team_scratch(scratch_level), nvar, n1);

            PLOOP parthenon::par_for_inner(member, ib.s, ib.e,
                [&](const int& i) {
                    P_s(ip, i) = P_all(ip, k, j, i);
                    U_s(ip, i) = U_all(ip, k, j, i);
                }
            );

            parthenon::par_for_inner(member, ib.s, ib.e,
                [&](const int& i) {
                    auto P = Kokkos::subview(P_s, Kokkos::ALL(), i);
                    auto U = Kokkos::subview(U_s, Kokkos::ALL(), i);
                    Flux::p_to_u(G, P, m_p, emhd_params, gam, j, i, U, m_u);
                }
            );

            PLOOP parthenon::par_for_inner(member, ib.s, ib.e,
                [&](const int& i) {
                    P_all(ip, k, j, i) = P_s(ip, i);
                    U_all(ip, k, j, i) = U_s(ip, i);
                }
            );
        }
    );

    Flag(rc, "Got conserved variables");
    return TaskStatus::complete;
}

TaskStatus Flux::AddSource(MeshData<Real> *md, MeshData<Real> *mdudt)
{
    Flag(mdudt, "Adding source terms to uu, uvec");
    // Pointers
    auto pmesh = md->GetMeshPointer();
    auto& mbd  = md->GetBlockData(0);
    auto pmb0  = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");
    const auto use_emhd = pmb0->packages.AllPackages().count("EMHD");

    // Pack variables
    const MetadataFlag isPrimitive = pmb0->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    PackIndexMap prims_map, cons_map;
    auto P    = md->PackVariables(std::vector<MetadataFlag>{isPrimitive}, prims_map);
    auto dUdt = mdudt->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_p(prims_map, false), m_u(cons_map, true);

    // EMHD params
    EMHD::EMHD_parameters emhd_params_tmp;
    if (use_emhd) {
        const auto& emhd_pars = pmb0->packages.Get("EMHD")->AllParams();
        emhd_params_tmp = emhd_pars.Get<EMHD::EMHD_parameters>("emhd_params");
    }
    const EMHD::EMHD_parameters& emhd_params = emhd_params_tmp;
    
    // Get sizes
    IndexDomain domain = IndexDomain::interior;
    auto ib = md->GetBoundsI(domain);
    auto jb = md->GetBoundsJ(domain);
    auto kb = md->GetBoundsK(domain);
    auto block = IndexRange{0, P.GetDim(5)-1};

    pmb0->par_for("tmunu_source", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D {
            const auto& G = dUdt.GetCoords(b);
            FourVectors D;
            GRMHD::calc_4vecs(G, P(b), m_p, k, j, i, Loci::center, D);
            // Call Flux::calc_tensor which will in turn call the right calc_tensor based on the number of primitives
            Real T[GR_DIM]      = {0};
            Real new_du[GR_DIM] = {0};
            DLOOP2 {
                Flux::calc_tensor(G, P(b), m_p, D, emhd_params, gam, k, j, i, mu, T);
                Real Tmunu  = T[nu];

                // Contract mhd stress tensor with connection, and multiply by metric determinant
                for (int lam = 0; lam < GR_DIM; ++lam) {
                    new_du[lam] += Tmunu * G.gdet_conn(j, i, nu, lam, mu);
                }
            }

            dUdt(b, m_u.UU, k, j, i)           += new_du[0];
            VLOOP dUdt(b, m_u.U1 + v, k, j, i) += new_du[1 + v];
        }
    );

    Flag(mdudt, "Added");
    return TaskStatus::complete;
}