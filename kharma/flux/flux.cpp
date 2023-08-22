/* 
 *  File: flux.cpp
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
// Most includes are in the header TODO fix?

#include "grmhd.hpp"
#include "kharma.hpp"

using namespace parthenon;

// GetFlux is in the header file get_flux.hpp, as it is templated on reconstruction scheme and flux direction

std::shared_ptr<KHARMAPackage> Flux::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    Flag("Initializing Flux");
    auto pkg = std::make_shared<KHARMAPackage>("Flux");
    Params &params = pkg->AllParams();

    // We can't just use GetVariables or something since there's no mesh yet.
    // That's what this function is for.
    int nvar = KHARMA::PackDimension(packages.get(), Metadata::WithFluxes);
    std::cout << "Allocating fluxes with nvar: " << nvar << std::endl;
    std::vector<int> s_flux({nvar});
    // TODO optionally move all these to faces? Not important yet, no output, more memory
    std::vector<MetadataFlag> flags_flux = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy};
    Metadata m = Metadata(flags_flux, s_flux);
    pkg->AddField("Flux.Pr", m);
    pkg->AddField("Flux.Pl", m);
    pkg->AddField("Flux.Ur", m);
    pkg->AddField("Flux.Ul", m);
    pkg->AddField("Flux.Fr", m);
    pkg->AddField("Flux.Fl", m);

    // TODO could formally move this to face
    std::vector<int> s_vector({NVEC});
    std::vector<MetadataFlag> flags_speed = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy};
    m = Metadata(flags_speed, s_vector);
    pkg->AddField("Flux.cmax", m);
    pkg->AddField("Flux.cmin", m);

    // Preserve all velocities at faces, for upwinded constrained transport
    if (packages->AllPackages().count("B_CT")) {
        std::vector<MetadataFlag> flags_vel = {Metadata::Real, Metadata::Face, Metadata::Derived, Metadata::OneCopy};
        m = Metadata(flags_vel, s_vector);
        pkg->AddField("Flux.vr", m);
        pkg->AddField("Flux.vl", m);
    }

    Flag("Initialized");
    return pkg;
}

TaskStatus Flux::BlockPtoUMHD(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    // Pointers
    auto pmb = rc->GetBlockPointer();
    // Options
    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Pack variables
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const auto& U = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);

    const auto& G = pmb->coords;

    pmb->par_for("p_to_u_mhd", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Flux::p_to_u_mhd(G, P, m_p, emhd_params, gam, k, j, i, U, m_u);
        }
    );

    return TaskStatus::complete;
}

TaskStatus Flux::BlockPtoU(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    // Pointers
    auto pmb = rc->GetBlockPointer();
    // Options
    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Pack variables
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const auto& U = rc->PackVariables({Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);
    const int nvar = U.GetDim(4);

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);

    const auto& G = pmb->coords;

    pmb->par_for("p_to_u", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Flux::p_to_u(G, P, m_p, emhd_params, gam, k, j, i, U, m_u);
        }
    );

    return TaskStatus::complete;
}

TaskStatus Flux::MeshPtoU(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    for (int i=0; i < md->NumBlocks(); ++i)
        Flux::BlockPtoU(md->GetBlockData(i).get(), domain, coarse);
    return TaskStatus::complete;
}

TaskStatus Flux::BlockPtoU_Send(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    // Pointers
    auto pmb = rc->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;
    // Options
    const auto& pars = pmb->packages.Get("GRMHD")->AllParams();
    const Real gam = pars.Get<Real>("gamma");

    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb->packages);

    // Pack variables
    PackIndexMap prims_map, cons_map;
    const auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const auto& U = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    IndexRange ib = bounds.GetBoundsI(domain);
    IndexRange jb = bounds.GetBoundsJ(domain);
    IndexRange kb = bounds.GetBoundsK(domain);

    // Modify the bounds to reflect zones we're sending, rather than actual ghosts
    int ng = Globals::nghost;
    if (domain == IndexDomain::inner_x1) {
        ib.s += ng;
        ib.e += ng;
    } else if (domain == IndexDomain::outer_x1) {
        ib.s -= ng;
        ib.e -= ng;
    } else if (domain == IndexDomain::inner_x2) {
        if (ndim < 2) return TaskStatus::complete;
        jb.s += ng;
        jb.e += ng;
    } else if (domain == IndexDomain::outer_x2) {
        if (ndim < 2) return TaskStatus::complete;
        jb.s -= ng;
        jb.e -= ng;
    } else if (domain == IndexDomain::inner_x3) {
        if (ndim < 3) return TaskStatus::complete;
        kb.s += ng;
        kb.e += ng;
    } else if (domain == IndexDomain::outer_x3) {
        if (ndim < 3) return TaskStatus::complete;
        kb.s -= ng;
        kb.e -= ng;
    } // TODO(BSP) error?

    const auto& G = pmb->coords;

    pmb->par_for("p_to_u_send", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Flux::p_to_u(G, P, m_p, emhd_params, gam, k, j, i, U, m_u);
        }
    );

    return TaskStatus::complete;
}

void Flux::AddGeoSource(MeshData<Real> *md, MeshData<Real> *mdudt)
{
    // Pointers
    auto pmesh = md->GetMeshPointer();
    auto pmb0  = md->GetBlockData(0)->GetBlockPointer();
    auto pkgs = pmb0->packages;
    // Options
    const auto& pars = pkgs.Get("GRMHD")->AllParams();
    const Real gam   = pars.Get<Real>("gamma");

    // All connection coefficients are zero in Cartesian Minkowski space
    // TODO do we know this fully in init?
    if (pmb0->coords.coords.is_cart_minkowski()) return;

    // Pack variables
    PackIndexMap prims_map, cons_map;
    auto P    = md->PackVariables(std::vector<MetadataFlag>{Metadata::GetUserFlag("Primitive")}, prims_map);
    auto dUdt = mdudt->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    const VarMap m_p(prims_map, false), m_u(cons_map, true);

    // EMHD params
    const EMHD::EMHD_parameters& emhd_params = EMHD::GetEMHDParameters(pmb0->packages);
    
    // Get sizes
    IndexDomain domain = IndexDomain::interior;
    auto ib = md->GetBoundsI(domain);
    auto jb = md->GetBoundsJ(domain);
    auto kb = md->GetBoundsK(domain);
    auto block = IndexRange{0, P.GetDim(5)-1};

    pmb0->par_for("tmunu_source", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            const auto& G = dUdt.GetCoords(b);
            FourVectors D;
            GRMHD::calc_4vecs(G, P(b), m_p, k, j, i, Loci::center, D);
            // Call Flux::calc_tensor which will in turn call the right calc_tensor based on the number of primitives
            Real Tmu[GR_DIM]    = {0};
            Real new_du[GR_DIM] = {0};
            for (int mu = 0; mu < GR_DIM; ++mu) {
                Flux::calc_tensor(P(b), m_p, D, emhd_params, gam, k, j, i, mu, Tmu);
                for (int nu = 0; nu < GR_DIM; ++nu) {
                    // Contract mhd stress tensor with connection, and multiply by metric determinant
                    for (int lam = 0; lam < GR_DIM; ++lam) {
                        new_du[lam] += Tmu[nu] * G.gdet_conn(j, i, nu, lam, mu);
                    }
                }
            }

            dUdt(b, m_u.UU, k, j, i)           += new_du[0];
            VLOOP dUdt(b, m_u.U1 + v, k, j, i) += new_du[1 + v];
        }
    );
}

TaskStatus Flux::CheckCtop(MeshData<Real> *md)
{
    Reductions::DomainReduction<Reductions::Var::nan_ctop, int>(md, UserHistoryOperation::sum, 0);
    Reductions::DomainReduction<Reductions::Var::zero_ctop, int>(md, UserHistoryOperation::sum, 1);
    return TaskStatus::complete;
}

TaskStatus Flux::PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int extra_checks = pars.Get<int>("extra_checks");

    // Check for a soundspeed (ctop) of 0 or NaN
    // This functions as a "last resort" check to stop a
    // simulation on obviously bad data
    if (extra_checks >= 1) {
        int nnan = Reductions::Check<int>(md, 0);
        int nzero = Reductions::Check<int>(md, 1);

        if (MPIRank0() && (nzero > 0 || nnan > 0)) {
            // TODO string formatting in C++ that doesn't suck
            fprintf(stderr, "Max signal speed ctop of 0 or NaN (%d zero, %d NaN)", nzero, nnan);
            throw std::runtime_error("Bad ctop!");
        }

    }

    return TaskStatus::complete;
}
