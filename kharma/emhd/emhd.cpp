/* 
 *  File: emhd.cpp
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
#include "emhd.hpp"

#include "decs.hpp"
#include "emhd_sources.hpp"
#include "emhd_utils.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

namespace EMHD
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    auto pkg = std::make_shared<StateDescriptor>("EMHD");
    Params &params = pkg->AllParams();

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // EMHD Problem/Closure parameters
    // GRIM uses a callback to a problem-specific implementation which sets these
    // We share implementations in one function, controlled by these parameters
    // These are always necessary for performing EGRMHD.

    bool higher_order_terms  = pin->GetOrAddBoolean("emhd", "higher_order_terms", false);
    std::string closure_type = pin->GetOrAddString("emhd", "closure_type", "torus");

    // Should the EMHD sector feedback onto the ideal MHD variables? The default is 'yes'.
    // So far it's just the viscous Bondi problem that doesn't require feedback
    bool feedback = pin->GetOrAddBoolean("emhd", "feedback", true);

    Real tau              = pin->GetOrAddReal("emhd", "tau", 1.0);
    Real conduction_alpha = pin->GetOrAddReal("emhd", "conduction_alpha", 1.0);
    Real viscosity_alpha  = pin->GetOrAddReal("emhd", "viscosity_alpha", 1.0);
    
    Real kappa = pin->GetOrAddReal("emhd", "kappa", 1.0);
    Real eta   = pin->GetOrAddReal("emhd", "eta", 1.0);

    EMHD_parameters emhd_params;
    emhd_params.higher_order_terms = higher_order_terms;
    emhd_params.feedback           = feedback;
    if (closure_type == "constant") { 
        emhd_params.type = ClosureType::constant;
    } else if (closure_type == "sound_speed" || closure_type == "soundspeed") {
        emhd_params.type = ClosureType::soundspeed;
    } else if (closure_type == "kappa_eta") {
        emhd_params.type = ClosureType::kappa_eta;
    } else if (closure_type == "torus") {
        emhd_params.type = ClosureType::torus;
    } else {
        throw std::invalid_argument("Invalid Closure type: "+closure_type+". Use constant, sound_speed, or torus");
    }
    emhd_params.tau              = tau;
    emhd_params.conduction_alpha = conduction_alpha;
    emhd_params.viscosity_alpha  = viscosity_alpha;
    emhd_params.kappa            = kappa;
    emhd_params.eta              = eta;
    params.Add("emhd_params", emhd_params);

    // Slope reconstruction on faces. Always linear: default to MC unless we're using VL everywhere
    if (packages.Get("GRMHD")->Param<ReconstructionType>("recon") == ReconstructionType::linear_vl) {
        params.Add("slope_recon", ReconstructionType::linear_mc);
    } else {
        params.Add("slope_recon", ReconstructionType::linear_mc);
    }

    // Floors specific to EMHD calculations? Currently only need to enforce bsq>0 in one denominator

    MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    MetadataFlag isEMHD      = Metadata::AllocateNewFlag("EMHDFlag");
    params.Add("EMHDFlag", isEMHD);

    // General options for primitive and conserved scalar variables in ImEx driver
    // EMHD is supported only with imex driver and implicit evolution
    MetadataFlag isImplicit = packages.Get("Implicit")->Param<MetadataFlag>("ImplicitFlag");
    Metadata m_con  = Metadata({Metadata::Real, Metadata::Cell, Metadata::Independent, isImplicit,
                                Metadata::Conserved, Metadata::WithFluxes, isEMHD});
    Metadata m_prim = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, isImplicit,
                                Metadata::FillGhost, Metadata::Restart, isPrimitive, isEMHD});

    // Heat conduction
    pkg->AddField("cons.q", m_con);
    pkg->AddField("prims.q", m_prim);
    // Pressure anisotropy
    pkg->AddField("cons.dP", m_con);
    pkg->AddField("prims.dP", m_prim);

    // If we want to register an EMHD-specific UtoP for some reason?
    // Likely we'll only use the post-step summary hook
    //pkg->FillDerivedBlock = EMHD::FillDerived;
    //pkg->PostFillDerivedBlock = EMHD::PostFillDerived;
    return pkg;
}

TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt)
{

    Flag(mdudt, "Adding EMHD Explicit Sources");
    // Pointers
    auto pmesh = mdudt->GetMeshPointer();
    auto pmb0  = mdudt->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& gpars = pmb0->packages.Get("GRMHD")->AllParams();
    const Real gam    = gpars.Get<Real>("gamma");
    const int ndim    = pmesh->ndim;
    const MetadataFlag isPrimitive = gpars.Get<MetadataFlag>("PrimitiveFlag");

    const auto& pars                   = pmb0->packages.Get("EMHD")->AllParams();
    const EMHD_parameters& emhd_params = pars.Get<EMHD_parameters>("emhd_params");

    // Pack variables
    PackIndexMap prims_map, cons_map;
    auto P    = md->PackVariables(std::vector<MetadataFlag>{isPrimitive}, prims_map);
    auto U    = md->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved}, cons_map);
    auto dUdt = mdudt->PackVariables(std::vector<MetadataFlag>{Metadata::Conserved});
    const VarMap m_p(prims_map, false), m_u(cons_map, true);

    // Get sizes, declare temporary ucov, Theta for gradients
    const int n1 = pmb0->cellbounds.ncellsi(IndexDomain::entire);
    const int n2 = pmb0->cellbounds.ncellsj(IndexDomain::entire);
    const int n3 = pmb0->cellbounds.ncellsk(IndexDomain::entire);
    const int nb = dUdt.GetDim(5);
    GridVector ucov_s("ucov", nb, GR_DIM, n3, n2, n1);
    GridScalar theta_s("Theta", nb, n3, n2, n1);

    // Get ranges
    const IndexRange ib = mdudt->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = mdudt->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = mdudt->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, nb - 1};
    // 1-zone halo in nontrivial dimensions
    const IndexRange il = IndexRange{ib.s-1, ib.e+1};
    const IndexRange jl = (ndim > 1) ? IndexRange{jb.s-1, jb.e+1} : jb;
    const IndexRange kl = (ndim > 2) ? IndexRange{kb.s-1, kb.e+1} : kb;

    // Calculate & apply source terms
    pmb0->par_for("emhd_sources_pre", block.s, block.e, kl.s, kl.e, jl.s, jl.e, il.s, il.e,
        KOKKOS_LAMBDA_MESH_3D {
            const auto& G    = dUdt.GetCoords(b);
            const GReal gdet = G.gdet(Loci::center, j, i);
            // ucon
            Real ucon[GR_DIM], ucov[GR_DIM];
            GRMHD::calc_ucon(G, P(b), m_p, k, j, i, Loci::center, ucon);
            G.lower(ucon, ucov, k, j, i, Loci::center);
            DLOOP1 ucov_s(b, mu, k, j, i) = ucov[mu];
            // theta
            theta_s(b, k, j, i) = m::max((gam - 1) * P(b)(m_p.UU, k, j, i) / P(b)(m_p.RHO, k, j, i), SMALL);
        }
    );

    // Calculate & apply source terms
    pmb0->par_for("emhd_sources", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_3D {
            const auto& G = dUdt.GetCoords(b);

            // Get the EGRMHD parameters
            Real tau, chi_e, nu_e;
            EMHD::set_parameters(G, P(b), m_p, emhd_params, gam, k, j, i, tau, chi_e, nu_e, "explicit_sources");

            // and the 4-vectors
            FourVectors D;
            GRMHD::calc_4vecs(G, P(b), m_p, k, j, i, Loci::center, D);
            double bsq = m::max(dot(D.bcon, D.bcov), SMALL);

            // Compute gradient of ucov and Theta
            Real grad_ucov[GR_DIM][GR_DIM], grad_Theta[GR_DIM];
            EMHD::gradient_calc(G, P(b), ucov_s, theta_s, b, k, j, i, (ndim > 2), (ndim > 1), grad_ucov, grad_Theta);

            // Compute div of ucon (all terms but the time-derivative ones are nonzero)
            Real div_ucon    = 0;
            DLOOP2 div_ucon += G.gcon(Loci::center, j, i, mu, nu) * grad_ucov[mu][nu];

            // Compute+add explicit source terms (conduction and viscosity)
            const Real& rho     = P(b)(m_p.RHO, k, j, i);
            const Real& qtilde  = P(b)(m_p.Q, k, j, i);
            const Real& dPtilde = P(b)(m_p.DP, k, j, i);

            Real q0    = 0;
            DLOOP1 q0 -= rho * chi_e * (D.bcon[mu] / m::sqrt(bsq)) * grad_Theta[mu];
            DLOOP2 q0 -= rho * chi_e * (D.bcon[mu] / m::sqrt(bsq)) * theta_s(b, k, j, i) * D.ucon[nu] * grad_ucov[nu][mu];

            Real dP0     = -rho * nu_e * div_ucon;
            DLOOP2  dP0 += 3. * rho * nu_e * (D.bcon[mu] * D.bcon[nu] / bsq) * grad_ucov[mu][nu];

            Real q0_tilde  = q0; 
            Real dP0_tilde = dP0;
            if (emhd_params.higher_order_terms) {
                q0_tilde  *= (chi_e != 0) ? sqrt(tau / (chi_e * rho * pow(theta_s(b, k, j, i), 2)) ) : 0.;
                dP0_tilde *= (nu_e  != 0) ? sqrt(tau / (nu_e * rho * theta_s(b, k, j, i)) ) : 0.;
            }

            dUdt(b, m_u.Q, k, j, i)  += G.gdet(Loci::center, j, i) * q0_tilde / tau;
            dUdt(b, m_u.DP, k, j, i) += G.gdet(Loci::center, j, i) * dP0_tilde / tau;

            if (emhd_params.higher_order_terms) {
                dUdt(b, m_u.Q, k, j, i)  += G.gdet(Loci::center, j, i) * (qtilde / 2.) * div_ucon;
                dUdt(b, m_u.DP, k, j, i) += G.gdet(Loci::center, j, i) * (dPtilde / 2.) * div_ucon;
            }
        }
    );

    Flag(mdudt, "Added");
    return TaskStatus::complete;
}

} // namespace EMHD
