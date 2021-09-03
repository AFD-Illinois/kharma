/* 
 *  File: electrons.cpp
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
#include "electrons.hpp"

#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

// Do I really want to reintroduce this?
#define SMALL 1.e-20

namespace Electrons
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    auto pkg = std::make_shared<StateDescriptor>("Electrons");
    Params &params = pkg->AllParams();

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // Floors & fluid gamma
    Real gamma_e = pin->GetOrAddReal("electrons", "gamma_e", 4./3);
    params.Add("gamma_e", gamma_e);
    Real gamma_p = pin->GetOrAddReal("electrons", "gamma_p", 5./3);
    params.Add("gamma_p", gamma_p);
    Real fel_0 = pin->GetOrAddReal("electrons", "fel_0", 0.01);
    params.Add("fel_0", fel_0);

    Real tp_over_te_min = pin->GetOrAddReal("electrons", "tp_over_te_min", 0.001);
    params.Add("tp_over_te_min", tp_over_te_min);
    Real tp_over_te_max = pin->GetOrAddReal("electrons", "tp_over_te_max", 1000.0);
    params.Add("tp_over_te_max", tp_over_te_max);

    bool suppress_highb_heat = pin->GetOrAddReal("electrons", "suppress_highb_heat", true);
    params.Add("suppress_highb_heat", suppress_highb_heat);

    // Model options
    bool do_howes = pin->GetOrAddBoolean("electrons", "howes", false);
    params.Add("do_howes", do_howes);
    bool do_kawazura = pin->GetOrAddBoolean("electrons", "kawazura", false);
    params.Add("do_kawazura", do_kawazura);
    bool do_werner = pin->GetOrAddBoolean("electrons", "werner", false);
    params.Add("do_werner", do_werner);
    bool do_rowan = pin->GetOrAddBoolean("electrons", "rowan", false);
    params.Add("do_rowan", do_rowan);
    bool do_sharma = pin->GetOrAddBoolean("electrons", "sharma", false);
    params.Add("do_sharma", do_sharma);

    MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    MetadataFlag isElectrons = Metadata::AllocateNewFlag("Electrons");
    params.Add("ElectronsFlag", isElectrons);

    // General options for primitive and conserved variables in KHARMA
    Metadata m_con  = Metadata({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                 Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes, isElectrons});
    Metadata m_prim = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
                  Metadata::Restart, isPrimitive, isElectrons});
    
    // Total entropy, used to track changes
    int nKs = 1;
    pkg->AddField("cons.Ktot", m_con);
    pkg->AddField("prims.Ktot", m_prim);

    // Individual models
    if (do_howes) {
        nKs += 1;
        pkg->AddField("cons.Kel_Howes", m_con);
        pkg->AddField("prims.Kel_Howes", m_prim);
    }
    if (do_kawazura) {
        nKs += 1;
        pkg->AddField("cons.Kel_Kawazura", m_con);
        pkg->AddField("prims.Kel_Kawazura", m_prim);
    }
    if (do_werner) {
        nKs += 1;
        pkg->AddField("cons.Kel_Werner", m_con);
        pkg->AddField("prims.Kel_Werner", m_prim);
    }
    if (do_rowan) {
        nKs += 1;
        pkg->AddField("cons.Kel_Rowan", m_con);
        pkg->AddField("prims.Kel_Rowan", m_prim);
    }
    if (do_sharma) {
        nKs += 1;
        pkg->AddField("cons.Kel_Sharma", m_con);
        pkg->AddField("prims.Kel_Sharma", m_prim);
    }
    // TODO if nKs == 1 then rename Kel_Whatever -> Kel?

    pkg->FillDerivedBlock = Electrons::FillDerived;
    pkg->PostFillDerivedBlock = Electrons::PostFillDerived;
    return pkg;
}

TaskStatus InitElectrons(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();

    MetadataFlag isElectrons = pmb->packages.Get("Electrons")->Param<MetadataFlag>("ElectronsFlag");
    MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    // Need to distinguish KTOT from the other variables, so we record which it is
    PackIndexMap prims_map;
    auto& e_P = rc->PackVariables({isElectrons, isPrimitive}, prims_map);
    const int ktot_index = prims_map["prims.Ktot"].first;
    // Just need these two from the rest of Prims
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
    const Real fel0 = pmb->packages.Get("Electrons")->Param<Real>("fel_0");

    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("UtoP_electrons", 0, e_P.GetDim(4)-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_VARS {
            if (p == ktot_index) {
                // Initialize total entropy by definition,
                e_P(p, k, j, i) = (gam - 1.) * u(k, j, i) * pow(rho(k, j, i), -gam);
            } else {
                // and e- entropy by given constant initial fraction
                e_P(p, k, j, i) = (game - 1.) * fel0 * u(k, j, i) * pow(rho(k, j, i), -game);
            }
        }
    );

    // iharm3d syncs bounds here
    return TaskStatus::complete;
}

void UtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    FLAG("UtoP electrons");
    auto pmb = rc->GetBlockPointer();

    MetadataFlag isElectrons = pmb->packages.Get("Electrons")->Param<MetadataFlag>("ElectronsFlag");
    MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    // No need for a "map" here, we just want everything that fits these
    auto& e_P = rc->PackVariables({isElectrons, isPrimitive});
    auto& e_U = rc->PackVariables({isElectrons, Metadata::Conserved});
    // And then the local density
    GridScalar rho_U = rc->Get("cons.rho").data;

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int is = bounds.is(domain), ie = bounds.ie(domain);
    int js = bounds.js(domain), je = bounds.je(domain);
    int ks = bounds.ks(domain), ke = bounds.ke(domain);
    pmb->par_for("UtoP_electrons", 0, e_P.GetDim(4)-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_VARS {
            e_P(p, k, j, i) = e_U(p, k, j, i) / rho_U(k, j, i);
        }
    );

}

void PostUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    MetadataFlag isElectrons = pmb->packages.Get("Electrons")->Param<MetadataFlag>("ElectronsFlag");
    MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    // Need to distinguish KTOT from the other variables, so we record which it is
    PackIndexMap prims_map;
    auto& e_P = rc->PackVariables({isElectrons, isPrimitive}, prims_map);
    const int ktot_index = prims_map["prims.Ktot"].first;
    // Just need these two from the rest of Prims
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real gamp = pmb->packages.Get("Electrons")->Param<Real>("gamma_p");
    const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
    const Real tptemin = pmb->packages.Get("Electrons")->Param<Real>("tp_over_te_min");
    const Real tptemax = pmb->packages.Get("Electrons")->Param<Real>("tp_over_te_max");

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int is = bounds.is(domain), ie = bounds.ie(domain);
    int js = bounds.js(domain), je = bounds.je(domain);
    int ks = bounds.ks(domain), ke = bounds.ke(domain);
    pmb->par_for("UtoP_electrons", 0, e_P.GetDim(4)-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_VARS {
            if (p != ktot_index) {
                // Note tp_te_min -> kel_max & vice versa
                const Real kel_max = e_P(ktot_index, k, j, i) * pow(rho(k, j, i), gam - game) /
                                        (tptemin * (gam - 1.) / (gamp-1.) + (gam-1.) / (game-1.));
                const Real kel_min = e_P(ktot_index, k, j, i) * pow(rho(k, j, i), gam - game) /
                                        (tptemax * (gam - 1.) / (gamp-1.) + (gam-1.) / (game-1.));

                // Replace NANs with cold electrons
                if (isnan(e_P(p, k, j, i))) {
                   e_P(p, k, j, i) = kel_min;
                }

                // Enforce maximum Tp/Te
                if (e_P(p, k, j, i) < kel_min) {
                    e_P(p, k, j, i) = kel_min;
                }

                // Enforce minimum Tp/Te
                if (e_P(p, k, j, i) > kel_max) {
                    e_P(p, k, j, i) = kel_max;
                }
            }
        }
    );
}

TaskStatus ApplyHeatingModels(MeshBlockData<Real> *rc_old, MeshBlockData<Real> *rc)
{
    FLAG("Applying electron heating");
    auto pmb = rc->GetBlockPointer();

    MetadataFlag isElectrons = pmb->packages.Get("Electrons")->Param<MetadataFlag>("ElectronsFlag");
    MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    // Need to distinguish different electron models
    // So far maps are consistent, so we re-use
    // If Parthenon starts making packing *random*, I'm in trouble
    PackIndexMap prims_map, cons_map;
    auto& P = rc_old->PackVariables({isPrimitive}, prims_map);
    auto& P_new = rc->PackVariables({isPrimitive}, prims_map);
    auto& U_new = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_p(prims_map, false), m_u(cons_map, true);

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real gamp = pmb->packages.Get("Electrons")->Param<Real>("gamma_p");
    const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
    const bool suppress_highb_heat = pmb->packages.Get("Electrons")->Param<bool>("suppress_highb_heat");

    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("heat_electrons", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // GET HEATING FRACTION
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
            Real bsq = dot(Dtmp.bcon, Dtmp.bcov);
            // Suppress all heating at high Bsq. TODO can we skip the rest?
            //if(suppress_highb_heat && (bsq / P(m_p.RHO, k, j, i) > 1.)) return;

            // Calculate total dissipation, update KTOT to reflect real entropy
            const Real kNew = (gam-1.) * P_new(m_p.UU, k, j, i) / pow(P_new(m_p.RHO, k, j, i) ,gam);
            // Dissipation is the real entropy kNew minus the advected entropy from the previous step P_new(KTOT)
            const Real diss = (game-1.) / (gam-1.) * pow(P(m_p.RHO, k, j, i), gam - game) * (kNew - P_new(m_p.KTOT, k, j, i));
            // Reset the entropy to measure next step's dissipation
            P_new(m_p.KTOT, k, j, i) = kNew;

            // Common values we'll need for several models
            // TODO curious to know whether we hit these low temperatures
            // Note that the Tp & Te guards are just to keep the ratio
            // Tp/Te from going -> NaN (TODO can we get away with only Te?)
            Real Tpr;
            if (m_p.K_KAWAZURA >= 0 || m_p.K_SHARMA >= 0) {
                Tpr = (gamp - 1.) * P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i);
                if(Tpr <= SMALL) Tpr = SMALL;
            }

            // TODO Howes
            if (m_p.K_KAWAZURA >= 0) {
                // Equation (2) in http://www.pnas.org/lookup/doi/10.1073/pnas.1812491116
                Real uel = 1./(game - 1.) * P(m_p.K_KAWAZURA, k, j, i) * pow(P(m_p.RHO, k, j, i), game);
                Real Tel = (game - 1.) * uel / P(m_p.RHO, k, j, i);
                if(Tel <= SMALL) Tel = SMALL;

                Real Trat = fabs(Tpr/Tel);
                Real pres = P(m_p.RHO, k, j, i) * Tpr; // Proton pressure
                Real beta = pres / bsq * 2;
                if(beta > 1.e20) beta = 1.e20; // If somebody enables electrons in a GRHD sim
                
                Real QiQe = 35. / (1. + pow(beta/15., -1.4) * exp(-0.1 / Trat));
                Real fel = 1./(1. + QiQe);
                // Measure dissipation as (total Entropy) - (expected advected entropy) at the same time ("new")
                P_new(m_p.K_KAWAZURA, k, j, i) += fel * diss;
            }
            if (m_p.K_WERNER >= 0) {
                // Equation (3) in http://academic.oup.com/mnras/article/473/4/4840/4265350
                Real sigma = bsq / P(m_p.RHO, k, j, i);
                Real fel = 0.25 * (1 + pow((sigma/5.) / (2 + (sigma/5.)), .5));
                P_new(m_p.K_WERNER, k, j, i) += fel * diss;
            }
            if (m_p.K_ROWAN >= 0) {
                // Equation (34) in https://iopscience.iop.org/article/10.3847/1538-4357/aa9380
                Real pres = (gamp - 1.) * P(m_p.UU, k, j, i); // Proton pressure
                Real pg = (gam - 1) * P(m_p.UU, k, j, i);
                Real beta = pres / bsq * 2;
                Real sigma = bsq / (P(m_p.RHO, k, j, i) + P(m_p.UU, k, j, i) + pg);
                Real betamax = 0.25 / sigma;
                Real fel = 0.5 * exp(-pow(1 - beta/betamax, 3.3) / (1 + 1.2*pow(sigma, 0.7)));
                P_new(m_p.K_ROWAN, k, j, i) += fel * diss;
            }
            if (m_p.K_SHARMA >= 0) {
                // Equation for \delta on  pg. 719 (Section 4) in https://iopscience.iop.org/article/10.1086/520800
                Real uel = 1./(game - 1.) * P(m_p.K_SHARMA, k, j, i) * pow(P(m_p.RHO, k, j, i), game);
                Real Tel = (game - 1.) * uel / P(m_p.RHO, k, j, i);
                if(Tel <= SMALL) Tel = SMALL;

                Real Trat_inv = fabs(Tel/Tpr); //Inverse of the temperature ratio in KAWAZURA
                Real QeQi = 0.33 * pow(Trat_inv, 0.5);
                Real fel = 1./(1.+1./QeQi);
                P_new(m_p.K_SHARMA, k, j, i) += fel * diss;
            }
            // Finally, make sure we update the conserved variables to keep them in sync
            Electrons::p_to_u(G, P_new, m_p, k, j, i, U_new, m_u);
        }
    );

    FLAG("Applied");
    return TaskStatus::complete;
}

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc)
{
    FLAG("Printing electron diagnostics");

    // Output any diagnostics after a step completes

    FLAG("Printed")
    return TaskStatus::complete;
}

void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // Anything specially written to files goes here.
    // Normal primitives are written automatically as long as they are specified at runtime
}

} // namespace B_FluxCT
