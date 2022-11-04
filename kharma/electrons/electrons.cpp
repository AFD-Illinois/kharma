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
#include "flux.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "gaussian.hpp"

#include <parthenon/parthenon.hpp>
#include <utils/string_utils.hpp>

#include <string>

using namespace parthenon;

// Used only in Howes model
#define ME (9.1093826e-28  ) // Electron mass
#define MP (1.67262171e-24 ) // Proton mass

namespace Electrons
{

/**
 * Take a parthenon parameter, split on commas,
 * and parse as doubles
 */
std::vector<Real> parse_list(std::string s)
{
    // Tokenizer stolen from Parthenon's output machinery
    std::string delimiter = ",";
    size_t pos = 0;
    std::string token;
    std::vector<Real> reals;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        reals.push_back(std::stod(string_utils::trim(token)));
        s.erase(0, pos + delimiter.length());
    }
    reals.push_back(std::stod(string_utils::trim(s)));
    return reals;
}

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

    // Evolution parameters
    Real gamma_e = pin->GetOrAddReal("electrons", "gamma_e", 4./3);
    params.Add("gamma_e", gamma_e);
    Real gamma_p = pin->GetOrAddReal("electrons", "gamma_p", 5./3);
    params.Add("gamma_p", gamma_p);
    Real fel_0 = pin->GetOrAddReal("electrons", "fel_0", 0.01);
    params.Add("fel_0", fel_0);
    bool diss_sign = pin->GetOrAddBoolean("electrons", "diss_sign", true);
    params.Add("diss_sign", diss_sign);
    bool kel_lim = pin->GetOrAddBoolean("electrons", "kel_lim", true);
    params.Add("kel_lim", kel_lim);
    // This is used only in constant model
    Real fel_const = pin->GetOrAddReal("electrons", "fel_constant", 0.1);
    params.Add("fel_constant", fel_const);
    // This prevented spurious heating when heat_electrons used pre-floored dissipation
    bool suppress_highb_heat = pin->GetOrAddBoolean("electrons", "suppress_highb_heat", false);
    params.Add("suppress_highb_heat", suppress_highb_heat);

    // Floors
    Real tp_over_te_min = pin->GetOrAddReal("electrons", "tp_over_te_min", 0.001);
    params.Add("tp_over_te_min", tp_over_te_min);
    Real tp_over_te_max = pin->GetOrAddReal("electrons", "tp_over_te_max", 1000.0);
    params.Add("tp_over_te_max", tp_over_te_max);
    Real ktot_max = pin->GetOrAddReal("floors", "ktot_max", 1.e20);
    params.Add("ktot_max", ktot_max);

    // Model options
    bool do_constant = pin->GetOrAddBoolean("electrons", "constant", false);
    params.Add("do_constant", do_constant);
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

    // Parse various mass and density units to set the different cooling rates
    // These could maybe tie in with Parthenon::Units when we add radiation
    std::vector<Real> masses = parse_list(pin->GetOrAddString("units", "MBH", "1.0"));
    if (masses != std::vector<Real>{1.0})
    {
        std::vector<std::vector<Real>> munits;
        for (int i=1; i <= masses.size(); ++i) {
            munits.push_back(parse_list(pin->GetString("units", "M_unit_"+to_string(i))));
        }

        if (MPIRank0() && verbose > 0) {
            cout << "Using unit sets:" << endl;
            for (int i=0; i < masses.size(); ++i) {
                cout << endl << masses[i] << ":";
                for (auto munit : munits[i]) {
                    cout << " " << munit;
                }
            }
            cout << endl;
        }
        // This is a vector of Reals
        params.Add("masses", masses);
        // This is a vector of vectors of Reals
        params.Add("munits", munits);
    }

    MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    MetadataFlag isElectrons = Metadata::AllocateNewFlag("Electrons");
    params.Add("ElectronsFlag", isElectrons);

    // Default implicit iff GRMHD is done implicitly. TODO can we do explicit?
    auto driver_type = pin->GetString("driver", "type");
    bool grmhd_implicit = packages.Get("GRMHD")->Param<bool>("implicit"); // usually false
    bool implicit_e = (driver_type == "imex" && pin->GetOrAddBoolean("electrons", "implicit", grmhd_implicit)); // so this false too
    params.Add("implicit", implicit_e);

    // B fields.  "Primitive" form is field, "conserved" is flux
    // See notes there about changes for the Imex driver
    std::vector<MetadataFlag> flags_prim, flags_cons;
    if (driver_type == "harm") {
        flags_cons = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                                    Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes, isElectrons});
        flags_prim = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Derived,
                                                isPrimitive, isElectrons});
    } else if (driver_type == "imex") {
        // See grmhd.cpp for full notes on flag changes for ImEx driver
        // Note that default for B is *explicit* evolution
        MetadataFlag areWeImplicit = (implicit_e) ? packages.Get("Implicit")->Param<MetadataFlag>("ImplicitFlag")
                                                  : packages.Get("Implicit")->Param<MetadataFlag>("ExplicitFlag"); // so setting entropies as explicit
        flags_cons = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::Conserved,
                                                Metadata::WithFluxes, areWeImplicit, isElectrons});
        flags_prim = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::FillGhost,
                                                Metadata::Restart, isPrimitive, areWeImplicit, isElectrons});
    }

    // Total entropy, used to track changes
    int nKs = 1;
    pkg->AddField("cons.Ktot", flags_cons);
    pkg->AddField("prims.Ktot", flags_prim);

    if ("driven_turbulence" == packages.Get("GRMHD")->Param<std::string>("problem")) {
        std::vector<int> s_vector({2});
        Metadata m_vector = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
        Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("grf_normalized", m_vector);
        pkg->AddField("alfven_speed", m);
    }

    // Individual models
    // TO ADD A MODEL:
    // 1. Define fields here
    // 2. Define names in types.hpp
    // 3. Add clauses in p_to_u in electrons.hpp and prim_to_flux in flux_functions.hpp
    // 4. Add heating model in ApplyElectronHeating, below
    if (do_constant) {
        nKs += 1;
        pkg->AddField("cons.Kel_Constant", flags_cons);
        pkg->AddField("prims.Kel_Constant", flags_prim);
    }
    if (do_howes) {
        nKs += 1;
        pkg->AddField("cons.Kel_Howes", flags_cons);
        pkg->AddField("prims.Kel_Howes", flags_prim);
    }
    if (do_kawazura) {
        nKs += 1;
        pkg->AddField("cons.Kel_Kawazura", flags_cons);
        pkg->AddField("prims.Kel_Kawazura", flags_prim);
    }
    if (do_werner) {
        nKs += 1;
        pkg->AddField("cons.Kel_Werner", flags_cons);
        pkg->AddField("prims.Kel_Werner", flags_prim);
    }
    if (do_rowan) {
        nKs += 1;
        pkg->AddField("cons.Kel_Rowan", flags_cons);
        pkg->AddField("prims.Kel_Rowan", flags_prim);
    }
    if (do_sharma) {
        nKs += 1;
        pkg->AddField("cons.Kel_Sharma", flags_cons);
        pkg->AddField("prims.Kel_Sharma", flags_prim);
    }
    // TODO if nKs == 1 then rename Kel_Whatever -> Kel?
    // TODO record nKs and find a nice way to loop/vector the device-side layout?

    pkg->FillDerivedBlock = Electrons::FillDerivedBlock;
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

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("UtoP_electrons", 0, e_P.GetDim(4)-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_VARS {
            if (p == ktot_index) { // Initialize it even when using Hubble, it will be "erased" immediately after anyway in ApplyElectronHeating
                // Initialize total entropy by definition,
                e_P(p, k, j, i) = (gam - 1.) * u(k, j, i) * m::pow(rho(k, j, i), -gam);
            } else {
                // and e- entropy by given constant initial fraction
                e_P(p, k, j, i) = (game - 1.) * fel0 * u(k, j, i) * m::pow(rho(k, j, i), -game);
            }
        }
    );

    // iharm3d syncs bounds here
    return TaskStatus::complete;
}

void UtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "UtoP electrons");
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

TaskStatus ApplyElectronHeating(MeshBlockData<Real> *rc_old, MeshBlockData<Real> *rc)
{
    Flag(rc, "Applying electron heating");
    auto pmb = rc->GetBlockPointer();

    MetadataFlag isElectrons = pmb->packages.Get("Electrons")->Param<MetadataFlag>("ElectronsFlag");
    MetadataFlag isPrimitive = pmb->packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    // Need to distinguish different electron models
    // So far, Parthenon's maps of the same sets of variables are consistent,
    // so we only bother with one map of the primitives
    // TODO Parthenon can definitely build a pack from a map, though
    PackIndexMap prims_map, cons_map;
    auto& P = rc_old->PackVariables({isPrimitive}, prims_map);
    auto& P_new = rc->PackVariables({isPrimitive}, prims_map);
    auto& U_new = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_p(prims_map, false), m_u(cons_map, true);

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real gamp = pmb->packages.Get("Electrons")->Param<Real>("gamma_p");
    const Real game = pmb->packages.Get("Electrons")->Param<Real>("gamma_e");
    const Real fel_const = pmb->packages.Get("Electrons")->Param<Real>("fel_constant");
    const bool suppress_highb_heat = pmb->packages.Get("Electrons")->Param<bool>("suppress_highb_heat");
    // Floors
    const Real tptemin = pmb->packages.Get("Electrons")->Param<Real>("tp_over_te_min");
    const Real tptemax = pmb->packages.Get("Electrons")->Param<Real>("tp_over_te_max");

    // This function (and any primitive-variable sources) needs to be run over the entire domain,
    // because the boundary zones have already been updated and so the same calculations must be applied
    // in order to keep them consistent.
    // See harm_driver.cpp for the full picture of what gets updated when.
    const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
    pmb->par_for("heat_electrons", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
            Real bsq = dot(Dtmp.bcon, Dtmp.bcov);

            // Calculate the new total entropy in this cell
            const Real kNew = (gam-1.) * P_new(m_p.UU, k, j, i) / m::pow(P_new(m_p.RHO, k, j, i) ,gam);

            // Dissipation is the real entropy k_energy_conserving minus any advected entropy from the previous (sub-)step P_new(KTOT)
            // Due to floors we can end up with diss==0 or even *slightly* <0, so we require it to be positive here
            // Under the flag "suppress_highb_heat", we set all dissipation to zero at sigma > 1.
            const Real diss = (suppress_highb_heat && (bsq / P(m_p.RHO, k, j, i) > 1.)) ? 0.0 :
                                m::max((game-1.) / (gam-1.) * m::pow(P(m_p.RHO, k, j, i), gam - game) * (kNew - P_new(m_p.KTOT, k, j, i)), 0.0);

            // Reset the entropy to measure next (sub-)step's dissipation
            P_new(m_p.KTOT, k, j, i) = k_energy_conserving;

            // We'll be applying floors inline as we heat electrons, so
            // we cache the floors as entropy limits so they'll be cheaper to apply.
            // Note tp_te_min -> kel_max & vice versa
            const Real kel_max = P(m_p.KTOT, k, j, i) * m::pow(P(m_p.RHO, k, j, i), gam - game) /
                                    (tptemin * (gam - 1.) / (gamp-1.) + (gam-1.) / (game-1.));
            const Real kel_min = P(m_p.KTOT, k, j, i) * m::pow(P(m_p.RHO, k, j, i), gam - game) /
                                    (tptemax * (gam - 1.) / (gamp-1.) + (gam-1.) / (game-1.));
            // Note this differs a little from Ressler '15, who ensure u_e/u_g > 0.01 rather than use temperatures

            // The ion temperature is useful for a few models, cache it too.
            // The minimum values on Tpr & Tel here ensure that for un-initialized zones,
            // Tpr/Tel == Tel/Tpr == 1 != NaN.  This condition should not be hit after step 1
            const Real Tpr = m::max((gamp - 1.) * P(m_p.UU, k, j, i) / P(m_p.RHO, k, j, i), SMALL);

            // Heat different electron passives based on different dissipation fraction models
            // Expressions here closely adapted (read: stolen) from implementation in iharm3d
            // courtesy of Cesar Diaz, see https://github.com/AFD-Illinois/iharm3d
            
            // In all of these the electron entropy stored value is the entropy conserving solution 
                                 // and then when updated it becomes the energy conserving solution
            if (m_p.K_CONSTANT >= 0) {
                const Real fel = fel_const;
                // Default is true then enforce kel limits with clamp/clip, else no restrictions on kel
                if (pmb->packages.Get("Electrons")->Param<bool>("kel_lim")) {
                    P_new(m_p.K_CONSTANT, k, j, i) = clip(P_new(m_p.K_CONSTANT, k, j, i) + fel * diss, kel_min, kel_max);
                } else {
                    P_new(m_p.K_CONSTANT, k, j, i) += fel * diss;
                }
            }
            if (m_p.K_HOWES >= 0) {
                const Real Tel = m::max(P(m_p.K_HOWES, k, j, i) * m::pow(P(m_p.RHO, k, j, i), game-1), SMALL);

                const Real Trat = Tpr / Tel;
                const Real pres = P(m_p.RHO, k, j, i) * Tpr; // Proton pressure
                const Real beta = m::min(pres / bsq * 2, 1.e20);// If somebody enables electrons in a GRHD sim

                const Real logTrat = log10(Trat);
                const Real mbeta = 2. - 0.2*logTrat;

                const Real c2 = (Trat <= 1.) ? 1.6/Trat : 1.2/Trat;
                const Real c3 = (Trat <= 1.) ? 18. + 5.*logTrat : 18.;

                const Real beta_pow = m::pow(beta, mbeta);
                const Real qrat = 0.92 * (c2*c2 + beta_pow)/(c3*c3 + beta_pow) * exp(-1./beta) * m::sqrt(MP/ME * Trat);
                const Real fel = 1./(1. + qrat);
                P_new(m_p.K_HOWES, k, j, i) = clip(P_new(m_p.K_HOWES, k, j, i) + fel * diss, kel_min, kel_max);
            }
            if (m_p.K_KAWAZURA >= 0) {
                // Equation (2) in http://www.pnas.org/lookup/doi/10.1073/pnas.1812491116
                const Real Tel = m::max(P(m_p.K_KAWAZURA, k, j, i) * m::pow(P(m_p.RHO, k, j, i), game-1), SMALL);

                const Real Trat = Tpr / Tel;
                const Real pres = P(m_p.RHO, k, j, i) * Tpr; // Proton pressure
                const Real beta = m::min(pres / bsq * 2, 1.e20);// If somebody enables electrons in a GRHD sim

                const Real QiQe = 35. / (1. + m::pow(beta/15., -1.4) * exp(-0.1 / Trat));
                const Real fel = 1./(1. + QiQe);
                P_new(m_p.K_KAWAZURA, k, j, i) = clip(P_new(m_p.K_KAWAZURA, k, j, i) + fel * diss, kel_min, kel_max);
            }
            // TODO KAWAZURA 19/20/21 separately?
            if (m_p.K_WERNER >= 0) {
                // Equation (3) in http://academic.oup.com/mnras/article/473/4/4840/4265350
                const Real sigma = bsq / P(m_p.RHO, k, j, i);
                const Real fel = 0.25 * (1 + m::sqrt((sigma/5.) / (2 + (sigma/5.))));
                P_new(m_p.K_WERNER, k, j, i) = clip(P_new(m_p.K_WERNER, k, j, i) + fel * diss, kel_min, kel_max);
            }
            if (m_p.K_ROWAN >= 0) {
                // Equation (34) in https://iopscience.iop.org/article/10.3847/1538-4357/aa9380
                const Real pres = (gamp - 1.) * P(m_p.UU, k, j, i); // Proton pressure
                const Real pg = (gam - 1) * P(m_p.UU, k, j, i);
                const Real beta = pres / bsq * 2;
                const Real sigma = bsq / (P(m_p.RHO, k, j, i) + P(m_p.UU, k, j, i) + pg);
                const Real betamax = 0.25 / sigma;
                const Real fel = 0.5 * exp(-m::pow(1 - beta/betamax, 3.3) / (1 + 1.2*m::pow(sigma, 0.7)));
                P_new(m_p.K_ROWAN, k, j, i) = clip(P_new(m_p.K_ROWAN, k, j, i) + fel * diss, kel_min, kel_max);
            }
            if (m_p.K_SHARMA >= 0) {
                // Equation for \delta on  pg. 719 (Section 4) in https://iopscience.iop.org/article/10.1086/520800
                const Real Tel = m::max(P(m_p.K_SHARMA, k, j, i) * m::pow(P(m_p.RHO, k, j, i), game-1), SMALL);

                const Real Trat_inv = Tel / Tpr; // Inverse of the temperature ratio in KAWAZURA
                const Real QeQi = 0.33 * m::sqrt(Trat_inv);
                const Real fel = 1./(1.+1./QeQi);
                P_new(m_p.K_SHARMA, k, j, i) = clip(P_new(m_p.K_SHARMA, k, j, i) + fel * diss, kel_min, kel_max);
            }

            // Finally, make sure we update the conserved electron variables to keep them in sync
            Electrons::p_to_u(G, P_new, m_p, k, j, i, U_new, m_u);
        }
    );

    const IndexRange myib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange myjb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange mykb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    // A couple of the electron test problems add source terms
    // TODO move this to dUdt with other source terms?
    const std::string prob = pmb->packages.Get("GRMHD")->Param<std::string>("problem");
    if (prob == "hubble") {
        const Real v0 = pmb->packages.Get("GRMHD")->Param<Real>("v0");
        const Real ug0 = pmb->packages.Get("GRMHD")->Param<Real>("ug0");
        const Real dt = pmb->packages.Get("Globals")->Param<Real>("dt_last");  // Close enough?
        const Real t = pmb->packages.Get("Globals")->Param<Real>("time") + dt;
        Real Q = (ug0 * v0 * (gam - 2) / pow(1 + v0 * t, 3));
        pmb->par_for("hubble_Q_source_term", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA_3D {
                const Real Q = -(ug0 * v0 * (gam - 2) / m::pow(1 + v0 * t, 3));
                P_new(m_p.UU, k, j, i) += Q * dt;
                // TODO all flux
                GRMHD::p_to_u(G, P_new, m_p, gam, k, j, i, U_new, m_u);
            }
        );
    } else if (prob == "rest_conserve" && pmb->packages.Get("GRMHD")->Param<Real>("q") != 0. && generate_grf) {
        const Real dt = pmb->packages.Get("Globals")->Param<Real>("dt_last");  // Close enough?
        const Real Q = pmb->packages.Get("GRMHD")->Param<Real>("q");
         pmb->par_for("rest_conserve_Q_source_term", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA_3D {
                P_new(m_p.UU, k, j, i) += Q * dt;
                GRMHD::p_to_u(G, P_new, m_p, gam, k, j, i, U_new, m_u);
            }
        );
    } else if (prob == "driven_turbulence") { // Gaussian random field:
        const auto& G = pmb->coords;
        GridScalar rho = rc->Get("prims.rho").data;
        GridVector uvec = rc->Get("prims.uvec").data;
        GridVector grf_normalized = rc->Get("grf_normalized").data;
        const Real t = pmb->packages.Get("Globals")->Param<Real>("time");
        Real counter = pmb->packages.Get("GRMHD")->Param<Real>("counter");
        const Real dt_kick=  pmb->packages.Get("GRMHD")->Param<Real>("dt_kick");
        if (generate_grf && counter < t) {  counter+=dt_kick;
            pmb->packages.Get("GRMHD")->UpdateParam<Real>("counter", counter);
            printf("Kick applied at time %.32f\n", t);
            const Real lx1=  pmb->packages.Get("GRMHD")->Param<Real>("lx1");
            const Real lx2=  pmb->packages.Get("GRMHD")->Param<Real>("lx2");
            const Real edot= pmb->packages.Get("GRMHD")->Param<Real>("drive_edot");
            GridScalar alfven_speed = rc->Get("alfven_speed").data;
            
            int Nx1 = pmb->cellbounds.ncellsi(IndexDomain::interior);
            int Nx2 = pmb->cellbounds.ncellsj(IndexDomain::interior);
            Real *dv0 =  (Real*) malloc(sizeof(Real)*Nx1*Nx2);
            Real *dv1 =  (Real*) malloc(sizeof(Real)*Nx1*Nx2);
            create_grf(Nx1, Nx2, lx1, lx2, dv0, dv1);

            Real mean_velocity_num0 = 0;    Kokkos::Sum<Real> mean_velocity_num0_reducer(mean_velocity_num0);
            Real mean_velocity_num1 = 0;    Kokkos::Sum<Real> mean_velocity_num1_reducer(mean_velocity_num1);
            Real tot_mass = 0;              Kokkos::Sum<Real> tot_mass_reducer(tot_mass);
            pmb->par_reduce("forced_mhd_normal_kick_centering_mean_vel0", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA_3D_REDUCE {
                    Real cell_mass = (rho(k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i));
                    local_result += cell_mass * dv0[(i-4)*Nx1+(j-4)];
                }
            , mean_velocity_num0_reducer);
            pmb->par_reduce("forced_mhd_normal_kick_centering_mean_vel1", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA_3D_REDUCE {
                    Real cell_mass = (rho(k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i));
                    local_result += cell_mass * dv1[(i-4)*Nx1+(j-4)];
                }
            , mean_velocity_num1_reducer);
            pmb->par_reduce("forced_mhd_normal_kick_centering_tot_mass", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA_3D_REDUCE {
                    local_result += (rho(k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i));
                }
            , tot_mass_reducer);
            Real mean_velocity0 = mean_velocity_num0/tot_mass;
            Real mean_velocity1 = mean_velocity_num1/tot_mass;
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < Nx1 ; i ++) {
                for (size_t j = 0; j < Nx2 ; j ++) {
                    dv0[i*Nx1+j] -= mean_velocity0;
                    dv1[i*Nx1+j] -= mean_velocity1;
                }
            } 

            Real Bhalf = 0; Real A = 0; Real init_e = 0; 
            Kokkos::Sum<Real> Bhalf_reducer(Bhalf); Kokkos::Sum<Real> A_reducer(A); Kokkos::Sum<Real> init_e_reducer(init_e);
            pmb->par_reduce("forced_mhd_normal_kick_normalization_Bhalf", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA_3D_REDUCE {
                    Real cell_mass = (rho(k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i));
                    local_result += cell_mass * (dv0[(i-4)*Nx1+(j-4)]*uvec(0, k, j, i) + dv1[(i-4)*Nx1+(j-4)]*uvec(1, k, j, i));
                }
            , Bhalf_reducer);
            pmb->par_reduce("forced_mhd_normal_kick_normalization_A", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA_3D_REDUCE {
                    Real cell_mass = (rho(k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i));
                    local_result += cell_mass * (pow(dv0[(i-4)*Nx1+(j-4)], 2) + pow(dv1[(i-4)*Nx1+(j-4)], 2));
                }
            , A_reducer);
            pmb->par_reduce("forced_mhd_normal_kick_init_e", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA_3D_REDUCE {
                    Real cell_mass = (rho(k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i));
                    local_result += 0.5 * cell_mass * (pow(uvec(0, k, j, i), 2) + pow(uvec(1, k, j, i), 2));
                }
            , init_e_reducer);

            Real norm_const = (-Bhalf + pow(pow(Bhalf,2) + A*2*dt_kick*edot, 0.5))/A;  // going from k:(0, 0), j:(4, 515), i:(4, 515) inclusive
            pmb->par_for("forced_mhd_normal_kick_setting", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA_3D {
                    grf_normalized(0, k, j, i) = (dv0[(i-4)*Nx1+(j-4)]*norm_const);
                    grf_normalized(1, k, j, i) = (dv1[(i-4)*Nx1+(j-4)]*norm_const);
                    uvec(0, k, j, i) += grf_normalized(0, k, j, i);
                    uvec(1, k, j, i) += grf_normalized(1, k, j, i);
                    FourVectors Dtmp;
                    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
                    Real bsq = dot(Dtmp.bcon, Dtmp.bcov);
                    alfven_speed(k,j,i) = bsq/rho(k, j, i); //saving alfven speed for analysis purposes
                }
            );

            Real finl_e = 0;    Kokkos::Sum<Real> finl_e_reducer(finl_e);
            pmb->par_reduce("forced_mhd_normal_kick_finl_e", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
                KOKKOS_LAMBDA_3D_REDUCE {
                    Real cell_mass = (rho(k, j, i) * G.dx3v(k) * G.dx2v(j) * G.dx1v(i));
                    local_result += 0.5 * cell_mass * (pow(uvec(0, k, j, i), 2) + pow(uvec(1, k, j, i), 2));
                }
            , finl_e_reducer);
            printf("%.32f\n", A); printf("%.32f\n", Bhalf); printf("%.32f\n", norm_const);
            printf("%.32f\n", (finl_e-init_e)/dt_kick);
            free(dv0); free(dv1);
        } 
        // This could be only the GRMHD vars, for this problem, but speed isn't really an issue
        Flux::PtoU(rc);
    }
    Flag(rc, "Applied");
    return TaskStatus::complete;
}

// Only if prob is rest_conserve or hubble
TaskStatus ApplyHeating(MeshBlockData<Real> * mbase) {
    auto pmb0 = mbase->GetBlockPointer();
    const string prob = pmb0->packages.Get("GRMHD")->Param<string>("problem");
    if (prob != "rest_conserve" && prob != "hubble") return TaskStatus::complete;

    Flag(mbase, "Applying heating");

    PackIndexMap prims_map;
    auto P_mbase = GRMHD::PackHDPrims(mbase, prims_map);
    const VarMap m_p(prims_map, false);

    Real Q = 0;
    const Real dt = pmb0->packages.Get("Globals")->Param<Real>("dt_last");  // Close enough?
    const Real t = pmb0->packages.Get("Globals")->Param<Real>("time") + 0.5*dt;
    if (prob == "rest_conserve") {
        const Real q = pmb0->packages.Get("GRMHD")->Param<Real>("q");
        Q = q*pow(t, 2);
    } else {
        const Real v0 = pmb0->packages.Get("GRMHD")->Param<Real>("v0");
        const Real ug0 = pmb0->packages.Get("GRMHD")->Param<Real>("ug0");
        const Real gam = pmb0->packages.Get("GRMHD")->Param<Real>("gamma");
        Q = (ug0 * v0 * (gam - 2) / pow(1 + v0 * t, 3));
    }
    IndexDomain domain = IndexDomain::interior;
    auto ib = mbase->GetBoundsI(domain);
    auto jb = mbase->GetBoundsJ(domain);
    auto kb = mbase->GetBoundsK(domain);
    auto block = IndexRange{0, P_mbase.GetDim(5)-1};
    
    pmb0->par_for("heating_substep", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            P_mbase(m_p.UU, k, j, i) += Q*dt*0.5;
        }
    );
    Flux::PtoU(mbase);
    Flag(mbase, "Applied heating");
    return TaskStatus::complete;
}

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc)
{
    Flag(rc, "Printing electron diagnostics");

    // Output any diagnostics after a step completes

    Flag(rc, "Printed");
    return TaskStatus::complete;
}

void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // Any variables or diagnostics that should be computed especially for output to a file,
    // but which are not otherwise updated.
}

} // namespace B_FluxCT
