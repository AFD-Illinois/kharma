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

std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Electrons");
    Params &params = pkg->AllParams();

    // Evolution parameters
    Real gamma_e = pin->GetOrAddReal("electrons", "gamma_e", 4./3);
    params.Add("gamma_e", gamma_e);
    Real gamma_p = pin->GetOrAddReal("electrons", "gamma_p", 5./3);
    params.Add("gamma_p", gamma_p);
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

    // Initialization
    bool init_to_fel_0 = pin->GetOrAddBoolean("electrons", "init_to_fel_0", true);
    params.Add("init_to_fel_0", init_to_fel_0);
    Real fel_0 = pin->GetOrAddReal("electrons", "fel_0", 0.01);
    params.Add("fel_0", fel_0);

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
    // TODO pretty soon this can be a GetVector<std::string>!!!
    // std::vector<Real> masses = parse_list(pin->GetOrAddString("units", "MBH", "1.0"));
    // if (masses != std::vector<Real>{1.0})
    // {
    //     std::vector<std::vector<Real>> munits;
    //     for (int i=1; i <= masses.size(); ++i) {
    //         munits.push_back(parse_list(pin->GetString("units", "M_unit_" + std::to_string(i))));
    //     }

    //     if (MPIRank0() && packages->Get("Globals")->Param<int>("verbose") > 0) {
    //         std::cout << "Using unit sets:" << std::endl;
    //         for (int i=0; i < masses.size(); ++i) {
    //             std::cout << std::endl << masses[i] << ":";
    //             for (auto munit : munits[i]) {
    //                 std::cout << " " << munit;
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    //     // This is a vector of Reals
    //     params.Add("masses", masses);
    //     // This is a vector of vectors of Reals
    //     params.Add("munits", munits);
    // }

    // Default implicit iff GRMHD is done implicitly. TODO can we do explicit?
    auto& driver = packages->Get("Driver")->AllParams();
    auto driver_type = driver.Get<std::string>("type");
    bool grmhd_implicit = packages->Get("GRMHD")->Param<bool>("implicit"); // usually false
    bool implicit_e = (driver_type == "imex" && pin->GetOrAddBoolean("electrons", "implicit", grmhd_implicit)); // so this false too
    params.Add("implicit", implicit_e);

    Metadata::AddUserFlag("Electrons");
    MetadataFlag areWeImplicit = (implicit_e) ? Metadata::GetUserFlag("Implicit")
                                              : Metadata::GetUserFlag("Explicit");

    std::vector<MetadataFlag> flags_cons = {Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::Conserved,
                                            Metadata::WithFluxes, Metadata::FillGhost, areWeImplicit, Metadata::GetUserFlag("Electrons")};
    std::vector<MetadataFlag> flags_prim = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::GetUserFlag("Primitive"),
                                            Metadata::Restart, areWeImplicit, Metadata::GetUserFlag("Electrons")};

    // Total entropy, used to track changes
    int nKs = 1;
    pkg->AddField("cons.Ktot", flags_cons);
    pkg->AddField("prims.Ktot", flags_prim);

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

    // Update variable numbers
    if (implicit_e) {
        int n_current = driver.Get<int>("n_implicit_vars");
        driver.Update("n_implicit_vars", n_current+nKs);
    } else {
        int n_current = driver.Get<int>("n_explicit_vars");
        driver.Update("n_explicit_vars", n_current+nKs);
    }

    // Problem-specific fields
    if (packages->Get("Globals")->Param<std::string>("problem") == "driven_turbulence") {
        std::vector<int> s_vector({2});
        Metadata m_vector = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
        Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("grf_normalized", m_vector);
        pkg->AddField("alfven_speed", m);
    }

    pkg->BlockUtoP = Electrons::BlockUtoP;

    return pkg;
}

TaskStatus InitElectrons(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    Flag("Initializing electron/fluid entropy values");
    auto pmb = rc->GetBlockPointer();

    // Don't initialize entropies if we've already done so e.g. in Hubble problem
    if (!pmb->packages.Get("Electrons")->Param<bool>("init_to_fel_0")) {
        return TaskStatus::complete;
    }

    // Need to distinguish KTOT from the other variables, so we record which it is
    PackIndexMap prims_map;
    auto& e_P = rc->PackVariables({Metadata::GetUserFlag("Electrons"), Metadata::GetUserFlag("Primitive")}, prims_map);
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
        KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
            if (p == ktot_index) {
                // Initialize total entropy by definition,
                e_P(p, k, j, i) = (gam - 1.) * u(k, j, i) * m::pow(rho(k, j, i), -gam);
            } else {
                // and e- entropy by given constant initial fraction
                e_P(p, k, j, i) = (game - 1.) * fel0 * u(k, j, i) * m::pow(rho(k, j, i), -game);
            }
        }
    );

    // iharm3d syncs bounds here, but we do all that in PostInit

    Flag("Initialized electron/fluid entropy values");
    return TaskStatus::complete;
}

void BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "UtoP electrons");
    auto pmb = rc->GetBlockPointer();

    // No need for a "map" here, we just want everything that fits these
    auto& e_P = rc->PackVariables({Metadata::GetUserFlag("Electrons"), Metadata::GetUserFlag("Primitive")});
    auto& e_U = rc->PackVariables({Metadata::GetUserFlag("Electrons"), Metadata::Conserved});
    // And then the local density
    GridScalar rho_U = rc->Get("cons.rho").data;

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int is = bounds.is(domain), ie = bounds.ie(domain);
    int js = bounds.js(domain), je = bounds.je(domain);
    int ks = bounds.ks(domain), ke = bounds.ke(domain);
    pmb->par_for("UtoP_electrons", 0, e_P.GetDim(4)-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
            e_P(p, k, j, i) = e_U(p, k, j, i) / rho_U(k, j, i);
        }
    );
}

void BlockPtoU(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "PtoU electrons");
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    auto& U = rc->PackVariables({Metadata::Conserved}, cons_map);
    const VarMap m_p(prims_map, false), m_u(cons_map, true);
    // And then the local density
    GridScalar rho_P = rc->Get("cons.rho").data;

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    int is = bounds.is(domain), ie = bounds.ie(domain);
    int js = bounds.js(domain), je = bounds.je(domain);
    int ks = bounds.ks(domain), ke = bounds.ke(domain);
    pmb->par_for("PtoU_electrons", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Electrons::p_to_u(G, P, m_p, k, j, i, U, m_u);
        }
    );
}

TaskStatus ApplyElectronHeating(MeshBlockData<Real> *rc_old, MeshBlockData<Real> *rc)
{   // takes in '_sub_step_init' and '_sub_step_final'
    Flag(rc, "Applying electron heating");
    auto pmb = rc->GetBlockPointer();

    // Need to distinguish different electron models
    // So far, Parthenon's maps of the same sets of variables are consistent,
    // so we only bother with one map of the primitives
    // TODO Parthenon can definitely build a pack from a map, though
    PackIndexMap prims_map, cons_map;
    auto& P = rc_old->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    auto& P_new = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
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
    // See kharma_step.cpp for the full picture of what gets updated when.
    const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
    pmb->par_for("heat_electrons", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            FourVectors Dtmp;
            GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
            Real bsq = dot(Dtmp.bcon, Dtmp.bcov);

            // Calculate the new total entropy in this cell considering heating
            const Real k_energy_conserving = (gam-1.) * P_new(m_p.UU, k, j, i) / m::pow(P_new(m_p.RHO, k, j, i), gam);

            // Dissipation is the real entropy k_energy_conserving minus any advected entropy from the previous (sub-)step P_new(KTOT)
            Real diss_tmp = (game-1.) / (gam-1.) * m::pow(P(m_p.RHO, k, j, i), gam - game) * (k_energy_conserving - P_new(m_p.KTOT, k, j, i));
            //this is eq27                  ratio of heating: Qi/Qe                           advected entropy from prev step
            // ^ denotes the solution corresponding to entropy conservation

            // Under the flag "suppress_highb_heat", we set all dissipation to zero at sigma > 1.
            diss_tmp = (suppress_highb_heat && (bsq / P(m_p.RHO, k, j, i) > 1.)) ? 0.0 : diss_tmp;

            // Default is True diss_sign == Enforce nonnegative
            // Due to floors we can end up with diss==0 or even *slightly* <0, so we require it to be positive here
            const Real diss = pmb->packages.Get("Electrons")->Param<bool>("diss_sign") ? m::max(diss_tmp, 0.0) : diss_tmp;

            // Reset the entropy to measure next (sub-)step's dissipation
            P_new(m_p.KTOT, k, j, i) = k_energy_conserving;

            // We'll be applying floors inline as we heat electrons, so
            // we cache the floors as entropy limits so they'll be cheaper to apply.
            // Note tp_te_min -> kel_max & vice versa
            const Real kel_max = P(m_p.KTOT, k, j, i) * m::pow(P(m_p.RHO, k, j, i), gam - game) /
                                    (tptemin * (gam - 1.) / (gamp-1.) + (gam-1.) / (game-1.)); //0.001
            const Real kel_min = P(m_p.KTOT, k, j, i) * m::pow(P(m_p.RHO, k, j, i), gam - game) /
                                    (tptemax * (gam - 1.) / (gamp-1.) + (gam-1.) / (game-1.)); //1000
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
                const Real qrat = 0.92 * (c2*c2 + beta_pow)/(c3*c3 + beta_pow) * m::exp(-1./beta) * m::sqrt(MP/ME * Trat);
                const Real fel = 1./(1. + qrat);
                P_new(m_p.K_HOWES, k, j, i) = clip(P_new(m_p.K_HOWES, k, j, i) + fel * diss, kel_min, kel_max);
            }
            if (m_p.K_KAWAZURA >= 0) {
                // Equation (2) in http://www.pnas.org/lookup/doi/10.1073/pnas.1812491116
                const Real Tel = m::max(P(m_p.K_KAWAZURA, k, j, i) * m::pow(P(m_p.RHO, k, j, i), game-1), SMALL);

                const Real Trat = Tpr / Tel;
                const Real pres = P(m_p.RHO, k, j, i) * Tpr; // Proton pressure
                const Real beta = m::min(pres / bsq * 2, 1.e20);// If somebody enables electrons in a GRHD sim

                const Real QiQe = 35. / (1. + m::pow(beta/15., -1.4) * m::exp(-0.1 / Trat));
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
                const Real fel = 0.5 * m::exp(-m::pow(1 - beta/betamax, 3.3) / (1 + 1.2*m::pow(sigma, 0.7)));
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
            // Conserved variables are updated at the end of the step
        }
    );

    Flag(rc, "Applied");
    return TaskStatus::complete;
}

} // namespace B_FluxCT
