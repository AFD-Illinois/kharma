/* 
 *  File: reductions.cpp
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

#include "reductions.hpp"

#include <parthenon/parthenon.hpp>

std::shared_ptr<StateDescriptor> Reductions::Initialize(ParameterInput *pin)
{
    auto pkg = std::make_shared<StateDescriptor>("Reductions");
    Params &params = pkg->AllParams();

    bool add_zones = pin->GetOrAddBoolean("reductions", "add_zones_accretion", false);
    params.Add("add_zones", add_zones);
    bool add_fluxes = pin->GetOrAddBoolean("reductions", "add_fluxes_accretion", true);
    params.Add("add_fluxes", add_fluxes);
    bool add_totals = pin->GetOrAddBoolean("reductions", "add_totals", true);
    params.Add("add_totals", add_totals);
    bool add_flags = pin->GetOrAddBoolean("reductions", "add_flags", true);
    params.Add("add_flags", add_flags);

    // List (vector) of HistoryOutputVar that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    // Accretion reductions only apply in spherical coordinates
    if (pin->GetBoolean("coordinates", "spherical")) {
        // Zone-based sums
        if (add_zones) {
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, MdotBound, "Mdot"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, MdotEH, "Mdot_EH"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, EdotBound, "Edot"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, EdotEH, "Edot_EH"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, LdotBound, "Ldot"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, LdotEH, "Ldot_EH"));
        }

        // EH magnetization parameter
        // TODO option?  Or just record this always?
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, PhiBound, "Phi"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, PhiEH, "Phi_EH"));

        // Count accretion more accurately, as total flux through a spherical shell
        if (add_fluxes) {
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, MdotBoundFlux, "Mdot_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, MdotEHFlux, "Mdot_EH_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, EdotBoundFlux, "Edot_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, EdotEHFlux, "Edot_EH_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, LdotBoundFlux, "Ldot_Flux"));
            hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, LdotEHFlux, "Ldot_EH_Flux"));
        }
    }

    // Grid totals of various quantities potentially of interest
    if (add_totals) {
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, TotalM, "Mass"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, TotalE, "Egas"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, TotalL, "Ang_Mom"));

        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, TotalEHTLum, "EHT_Lum_Proxy"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, JetLum_50, "Jet_Lum"));
    }
    // Keep a slightly more granular log of flags than the usual dump cadence
    if (add_flags) {
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, NPFlags, "Num_PFlags"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, NFFlags, "Num_FFlags"));
    }

    // Possible additions:
    // 1. total 3- and 4-current numbers (best to add in "current" package)
    // 2. Luminosity proxy sums over smaller areas, e.g. just disk, just disk 3-10M, etc
    // 3. Total output power, using betagamma and/or just T^0_1 > 0
    // 4+ basically anything with MI correlated to final image MI...

    // Finally, add the whole list of callbacks to the package Params struct, using a special key
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}
