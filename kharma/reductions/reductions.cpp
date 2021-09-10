/* 
 *  File: reductions.hpp
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


    bool add_fluxes = pin->GetOrAddBoolean("reductions", "add_fluxes", false);
    params.Add("add_fluxes", add_fluxes);

    // List (vector) of HistoryOutputVar that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, MdotBound, "Mdot"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, MdotEH, "Mdot_EH"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, EdotBound, "Edot"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, EdotEH, "Edot_EH"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, LdotBound, "Ldot"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, LdotEH, "Ldot_EH"));

    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, MdotBoundFlux, "Mdot_Flux"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, MdotEHFlux, "Mdot_EH_Flux"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, EdotBoundFlux, "Edot_Flux"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, EdotEHFlux, "Edot_EH_Flux"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, LdotBoundFlux, "Ldot_Flux"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, LdotEHFlux, "Ldot_EH_Flux"));

    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, TotalM, "Mass"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, TotalE, "Egas"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, TotalL, "Ang_Mom"));

    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, TotalEHTLum, "EHT_Lum_Proxy"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, TotalJetLum, "Jet_Lum"));

    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, NPFlags, "Num_PFlags"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, NFFlags, "Num_FFlags"));
    // add callbacks for HST output identified by the `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}