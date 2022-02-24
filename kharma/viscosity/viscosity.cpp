/* 
 *  File: viscosity.cpp
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
#include "viscosity.hpp"

#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

namespace Viscosity
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    auto pkg = std::make_shared<StateDescriptor>("Viscosity");
    Params &params = pkg->AllParams();

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // Floors & fluid gamma
    // Any parameters, like above

    MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");
    MetadataFlag isNonideal = Metadata::AllocateNewFlag("Nonideal");
    params.Add("NonidealFlag", isNonideal);

    // General options for primitive and conserved scalar variables in KHARMA
    Metadata m_con  = Metadata({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                 Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes, isNonideal});
    Metadata m_prim = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
                  isPrimitive, isNonideal});

    // Heat conduction
    pkg->AddField("cons.q", m_con);
    pkg->AddField("prims.q", m_prim);
    // Pressure anisotropy
    pkg->AddField("cons.dP", m_con);
    pkg->AddField("prims.dP", m_prim);
    // Eventually also need (most or all of) Theta, bsq, nu_emhd, chi_emhd, tau

    // If we want to register viscosity-specific UtoP for some reason?
    // Likely we'll only use the post-step summary hook
    //pkg->FillDerivedBlock = Viscosity::FillDerived;
    //pkg->PostFillDerivedBlock = Viscosity::PostFillDerived;
    return pkg;
}

} // namespace Viscosity
