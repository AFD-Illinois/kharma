/* 
 *  File: b_none.cpp
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

#include <parthenon/parthenon.hpp>

#include "b_none.hpp"

#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

using namespace parthenon;

namespace B_None
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    auto pkg = std::make_shared<StateDescriptor>("B_None");
    Params &params = pkg->AllParams();

    std::vector<int> s_vector({3});

    MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");

    // Require B in order to do GRMHD
    // TODO OneCopy or eliminate these entirely with if statements
    Metadata m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                  Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes}, s_vector);
    pkg->AddField("c.c.bulk.B_con", m);
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Restart, isPrimitive}, s_vector);
    pkg->AddField("c.c.bulk.B_prim", m);

    return pkg;
}

void UtoP(MeshBlockData<Real> *rc) {}


TaskStatus PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime& tm) 
{ return TaskStatus::complete; }

void FillOutput(MeshBlock *pmb, ParameterInput *pin) {}

} // namespace B_None
