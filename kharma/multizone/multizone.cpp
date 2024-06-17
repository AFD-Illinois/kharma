/* 
 *  File: multizone.cpp
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
#include "kharma.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

std::shared_ptr<KHARMAPackage> Multizone::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Electrons");
    Params &params = pkg->AllParams();

    // Evolution parameters
    Real gamma_e = pin->GetOrAddReal("electrons", "gamma_e", 4./3);
    params.Add("gamma_e", gamma_e);
    Real gamma_p = pin->GetOrAddReal("electrons", "gamma_p", 5./3);
    params.Add("gamma_p", gamma_p);

    

    //pkg->BlockUtoP = Electrons::BlockUtoP;
    //pkg->BoundaryUtoP = Electrons::BlockUtoP;

    return pkg;
}

void DecideActiveBlocks(Mesh *pmesh, bool *is_active, bool **apply_boundary_condition)
{

}

TaskStatus AverageEMFSeams(MeshData *pmesh, bool **apply_boundary_condition)
{

}

TaskStatus Multizone::PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc)
{
    Flag("PostStepDiagnostics");

    // Output any diagnostics after a step completes

    EndFlag();
    return TaskStatus::complete;
}

