/* 
 *  File: containers.cpp
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

#include "containers.hpp"

#include "decs.hpp"

using namespace Kokkos;

TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string>& stage_name,
                            Integrator* integrator) {
    const Real beta = integrator->beta[stage-1];
    Real dt = integrator->dt;
    auto& base = pmb->real_containers.Get();
    auto& cin = pmb->real_containers.Get(stage_name[stage-1]);
    auto& cout = pmb->real_containers.Get(stage_name[stage]);
    auto& dudt = pmb->real_containers.Get("dUdt");
    parthenon::Update::AverageContainers(cin, base, beta);
    parthenon::Update::UpdateContainer(cin, dudt, beta*dt, cout);
    return TaskStatus::complete;
}

TaskStatus CopyField(std::string& var, std::shared_ptr<Container<Real>>& rc0, std::shared_ptr<Container<Real>>& rc1)
{
    auto pmb = rc0->GetBlockPointer();
    GridVars v0 = rc0->Get(var).data;
    GridVars v1 = rc1->Get(var).data;
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

    // TODO revisit this when par_for is restored to glory
    Kokkos::parallel_for("copy_field", MDRangePolicy<Rank<4>>({0, ks, js, is}, {NPRIM, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA_VARS {
            v1(p, k, j, i) = v0(p, k, j, i);
        }
    );
    return TaskStatus::complete;
}