/* 
 *  File: current.cpp
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

#include "current.hpp"

std::shared_ptr<StateDescriptor> Current::Initialize(ParameterInput *pin)
{
    auto pkg = std::make_shared<StateDescriptor>("Current");
    Params &params = pkg->AllParams();

    // 4-current jcon. Calculated only for output
    std::vector<int> s_fourvector({GR_DIM});
    auto m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_fourvector);
    pkg->AddField("jcon", m);

    return pkg;
}

TaskStatus Current::CalculateCurrent(MeshBlockData<Real> *rc0, MeshBlockData<Real> *rc1, const double& dt)
{
    Flag("Calculating current");

    auto pmb = rc0->GetBlockPointer();
    auto& u_old = rc0->Get("prims.uvec").data;
    auto& B_P_old = rc0->Get("prims.B").data;
    auto& u_new = rc1->Get("prims.uvec").data;
    auto& B_P_new = rc1->Get("prims.B").data;
    auto& jcon = rc1->Get("jcon").data;
    const auto& G = pmb->coords;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    const int ndim = pmb->pmy_mesh->ndim;

    GridVars u_c("P_c", NVEC, n3, n2, n1);
    GridVector B_P_c("P_c", NVEC, n3, n2, n1);

    // Calculate time-centered primitives
    // We could pack, but we just need the vectors, U1,2,3 and B1,2,3
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    IndexRange nb = IndexRange{0, NVEC-1};
    pmb->par_for("get_center", nb.s, nb.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_VARS {
            u_c(p, k, j, i) = 0.5*(u_old(p, k, j, i) + u_new(p, k, j, i));
            B_P_c(p, k, j, i) = 0.5*(B_P_old(p, k, j, i) + B_P_new(p, k, j, i));
        }
    );

    // Calculate j^{\mu} using centered differences for active zones
    ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    nb = IndexRange{0, GR_DIM-1};
    pmb->par_for("jcon_calc", nb.s, nb.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_VEC {
            // Get sqrt{-g}*F^{mu nu} at neighboring points
            Real gF2p = 0., gF2m = 0., gF3p = 0., gF3m = 0.;
            Real gF0p = G.gdet(Loci::center, j, i) * Current::get_Fcon(G, u_new, B_P_new, 0, mu, k, j, i);
            Real gF0m = G.gdet(Loci::center, j, i) * Current::get_Fcon(G, u_old, B_P_old, 0, mu, k, j, i);
            Real gF1p = G.gdet(Loci::center, j, i+1) * Current::get_Fcon(G, u_c, B_P_c, 1, mu, k, j, i+1);
            Real gF1m = G.gdet(Loci::center, j, i-1) * Current::get_Fcon(G, u_c, B_P_c, 1, mu, k, j, i-1);
            if (ndim > 1) {
                gF2p = G.gdet(Loci::center, j+1, i) * Current::get_Fcon(G, u_c, B_P_c, 2, mu, k, j+1, i);
                gF2m = G.gdet(Loci::center, j-1, i) * Current::get_Fcon(G, u_c, B_P_c, 2, mu, k, j-1, i);
            }
            if (ndim > 2) {
                gF3p = G.gdet(Loci::center, j, i) * Current::get_Fcon(G, u_c, B_P_c, 3, mu, k+1, j, i);
                gF3m = G.gdet(Loci::center, j, i) * Current::get_Fcon(G, u_c, B_P_c, 3, mu, k-1, j, i);
            }

            // Difference: D_mu F^{mu nu} = 4 \pi j^nu
            jcon(mu, k, j, i) = 1. / (sqrt(4. * M_PI) * G.gdet(Loci::center, j, i)) *
                                ((gF0p - gF0m) / dt +
                                (gF1p - gF1m) / (2. * G.dx1v(i)) +
                                (gF2p - gF2m) / (2. * G.dx2v(j)) +
                                (gF3p - gF3m) / (2. * G.dx3v(k)));
        }
    );

    Flag("Calculated");
    return TaskStatus::complete;
}

void Current::FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    Flag("Adding current");

    // The "preserve" container will only exist after we've taken a step,
    // catch that situation
    auto& rc1 = pmb->meshblock_data.Get();
    try {
        // Get the state at beginning of the step
        auto& rc0 = pmb->meshblock_data.Get("preserve");

        // Get the duration of the last timestep from the "Globals" package
        // (see kharma.cpp)
        Real dt_last = pmb->packages.Get("Globals")->Param<Real>("dt_last");

        Current::CalculateCurrent(rc0.get(), rc1.get(), dt_last);
    } catch (const std::runtime_error& e) {
        // We expect this to happen the first step
        // We just don't need to fill jcon the first time around
    }

    Flag("Added");
}
