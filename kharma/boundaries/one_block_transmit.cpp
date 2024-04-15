/* 
 *  File: one_block_transmit.cpp
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

#include "one_block_transmit.hpp"

#include "domain.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

void KBoundaries::TransmitImpl(MeshBlockData<Real> *rc, BoundaryFace bface, bool coarse)
{
    // Get all cell-centered ghosts, minus anything just used at startup
    using FC = Metadata::FlagCollection;
    FC ghost_vars = FC({Metadata::FillGhost, Metadata::Conserved})
                  + FC({Metadata::FillGhost, Metadata::GetUserFlag("Primitive")})
                  - FC({Metadata::GetUserFlag("StartupOnly")});
    auto q = rc->PackVariables(ghost_vars, coarse);
    TransmitSetTE(rc, q, bface, coarse, false);

    FC ghost_vars_f = FC({Metadata::FillGhost, Metadata::Face})
                  - FC({Metadata::GetUserFlag("StartupOnly")});
    auto q_f = rc->PackVariables(ghost_vars_f, coarse);
    TransmitSetTE(rc, q_f, bface, coarse, true);
}

void KBoundaries::TransmitSetTE(MeshBlockData<Real> *rc, VariablePack<Real> &q,
                                        BoundaryFace bface, bool coarse, bool do_face)
{
    // We're sometimes called without any variables to sync (e.g. syncing flags, EMFs), just return
    if (q.GetDim(4) == 0) return;

    // Indices
    auto pmb = rc->GetBlockPointer();
    const bool binner = BoundaryIsInner(bface);
    const int dir = BoundaryDirection(bface);
    const auto domain = BoundaryDomain(bface);
    const auto bname = BoundaryName(bface);

    std::vector<TopologicalElement> el_list;
    if (do_face) {
        el_list = {F1, F2, F3};
    } else {
        el_list = {CC};
    }
    int el_tot = el_list.size();
    for (auto el : el_list) {
        // Set boundary/ghost zones *only*, not zones on faces
        const IndexRange3 b = KDomain::GetRange(rc, domain, el, coarse);
        const IndexRange3 bi = KDomain::GetRange(rc, IndexDomain::interior, CC);

        if (domain == IndexDomain::inner_x2 || domain == IndexDomain::outer_x2) {
            const int Nk3p = (bi.ke - bi.ks + 1); // Physical/interior *zones* in dir 3
            const int Nk3p2 = Nk3p/2;             // pi/2 of those (boundary incompatible with slice sims TODO check+error)
            const int ksp = bi.ks;                // Offset of first physical zone or face (same number)
            // Pivot element for faces is first domain face (==0), pivot for cells is between b.js/e, b.js/e+/-1
            const int jpivot = (domain == IndexDomain::inner_x2) ? ((el == FaceOf(dir)) ? b.je + 1 : b.je)
                                                                 : ((el == FaceOf(dir)) ? b.js - 1 : b.js);
            const bool do_face_invert = (el == F3);
            pmb->par_for(
                "transmitting_polar_boundary_" + bname, 0, q.GetDim(4)/el_tot-1, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                    const int ki = ((k - ksp + Nk3p2) % Nk3p) + ksp;
                    const int ji = m::abs(jpivot - j);
                    const int ii = i;
                    const Real invert = (do_face_invert || q(el, v).vector_component == X3DIR) ? -1. : 1.;
                    q(el, v, k, j, i) = invert * q(el, v, ki, ji, ii);
                }
            );
            // Explicitly zero B2 face for some reason
            if (el == FaceOf(dir)) {
                pmb->par_for(
                    "transmitting_polar_boundary_" + bname, 0, q.GetDim(4)/el_tot-1, b.ks, b.ke, jpivot, jpivot, b.is, b.ie,
                    KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                        q(el, v, k, j, i) = 0.;
                    }
                );
            }
        } else {
            throw std::runtime_error("Transmitting polar conditions only defined for X2!");
        }
    }
}
