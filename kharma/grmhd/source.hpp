/* 
 *  File: source.hpp
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

#pragma once

#include "decs.hpp"
#include "mhd_functions.hpp"

namespace GRMHD
{

/**
 * Function to apply the GRMHD source term over the entire grid.
 * 
 * Note Flux::ApplyFluxes = parthenon::FluxDivergence + GRMHD::AddSource
 */
TaskStatus AddSource(MeshData<Real> *md, MeshData<Real> *mdudt);

/**
 * Function to add the source term in the GRMHD equations T^\mu_nu \Gamma^\nu_\lam\mu
 * Does not zero input, thus can be added after the flux divergence is calculated
 */
KOKKOS_INLINE_FUNCTION void add_source(const GRCoordinates &G, const VariablePack<Real>& P, const VarMap& m_p, const FourVectors& D,
                      const Real& gam, const int& k, const int& j, const int& i,
                      const VariablePack<Real>& dUdt, const VarMap& m_u)
{
    // Get stuff we don't want to recalculate every loop iteration
    // This is basically a manual version of GRMHD::calc_tensor but saves recalculating e.g. dot(bcon, bcov) 4 times
    Real pgas = (gam - 1) * P(m_p.UU, k, j, i);
    Real bsq = dot(D.bcon, D.bcov);
    Real eta = pgas + P(m_p.RHO, k, j, i) + P(m_p.UU, k, j, i) + bsq;
    Real ptot = pgas + 0.5 * bsq;

    // Contract mhd stress tensor with connection, and multiply by metric determinant
    Real new_du[GR_DIM] = {0};
    DLOOP2 {
        Real Tmunu = (eta * D.ucon[mu] * D.ucov[nu] +
                      ptot * (mu == nu) -
                      D.bcon[mu] * D.bcov[nu]);

        for (int lam = 0; lam < GR_DIM; ++lam)
            new_du[lam] += Tmunu * G.conn(j, i, nu, lam, mu);
    }

    dUdt(m_u.UU, k, j, i) += new_du[0] * G.gdet(Loci::center, j, i);
    VLOOP dUdt(m_u.U1 + v, k, j, i) += new_du[1 + v] * G.gdet(Loci::center, j, i);
}

} // namespace GRMHD
