/* 
 *  File: b_cd.hpp
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
#include "types.hpp"

#include <parthenon/parthenon.hpp>

#include <memory>

using namespace parthenon;

/**
 * This physics package implements B field transport with Flux-CT (Toth 2000)
 *
 * This requires only the values at cell centers
 * 
 * This implementation includes conversion from "primitive" to "conserved" B and back,
 * i.e. between field strength and flux via multiplying by gdet.
 */
namespace B_CD {
/**
 * Declare fields, initialize (few) parameters
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages);

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived".
 * Also applies floors to the calculated primitives, and fixes up any inversion errors
 *
 * input: Conserved B = sqrt(-gdet) * B^i
 * output: Primitive B = B^i
 */
void UtoP(MeshBlockData<Real> *rc, IndexDomain domain=IndexDomain::entire, bool coarse=false);
inline void FillDerived(MeshBlockData<Real> *rc) { UtoP(rc); }

/**
 * Add the source term to dUdt, before it is applied to U
 */
TaskStatus AddSource(MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt);

/**
 * Take a maximum over the divB array, which is updated every step
 * 
 * Used as a Parthenon History function, so must take exactly the
 * listed arguments
 */
Real MaxDivB(MeshData<Real> *md);

/**
 * Diagnostics printed/computed after each step
 * Currently nothing, divB is calculated in fluxes.cpp
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc);

/**
 * Fill fields which are calculated only for output to file
 * Currently nothing, soon the corner-centered divB values
 */
void FillOutput(MeshBlock *pmb, ParameterInput *pin);

/**
 * Turn the primitive B field into the local conserved flux
 */
KOKKOS_INLINE_FUNCTION void prim_to_flux(const GRCoordinates& G, const ScratchPad2D<Real> &P, const VarMap& m_p, const FourVectors D,
                                           const int& k, const int& j, const int& i, const int& dir,
                                           ScratchPad2D<Real>& flux, const VarMap& m_u, const Loci& loc=Loci::center)
{
    Real gdet = G.gdet(loc, j, i);
    if (dir == 0) { // Parent is templated on dir, so we should get the speed here still
        VLOOP flux(m_u.B1 + v, i) = P(m_p.B1 + v, i) * gdet;
        flux(m_u.PSI, i) = P(m_p.PSI, i) * gdet;
    } else {
        // Dual of Maxwell tensor
        // Dedner would have e.g. P(m.psip, i) * gdet,
        // but for us this is in the source term
        VLOOP flux(m_u.B1 + v, i) = (D.bcon[v+1] * D.ucon[dir] - D.bcon[dir] * D.ucon[v+1]) * gdet;
        // Psi field update as in Mosta et al (IllinoisGRMHD), alternate explanation Jesse et al (2020)
        //Real alpha = 1. / sqrt(-G.gcon(Loci::center, j, i, 0, 0));
        //Real beta_dir = G.gcon(Loci::center, j, i, 0, dir) * alpha * alpha;
        flux(m_u.PSI, i) = (D.bcon[dir] - G.gcon(Loci::center, j, i, 0, dir) * P(m_p.PSI, i)) * gdet;
    }
}

KOKKOS_INLINE_FUNCTION void p_to_u(const GRCoordinates& G, const VariablePack<Real>& P, const VarMap& m_p,
                                           const int& k, const int& j, const int& i,
                                           const VariablePack<Real>& U, const VarMap& m_u, const Loci& loc=Loci::center)
{
    Real gdet = G.gdet(loc, j, i);
    VLOOP U(m_u.B1 + v, k, j, i) = P(m_p.B1 + v, k, j, i) * gdet;
    U(m_u.PSI, k, j, i) = P(m_p.PSI, k, j, i) * gdet;

}

}