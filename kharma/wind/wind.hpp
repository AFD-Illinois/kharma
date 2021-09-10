/* 
 *  File: wind.hpp
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

#include "mhd_functions.hpp"

#include <parthenon/parthenon.hpp>

namespace Wind {

/**
 * Initialize the wind package with several options from the input deck
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

/**
 * Add the wind source term.  Applied just after the FluxDivergence/ApplyFluxes calculation
 */
TaskStatus AddWind(MeshData<Real> *mdudt, double time);

/**
 * Function to add a "wind" source term, in addition to the usual GRMHD coordinate source term
 * 
 * This is a purely geometry-based particle source concentrated near the poles,
 * which prevents too many fixups & floor hits on the coordinate pole.
 * It has proven effective at eliminating the use of floors in 2D, but less so in 3D/MAD simulations
 */
KOKKOS_INLINE_FUNCTION void add_wind(const GRCoordinates &G,
                      const Real& gam, const int& k, const int& j, const int& i,
                      const Real& n, const int& power, const Real& Tp,
                      const VariablePack<Real>& dUdt, const VarMap& m)
{

    // Need coordinates to evaluate particle addtn rate
    // Note that makes the wind spherical-only, TODO ensure this
    GReal Xembed[GR_DIM];
    G.coord_embed(k, j, i, Loci::center, Xembed);
    GReal r = Xembed[1], th = Xembed[2];

    // Particle addition rate: concentrate at poles & center
    // TODO poles only w/e.g. cos2?
    Real drhopdt = n * pow(cos(th), power) / pow(1. + r * r, 2);

    // Insert fluid in normal observer frame, without B field
    const Real uvec[NVEC] = {0}, B_P[NVEC] = {0};

    // Add plasma to the T^t_a component of the stress-energy tensor
    // Notice that U already contains a factor of sqrt{-g}
    Real rho_ut, T[GR_DIM];
    GRMHD::p_to_u_loc(G, drhopdt, drhopdt * Tp * 3., uvec, B_P, gam, k, j, i, rho_ut, T);

    dUdt(m.RHO, k, j, i) += rho_ut;
    dUdt(m.UU, k, j, i) += T[0];
    dUdt(m.U1, k, j, i) += T[1];
    dUdt(m.U2, k, j, i) += T[2];
    dUdt(m.U3, k, j, i) += T[3];
}

}