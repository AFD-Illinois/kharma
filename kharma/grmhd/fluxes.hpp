/* 
 *  File: fluxes.hpp
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

#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"

#include "decs.hpp"

#include "reconstruction.hpp"
#include "phys.hpp"

namespace LLF {
/**
 * Take reconstructed primitives at left and right face, and construct the local Lax-Friedrichs flux
 *
 * Also fills the "ctop" vector with the highest magnetosonic speed mhd_vchar -- used to estimate timestep later.
 *
 * Note that since this L and R are defined with respect to the *face*, they are actually the
 * opposite of the "r" and "l" in the caller, CalculateFluxes!
 */
TaskStatus LRToFlux(std::shared_ptr<Container<Real>>& rc, GridVars pl, GridVars pr, const int dir, GridVars flux);

/**
 * Reconstruct and calculate the flux, interleaved.  See Reconstruction namespace and LRToFlux for details
 */
TaskStatus ReconAndFlux(std::shared_ptr<Container<Real>>& rc, const int& dir);

}