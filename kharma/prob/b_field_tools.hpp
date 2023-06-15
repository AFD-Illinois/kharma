/* 
 *  File: b_field_tools.hpp
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

// Internal representation of the field initialization preference for quick switch
// Avoids string comparsion in kernels
enum BSeedType{constant, monopole, sane, ryan, ryan_quadrupole, r3s3, steep, gaussian, bz_monopole, vertical, r1s2};

/**
 * Function to parse a string indicating desired field to a BSeedType
 */
inline BSeedType ParseBSeedType(std::string b_field_type)
{
    if (b_field_type == "constant") {
        return BSeedType::constant;
    } else if (b_field_type == "monopole") {
        return BSeedType::monopole;
    } else if (b_field_type == "sane") {
        return BSeedType::sane;
    } else if (b_field_type == "mad" || b_field_type == "ryan") {
        return BSeedType::ryan;
    } else if (b_field_type == "mad_quadrupole" || b_field_type == "ryan_quadrupole") {
        return BSeedType::ryan_quadrupole;
    } else if (b_field_type == "r3s3") {
        return BSeedType::r3s3;
    } else if (b_field_type == "mad_steep" || b_field_type == "steep") {
        return BSeedType::steep;
    } else if (b_field_type == "gaussian") {
        return BSeedType::gaussian;
    } else if (b_field_type == "bz_monopole") {
        return BSeedType::bz_monopole;
    } else if (b_field_type == "vertical") {
        return BSeedType::vertical;
    } else if (b_field_type == "r1s2") {
        return BSeedType::r1s2;
    } else {
        throw std::invalid_argument("Magnetic field seed type not supported: " + b_field_type);
    }
}

/**
 * Get the minimum value of plasma beta on the (physical, non-ghost) domain
 * 
 * Likely not actually what you want
 */
Real GetLocalBetaMin(parthenon::MeshBlockData<Real> *rc);

/**
 * Get the maximum/minimum value of b^2 (twice the magnetic field pressure)
 * over the domain.  Latter a good check for >0 & for constant-field init.
 */
Real GetLocalBsqMax(parthenon::MeshBlockData<Real> *rc);
Real GetLocalBsqMin(parthenon::MeshBlockData<Real> *rc);

/**
 * Get the maximum fluid pressure over the domain
 */
Real GetLocalPMax(parthenon::MeshBlockData<Real> *rc);

/**
 * Normalize the magnetic field by dividing by 'factor'
 * 
 * LOCKSTEP: this function expects and preserves P==U
 */
TaskStatus NormalizeBField(parthenon::MeshBlockData<Real> *rc, Real factor);
