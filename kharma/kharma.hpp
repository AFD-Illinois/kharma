/* 
 *  File: kharma.hpp
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

/**
 * General preferences for KHARMA.  Anything semi-driver-independent, like loading packages, etc.
 */
namespace KHARMA {
    /**
     * This function messes with all Parthenon's parameters in-place before we hand them to the Mesh,
     * so that KHARMA decks can omit/infer some things parthenon needs.
     * This includes boundaries in spherical coordinates, coordinate system translations, etc.
     * This function also handles setting parameters from restart files
     */
    void FixParameters(std::unique_ptr<ParameterInput>& pin);

    /**
     * Currently just loads GRMHD.  Could also load GRHD only, scalars, e-, etc.
     */
    Packages_t ProcessPackages(std::unique_ptr<ParameterInput>& pin);

    /**
     * Fill any arrays that are calculated only for output, e.g.
     * divB
     * jcon
     * etc
     * 
     * This becomes a member function (!) of MeshBlock and is called for each block
     */
    void FillOutput(MeshBlock *pmb, ParameterInput *pin);

    /**
     * Print any diagnostics
     */
    void PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime& tm);
}