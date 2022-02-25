/* 
 *  File: imex_driver.hpp
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

#include <memory>

#include <parthenon/parthenon.hpp>

using namespace parthenon;

/**
 * A Driver object orchestrates everything that has to be done to a mesh to constitute a step.
 * This driver does pretty much the same thing as the HARMDriver, with one important difference:
 * ImexDriver syncs primitive variables and treats them as fundamental, whereas HARMDriver syncs conserved variables.
 * This allows ImexDriver to optionally use a semi-implicit step, adding a per-zone implicit solve via the 'Implicit'
 * package, instead of just explicit RK2 time-stepping.  This driver also allows explicit-only RK2 operation
 */
class ImexDriver : public MultiStageDriver {
    public:
        /**
         * Default constructor
         */
        ImexDriver(ParameterInput *pin, ApplicationInput *papp, Mesh *pm) : MultiStageDriver(pin, papp, pm) {}

        /**
         * All the tasks which constitute advancing the fluid in a mesh by one stage.
         * This includes calculation of the primitives and reconstruction of their face values,
         * calculation of conserved values and fluxes thereof at faces,
         * application of fluxes and a source term in order to update zone values,
         * and finally calculation of the next timestep based on the CFL condition.
         * 
         * The function is heavily documented since order changes can introduce subtle bugs,
         * usually w.r.t. fluid "state" being spread across the primitive and conserved quantities
         */
        TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
};
