/* 
 *  File: models.hpp
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

// Include the models and stuff them into a Variant type
#include "unpol_comparison.hpp"

using SomeEmissionModel = mpark::variant<Model_UnpolComp>;

using namespace std;

/**
 * Ipole needs to be able to load emission models at runtime
 * They need to be fast
 * They need to be easy to write
 * 
 * ... so here we are.
 */
class EmissionModel {
    public:
        SomeEmissionModel model;

        // Common code for constructors
#pragma hd_warning_disable
        KOKKOS_FUNCTION void EmplaceModel(const SomeEmissionModel& model) {
            // Isn't there some more elegant way to say "yeah the types are fine just copy da bits"?
            if (mpark::holds_alternative<Model_UnpolComp>(base_in)) {
                base.emplace<Model_UnpolComp>(mpark::get<Model_UnpolComp>(base_in));
            } else {
                printf("Tried to copy invalid base coordinates!");
                //throw std::invalid_argument("Tried to copy invalid base coordinates!");
            }
        }
        // Constructors
#pragma hd_warning_disable
        EmissionModel() = default;
#pragma hd_warning_disable
        KOKKOS_FUNCTION EmissionModel(SomeEmissionModel& model_in): model(model_in) {}
#pragma hd_warning_disable
        KOKKOS_FUNCTION EmissionModel(const EmissionModel& src)
        {
            EmplaceSystems(src.model);
        }
#pragma hd_warning_disable
        KOKKOS_FUNCTION const EmissionModel& operator=(const EmissionModel& src)
        {
            EmplaceSystems(src.model);
            return *this;
        }
}