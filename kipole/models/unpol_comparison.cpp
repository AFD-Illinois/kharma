/* 
 *  File: unpol_comparison.cpp
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
#include "unpol_comparison.hpp"

#include "constants.hpp"

#include <parthenon/parthenon.hpp>

void Model_UnpolComp::Initialize(ParameterInput *pin)
{
    int nmodel = pin->GetOrAddInteger("unpol_comp", "model", -1);
    switch (nmodel) {
    case 1:
        A_ = 0;
        alpha_ = -3;
        height_ = 0;
        l0_ = 0;
        a_ = 0.9;
        break;
    case 2:
        A_ = 0;
        alpha_ = -2;
        height_ = 0;
        l0_ = 1;
        a_ = 0;
        break;
    case 3:
        A_ = 0;
        alpha_ = 0;
        height_ = 10. / 3;
        l0_ = 1;
        a_ = 0.9;
        break;
    case 4:
        A_ = 1.e5;
        alpha_ = 0;
        height_ = 10. / 3;
        l0_ = 1;
        a_ = 0.9;
        break;
    case 5:
        A_ = 1.e6;
        alpha_ = 0;
        height_ = 100. / 3;
        l0_ = 1;
        a_ = 0.9;
        break;
    default:
        printf("Warning: GRRT code comparison model not specified.  Loading custom values\n");
        A_ = pin->GetReal("unpol_comp", "A");
        alpha_ = pin->GetReal("unpol_comp", "alpha");
        height_ = pin->GetReal("unpol_comp", "height");
        l0_ = pin->GetReal("unpol_comp", "l0");
        a_ = pin->GetReal("unpol_comp", "a");
        break;
    }

    // Copy in the units
    L_unit_ = pin->GetReal("units", "L_unit");
    T_unit_ = pin->GetReal("units", "T_unit");
    RHO_unit_ = pin->GetReal("units", "RHO_unit");
    B_unit_ = pin->GetReal("units", "B_unit");

    // And camera frequency
    freqcgs_ = pin->GetReal("camera", "freqcgs");

    printf("Running analytic model %d:\n", nmodel);
    printf("A: %g\nalpha: %g\nh: %g\nl0: %g\n\n", A_, alpha_, height_, l0_);
}

// void WriteParameters()
// {
//     hdf5_set_directory("/header/");
//     double zero = 0;
//     hdf5_write_single_val(&zero, "t", H5T_IEEE_F64LE);
//     hdf5_write_single_val(&a, "a", H5T_IEEE_F64LE);

//     hdf5_write_single_val(&model, "model", H5T_STD_I32LE);
//     hdf5_write_single_val(&A, "A", H5T_IEEE_F64LE);
//     hdf5_write_single_val(&alpha, "alpha", H5T_IEEE_F64LE);
//     hdf5_write_single_val(&height, "height", H5T_IEEE_F64LE);
//     hdf5_write_single_val(&l0, "l0", H5T_IEEE_F64LE);

//     hdf5_make_directory("units");
//     hdf5_set_directory("/header/units/");
//     hdf5_write_single_val(&zero, "M_unit", H5T_IEEE_F64LE);
//     hdf5_write_single_val(&L_unit, "L_unit", H5T_IEEE_F64LE);
//     hdf5_write_single_val(&T_unit, "T_unit", H5T_IEEE_F64LE);

//     hdf5_set_directory("/");
// }
