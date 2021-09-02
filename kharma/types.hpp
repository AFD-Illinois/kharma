/* 
 *  File: types.hpp
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

#include <parthenon/parthenon.hpp>

// KHARMA TYPES

// Denote reconstruction algorithms
enum ReconstructionType{donor_cell=0, linear_mc, linear_vl, ppm, mp5, weno5, weno5_lower_poles};

// Denote inversion failures (pflags). See U_to_P for status explanations
enum InversionStatus{success=0, neg_input, max_iter, bad_ut, bad_gamma, neg_rho, neg_u, neg_rhou};

// Floor codes are non-exclusive, so it makes little sense to use an enum
// Instead, we use bitflags, starting high enough that we can stick the enum in the bottom 5 bits
// See floors.hpp for explanations of the flags
#define HIT_FLOOR_GEOM_RHO 32
#define HIT_FLOOR_GEOM_U 64
#define HIT_FLOOR_B_RHO 128
#define HIT_FLOOR_B_U 256
#define HIT_FLOOR_TEMP 512
#define HIT_FLOOR_GAMMA 1024
#define HIT_FLOOR_KTOT 2048
#define HIT_FLOOR_GEOM_RHO_FLUX 4096
#define HIT_FLOOR_GEOM_U_FLUX 8192

// Struct for derived 4-vectors at a point, usually calculated and needed together
typedef struct {
    parthenon::Real ucon[GR_DIM];
    parthenon::Real ucov[GR_DIM];
    parthenon::Real bcon[GR_DIM];
    parthenon::Real bcov[GR_DIM];
} FourVectors;

// Map of the locations of particular variables in a VariablePack
// Used for operations conducted over all vars which must still
// distinguish them, e.g. fluxes.hpp
class VarMap {
    public:
        int8_t RHO, UU, U1, U2, U3, B1, B2, B3, PSI, PASSIVE;
        int8_t KTOT, K_HOWES, K_KAWAZURA, K_WERNER, K_ROWAN, K_SHARMA;

        VarMap(parthenon::PackIndexMap& name_map, bool is_cons)
        {
            if (is_cons) {
                RHO = name_map["cons.rho"].first;
                UU = name_map["cons.u"].first;
                U1 = name_map["cons.uvec"].first;
                B1 = name_map["cons.B"].first;
                PSI = name_map["cons.psi_cd"].first;
                KTOT = name_map["cons.Ktot"].first;
                K_HOWES = name_map["cons.Kel_Howes"].first;
                K_KAWAZURA = name_map["cons.Kel_Kawazura"].first;
                K_WERNER = name_map["cons.Kel_Werner"].first;
                K_ROWAN = name_map["cons.Kel_Rowan"].first;
                K_SHARMA = name_map["cons.Kel_Sharma"].first;
            } else {
                RHO = name_map["prims.rho"].first;
                UU = name_map["prims.u"].first;
                U1 = name_map["prims.uvec"].first;
                B1 = name_map["prims.B"].first;
                PSI = name_map["prims.psi_cd"].first;
                KTOT = name_map["prims.Ktot"].first;
                K_HOWES = name_map["prims.Kel_Howes"].first;
                K_KAWAZURA = name_map["prims.Kel_Kawazura"].first;
                K_WERNER = name_map["prims.Kel_Werner"].first;
                K_ROWAN = name_map["prims.Kel_Rowan"].first;
                K_SHARMA = name_map["prims.Kel_Sharma"].first;
            }
            U2 = U1 + 1;
            U3 = U1 + 2;
            B2 = B1 + 1;
            B3 = B1 + 2;
        }
};

/**
 * Struct to hold floor values without cumbersome dictionary/string logistics.
 * Hopefully faster.
 */
class FloorPrescription {
    public:
        // Purely geometric limits
        double rho_min_geom, u_min_geom, r_char;
        // Dynamic limits on magnetization/temperature
        double bsq_over_rho_max, bsq_over_u_max, u_over_rho_max;
        // Limit entropy
        double ktot_max;
        // Limit fluid Lorentz factor
        double gamma_max;
        // Floor options
        bool temp_adjust_u, fluid_frame;

        FloorPrescription(parthenon::Params& params)
        {
            rho_min_geom = params.Get<Real>("rho_min_geom");
            u_min_geom = params.Get<Real>("u_min_geom");
            r_char = params.Get<Real>("floor_r_char");

            bsq_over_rho_max = params.Get<Real>("bsq_over_rho_max");
            bsq_over_u_max = params.Get<Real>("bsq_over_u_max");
            u_over_rho_max = params.Get<Real>("u_over_rho_max");
            ktot_max = params.Get<Real>("ktot_max");
            gamma_max = params.Get<Real>("gamma_max");

            temp_adjust_u = params.Get<bool>("temp_adjust_u");
            fluid_frame = params.Get<bool>("fluid_frame");
        }
};