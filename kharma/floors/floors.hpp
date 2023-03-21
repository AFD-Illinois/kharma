/* 
 *  File: floors.hpp
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

#include "b_flux_ct.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "inverter.hpp"
#include "emhd.hpp"
#include "reductions.hpp"

// Return which floors are hit post-reconstruction
// Currently not recorded by the caller, so disabled
#define RECORD_POST_RECON 0

namespace FFlag {
// Floor codes are non-exclusive, so it makes little sense to use an enum
// Instead, we use bitflags, starting high enough that we can stick the pflag in the bottom 5 bits
// See floors.hpp for explanations of the flags
// This is the namespaced, typed equivalent of #define
static constexpr int GEOM_RHO = 32;
static constexpr int GEOM_U = 64;
static constexpr int B_RHO = 128;
static constexpr int B_U = 256;
static constexpr int TEMP = 512;
static constexpr int GAMMA = 1024;
static constexpr int KTOT = 2048;
// Separate flags for when the floors are applied after reconstruction.
// Not yet used, as this will likely have some speed penalty paid even if
// the flags aren't written
static constexpr int GEOM_RHO_FLUX = 4096;
static constexpr int GEOM_U_FLUX = 8192;
// Lowest flag value. Needed for combining floor and other return flags
static constexpr int MINIMUM = GEOM_RHO;

// Other advantage of a namespace is including full lists for iterating over
// TODO
// 1. prettier names?
// 2. What deep majicks would allow this to be constexpr?
static const std::map<int, std::string> flag_names = {
    {GEOM_RHO, "GEOM_RHO"},
    {GEOM_U, "GEOM_U"},
    {B_RHO, "B_RHO"},
    {B_U, "B_U"},
    {GAMMA, "GAMMA"},
    {TEMP, "TEMPERATURE"},
    {KTOT, "ENTROPY"},
    {GEOM_RHO_FLUX, "GEOM_RHO_ON_RECON"},
    {GEOM_U_FLUX, "GEOM_U_ON_RECON"}};
}

namespace Floors {

/**
 * Struct to hold floor values without cumbersome dictionary/string logistics.
 * Hopefully faster than dragging the full Params object device side,
 * similar reasoning to VarMap.
 */
class Prescription {
    public:
        // Purely geometric limits
        double rho_min_geom, u_min_geom, r_char, frame_switch;
        // Dynamic limits on magnetization/temperature
        double bsq_over_rho_max, bsq_over_u_max, u_over_rho_max;
        // Limit entropy
        double ktot_max;
        // Limit fluid Lorentz factor
        double gamma_max;
        // Floor options
        bool fluid_frame, mixed_frame, drift_frame;
        bool use_r_char, temp_adjust_u, adjust_k;

        Prescription() {}
        Prescription(const parthenon::Params& params)
        {
            rho_min_geom = params.Get<Real>("rho_min_geom");
            u_min_geom   = params.Get<Real>("u_min_geom");
            r_char       = params.Get<GReal>("r_char");
            frame_switch = params.Get<GReal>("frame_switch");

            bsq_over_rho_max = params.Get<Real>("bsq_over_rho_max");
            bsq_over_u_max   = params.Get<Real>("bsq_over_u_max");
            u_over_rho_max   = params.Get<Real>("u_over_rho_max");
            ktot_max         = params.Get<Real>("ktot_max");
            gamma_max        = params.Get<Real>("gamma_max");

            use_r_char    = params.Get<bool>("use_r_char");
            temp_adjust_u = params.Get<bool>("temp_adjust_u");
            adjust_k      = params.Get<bool>("adjust_k");

            fluid_frame   = params.Get<bool>("fluid_frame");
            mixed_frame   = params.Get<bool>("mixed_frame");
            drift_frame   = params.Get<bool>("drift_frame");
        }
};

/**
 * Initialization.  Set parameters.
 */
std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages);

/**
 * Apply density and internal energy floors and ceilings
 * 
 * This function definitely applies floors (regardless of "disable_floors")
 * over the stated domain, by default the entire grid incl. ghost zones.
 * 
 * LOCKSTEP: this function respects P and returns consistent P<->U
 */
TaskStatus ApplyGRMHDFloors(MeshBlockData<Real> *rc, IndexDomain domain);

/**
 * Apply the same floors as above, in the same way, except:
 * 1. No ceilings
 * 2. Don't record results to 'fflag' or 'pflag'
 * Used for problems where some part of the domain is initialized to
 * "whatever the floor value is."
 * This function can be called even if the Floors package is not initialized.
 */
TaskStatus ApplyInitialFloors(MeshBlockData<Real> *rc, IndexDomain domain);

/**
 * Print a summary of floors hit
 */
TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md);

} // namespace Floors
