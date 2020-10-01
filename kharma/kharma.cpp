/* 
 *  File: kharma.cpp
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

#include <iostream>

#include <parthenon/parthenon.hpp>

#include "decs.hpp"
#include "bondi.hpp"
#include "boundaries.hpp"
#include "containers.hpp"
#include "fixup.hpp"
#include "grmhd.hpp"
#include "harm.hpp"
#include "kharma.hpp"
#include "iharm_restart.hpp"

Properties_t KHARMA::ProcessProperties(std::unique_ptr<ParameterInput>& pin)
{
    // TODO actually use this?  Just globals, basically, maybe useful for debug flags etc.
    Properties_t properties;

    // Mostly this function is where I've chosen to mess with all Parthenon's parameters before
    // handing them over.  This includes reading restarts, setting native boundaries from KS, etc.

    // If we're restarting (not via Parthenon), read the restart file to get most parameters
    std::string prob = pin->GetString("parthenon/job", "problem_id");
    if (prob == "iharm_restart") {
        ReadIharmRestartHeader(pin->GetString("iharm_restart", "fname"), pin);
    }

    // TODO somehow only parse the coordinate system once, so we can know exactly whether we're spherical/modified
    // So far every non-null transform is exp(x1) but who knows
    std::string cb = pin->GetString("coordinates", "base");
    std::string ctf = pin->GetOrAddString("coordinates", "transform", "null");
    if (ctf != "null") {
        // Set Rin such that we have 5 zones completely inside the event horizon
        // If xeh = log(Rhor), xin = log(Rin), and xout = log(Rout),
        // then we want xeh = xin + 5.5 * (xout - xin) / N1TOT, or solving/replacing:
        int n1tot = pin->GetInteger("parthenon/mesh", "nx1");
        GReal Rout = pin->GetReal("coordinates", "r_out");
        Real a = pin->GetReal("coordinates", "a");
        GReal Rhor = 1 + sqrt(1 - a*a);
        GReal x1max = log(Rout);
        GReal x1min = (n1tot * log(Rhor) / 5.5 - x1max) / (-1. + n1tot / 5.5);
        if (x1min < 0.0) {
            throw std::invalid_argument("Not enough radial zones were specified to put 5 zones inside EH!");
        }
        //cerr << "Setting x1min: " << x1min << " x1max " << x1max << " based on BH with a=" << a << endl;
        pin->SetReal("parthenon/mesh", "x1min", x1min);
        pin->SetReal("parthenon/mesh", "x1max", x1max);
    }
    // Assumption: if we're in a spherical system...
    if (cb == "spherical_ks" || cb == "ks" || cb == "spherical_bl" || cb == "bl" || cb == "spherical_minkowski") {
        // ...then we definitely want spherical boundary conditions
        // TODO only set all this if it isn't already
        pin->SetString("parthenon/mesh", "ix1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ox1_bc", "outflow");
        pin->SetString("parthenon/mesh", "ix2_bc", "reflecting");
        pin->SetString("parthenon/mesh", "ox2_bc", "reflecting");
        pin->SetString("parthenon/mesh", "ix3_bc", "periodic");
        pin->SetString("parthenon/mesh", "ox3_bc", "periodic");

        // We also know the bounds for most transforms in spherical.  Set them.
        if (ctf == "none") {
            pin->SetReal("parthenon/mesh", "x2min", 0.0);
            pin->SetReal("parthenon/mesh", "x2max", M_PI);
            pin->SetReal("parthenon/mesh", "x3min", 0.0);
            pin->SetReal("parthenon/mesh", "x3max", 2*M_PI);
        } else if (ctf == "modified" || ctf == "mks" || ctf == "funky" || ctf == "fmks") {
            pin->SetReal("parthenon/mesh", "x2min", 0.0);
            pin->SetReal("parthenon/mesh", "x2max", 1.0);
            pin->SetReal("parthenon/mesh", "x3min", 0.0);
            pin->SetReal("parthenon/mesh", "x3max", 2*M_PI);
        } // TODO any other transforms/systems
    }  // TODO assume periodic conditions for all Cartesian simulations?

    return properties;
}

Packages_t KHARMA::ProcessPackages(std::unique_ptr<ParameterInput>& pin)
{
    Packages_t packages;

    // Turn off GRMHD only if set to false in input file
    bool do_grmhd = pin->GetOrAddBoolean("Packages", "GRMHD", true);
    bool do_grhd = pin->GetOrAddBoolean("Packages", "GRHD", false);
    bool do_electrons = pin->GetOrAddBoolean("Packages", "howes_electrons", false);

    // enable other packages as needed
    bool do_scalars = pin->GetOrAddBoolean("Packages", "scalars", false);

    // Just one base package: integrated B-fields, or not.
    if (do_grmhd) {
        packages["GRMHD"] = GRMHD::Initialize(pin.get());
    } else if (do_grhd) {

    }

    // Scalars can be added 
    // if (do_scalars) {
    //     packages["scalars"] = BetterScalars::Initialize(pin.get());
    // }

    // TODO electrons, like scalars but w/heating step...

    return std::move(packages);
}