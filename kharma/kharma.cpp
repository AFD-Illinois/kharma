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
#include "kharma.hpp"

#include <iostream>

#include <parthenon/parthenon.hpp>

#include "decs.hpp"

// Packages
#include "b_flux_ct.hpp"
#include "b_none.hpp"
#include "b_cd_glm.hpp"
#include "grmhd.hpp"

#include "bondi.hpp"
#include "boundaries.hpp"
#include "fixup.hpp"
#include "harm_driver.hpp"
#include "iharm_restart.hpp"

Properties_t KHARMA::ProcessProperties(std::unique_ptr<ParameterInput>& pin)
{
    // TODO this could benefit from some more use
    Properties_t properties;
    //properties.push_back(std::make_shared<KHARMAProperties>("Globals"));
    //StateDescriptor globals = (static_cast<KHARMAProperties*>(properties[0].get()))->State();

    // Mostly, though, this function is where I've chosen to mess with all Parthenon's parameters before
    // handing them over.  This includes reading restarts, setting native boundaries from KS, etc.

    // Set 4 ghost zones
    // TODO allow fewer for stencil-3 schemes
    // std::string recon = pin->GetOrAddString("GRMHD", "reconstruction", "weno5");
    // if (recon != "donor_cell" && recon != "linear_mc" && recon != "linear_vl") {
    //     pin->SetInteger("parthenon/mesh", "nghost", 4);
    //     Globals::nghost = pin->GetInteger("parthenon/mesh", "nghost");
    // }
    pin->SetInteger("parthenon/mesh", "nghost", 4);
    Globals::nghost = pin->GetInteger("parthenon/mesh", "nghost");

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
        int n1tot = pin->GetInteger("parthenon/mesh", "nx1");
        GReal Rout = pin->GetReal("coordinates", "r_out");
        Real a = pin->GetReal("coordinates", "a");
        GReal Rhor = 1 + sqrt(1 - a*a);
        GReal x1max = log(Rout);
        // Set Rin such that we have 5 zones completely inside the event horizon
        // If xeh = log(Rhor), xin = log(Rin), and xout = log(Rout),
        // then we want xeh = xin + 5.5 * (xout - xin) / N1TOT:
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
    }

    // If we're using constant field of some kind, we likely *don't* want to normalize to beta_min=N
    std::string field_type = pin->GetOrAddString("b_field", "type", "none");
    if (field_type == "constant" || field_type == "monopole") {
        pin->GetOrAddBoolean("b_field", "norm", false);
    }

    return properties;
}

Packages_t KHARMA::ProcessPackages(std::unique_ptr<ParameterInput>& pin)
{
    Packages_t packages;

    // Just one base package
    packages.Add(GRMHD::Initialize(pin.get()));

    // A bunch of B fields. Put them behind a different option than b_field/type,
    // to avoid mistakes
    std::string b_field_solver = pin->GetOrAddString("b_field", "solver", "flux_ct");
    if (b_field_solver == "none") {
        packages.Add(B_None::Initialize(pin.get(), packages));
    } else if (b_field_solver == "constraint_damping" || b_field_solver == "b_cd_glm") {
        packages.Add(B_CD_GLM::Initialize(pin.get(), packages));
    } else {
        // Don't even error on bad values.  This is probably what you want,
        // and we'll check for adaptive and error later
        packages.Add(B_FluxCT::Initialize(pin.get(), packages));
    }

    // TODO scalars, electrons...

    return std::move(packages);
}

void KHARMA::FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // TODO for package in packages...
    GRMHD::FillOutput(pmb, pin);
    if (pmb->packages.AllPackages().count("B_FluxCT") > 0)
        B_FluxCT::FillOutput(pmb, pin);
    if (pmb->packages.AllPackages().count("B_CD_GLM") > 0)
        B_CD_GLM::FillOutput(pmb, pin);
    // In case there are other packages that need this
}

void KHARMA::PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime& tm)
{
    // TODO for package in packages...
    GRMHD::PostStepDiagnostics(pmesh, pin, tm);
    if (pmesh->packages.AllPackages().count("B_FluxCT") > 0)
        B_FluxCT::PostStepDiagnostics(pmesh, pin, tm);
    if (pmesh->packages.AllPackages().count("B_CD_GLM") > 0)
        B_CD_GLM::PostStepDiagnostics(pmesh, pin, tm);
}
