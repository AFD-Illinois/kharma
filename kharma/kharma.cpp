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
#include "b_cd.hpp"
#include "current.hpp"
#include "electrons.hpp"
#include "grmhd.hpp"
#include "wind.hpp"

#include "bondi.hpp"
#include "boundaries.hpp"
#include "fixup.hpp"
#include "harm_driver.hpp"
#include "iharm_restart.hpp"

std::shared_ptr<StateDescriptor> KHARMA::InitializeGlobals(ParameterInput *pin)
{
    auto pkg = std::make_shared<StateDescriptor>("Globals");
    Params &params = pkg->AllParams();
    // Last step's dt (Parthenon SimTime tm.dt), which must be preserved to output jcon
    // Also used to prevent any 
    params.Add("dt_last", 0.0);
    // Accumulator for maximum ctop within an MPI process
    params.Add("ctop_max", 0.0);
    // Maximum between MPI processes, updated after each step
    params.Add("ctop_max_last", 0.0);
    // Whether we are computing initial outputs/timestep, or versions in the execution loop
    params.Add("in_loop", false);

    return pkg;
}

void KHARMA::FixParameters(std::unique_ptr<ParameterInput>& pin)
{
    // TODO dynamic ghost zones.  Dangerous?
    // std::string recon = pin->GetOrAddString("GRMHD", "reconstruction", "weno5");
    // if (recon != "donor_cell" && recon != "linear_mc" && recon != "linear_vl") {
    //     pin->SetInteger("parthenon/mesh", "nghost", 4);
    //     Globals::nghost = pin->GetInteger("parthenon/mesh", "nghost");
    // }
    // Set 4 ghost zones
    pin->SetInteger("parthenon/mesh", "nghost", 4);
    Globals::nghost = pin->GetInteger("parthenon/mesh", "nghost");

    // If we're restarting (not via Parthenon), read the restart file to get most parameters
    std::string prob = pin->GetString("parthenon/job", "problem_id");
    if (prob == "iharm_restart") {
        ReadIharmRestartHeader(pin->GetString("iharm_restart", "fname"), pin);
    }

    // TODO construct a coordinate system object here to check all the properties/coordinates
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
        pin->SetBoolean("coordinates", "spherical", true);
        // ...then we definitely want our special sauce boundary conditions
        // These are inflow in x1 and reflecting in x2, but applied to *primitives* in a custom operation
        // see boundaries.cpp
        pin->SetString("parthenon/mesh", "ix1_bc", "user");
        pin->SetString("parthenon/mesh", "ox1_bc", "user");
        pin->SetString("parthenon/mesh", "ix2_bc", "user");
        pin->SetString("parthenon/mesh", "ox2_bc", "user");
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
    } else {
        pin->SetBoolean("coordinates", "spherical", false);
    }

    // If we're using constant field of some kind, we likely *don't* want to normalize to beta_min=N
    std::string field_type = pin->GetOrAddString("b_field", "type", "none");
    if (field_type == "constant" || field_type == "monopole") {
        pin->GetOrAddBoolean("b_field", "norm", false);
    }
}

Packages_t KHARMA::ProcessPackages(std::unique_ptr<ParameterInput>& pin)
{
    // See above
    FixParameters(pin);

    // Then put together what we're supposed to
    Packages_t packages;

    // Read all options first so we can set their defaults here,
    // before any packages are initialized.
    std::string b_field_solver = pin->GetOrAddString("b_field", "solver", "flux_ct");
    bool do_wind = pin->GetOrAddBoolean("wind", "on", false);
    // TODO if jcon in outputs then...
    bool add_jcon = pin->GetOrAddBoolean("GRMHD", "add_jcon", true);
    bool do_electrons = pin->GetOrAddBoolean("electrons", "on", false);

    // Global variables "package."  Anything that just, really oughta be a global
    packages.Add(KHARMA::InitializeGlobals(pin.get()));

    // Most functions and variables are in the GRMHD package,
    // initialize it first among physics stuff
    packages.Add(GRMHD::Initialize(pin.get()));

    // B field solvers, to ensure divB == 0. 
    if (b_field_solver == "none") {
        // Don't add a B field
        // Currently this means fields are still allocated, and processing is done in GRMHD,
        // but no other operations are performed.
    } else if (b_field_solver == "constraint_damping" || b_field_solver == "b_cd") {
        // Constraint damping, probably only useful for non-GR MHD systems
        packages.Add(B_CD::Initialize(pin.get(), packages));
    } else {
        // Don't even error on bad values.  This is probably what you want,
        // and we'll check for adaptive and error later
        packages.Add(B_FluxCT::Initialize(pin.get(), packages));
    }

    if (do_wind) {
        packages.Add(Wind::Initialize(pin.get()));
    }

    if (add_jcon) {
        packages.Add(Current::Initialize(pin.get()));
    }

    if (do_electrons) {
        packages.Add(Electrons::Initialize(pin.get(), packages));
    }

    return std::move(packages);
}


// TODO decide on a consistent implementation of foreach packages -> do X
void KHARMA::FillDerivedDomain(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, int coarse)
{
    FLAG("Filling derived vars after applying physical boundaries");
    // We need to re-fill the "derived" (primitive) variables from everything
    // except GRMHD
    auto pmb = rc->GetBlockPointer();
    if (pmb->packages.AllPackages().count("B_CD"))
        B_CD::UtoP(rc.get(), domain, coarse);
    if (pmb->packages.AllPackages().count("B_FluxCT"))
        B_FluxCT::UtoP(rc.get(), domain, coarse);
    if (pmb->packages.AllPackages().count("Electrons"))
        Electrons::UtoP(rc.get(), domain, coarse);

    FLAG("Filled");
}

void KHARMA::PreStepMeshUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    if (!pmesh->packages.Get("Globals")->Param<bool>("in_loop")) {
        pmesh->packages.Get("Globals")->UpdateParam<bool>("in_loop", true);
    }
}

void KHARMA::PostStepMeshUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    // Knowing this works took a little digging into Parthenon's EvolutionDriver.
    // The order of operations is:
    // 1. Call PostStepUserWorkInLoop and PostStepDiagnostics (this function and following)
    // 2. Set the timestep tm.dt to the minimum from the EstimateTimestep calls
    // 3. Generate any outputs, e.g. jcon
    // Thus we preserve tm.dt (which has not yet been reset) as dt_last for Current::FillOutput
    pmesh->packages.Get("Globals")->UpdateParam<Real>("dt_last", tm.dt);

    // ctop_max has fewer rules. It's just convenient to set here since we're assured of no MPI hangs
    if (pmesh->packages.AllPackages().count("B_CD")) {
        Real ctop_max_last = MPIMax(pmesh->packages.Get("Globals")->Param<Real>("ctop_max"));
        pmesh->packages.Get("Globals")->UpdateParam<Real>("ctop_max_last", ctop_max_last);
        pmesh->packages.Get("Globals")->UpdateParam<Real>("ctop_max", 0.0);
    }
}

void KHARMA::PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    // Parthenon's version of this has a bug, but I would probably subclass it anyway.
    // very useful to have a single per-step spot for global prints
    for (auto &package : pmesh->packages.AllPackages()) {
        package.second->PostStepDiagnostics(tm, pmesh->mesh_data.GetOrAdd("base", 0).get());
    }
}

void KHARMA::FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // Don't fill the output arrays for the first dump, as trying to actually
    // calculate them can produce errors when we're not in the loop yet.
    // Instead, they just get added to the file as their starting values, i.e. 0
    if (pmb->packages.Get("Globals")->Param<bool>("in_loop")) {
        // TODO for package in packages with registered function...
        if (pmb->packages.AllPackages().count("Current"))
            Current::FillOutput(pmb, pin);
        if (pmb->packages.AllPackages().count("B_FluxCT"))
            B_FluxCT::FillOutput(pmb, pin);
        if (pmb->packages.AllPackages().count("B_CD"))
            B_CD::FillOutput(pmb, pin);
        if (pmb->packages.AllPackages().count("Electrons"))
            Electrons::FillOutput(pmb, pin);
    }
}

