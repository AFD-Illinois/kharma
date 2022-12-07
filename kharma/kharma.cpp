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
#include "b_cleanup.hpp"
#include "current.hpp"
#include "electrons.hpp"
#include "implicit.hpp"
#include "floors.hpp"
#include "grmhd.hpp"
#include "reductions.hpp"
#include "emhd.hpp"
#include "wind.hpp"

#include "bondi.hpp"
#include "boundaries.hpp"
#include "harm_driver.hpp"
#include "resize_restart.hpp"

std::shared_ptr<StateDescriptor> KHARMA::InitializeGlobals(ParameterInput *pin)
{
    Flag("Initializing Globals");
    // All global mutable state.  All of these and only these parameters are "mutable"
    auto pkg = std::make_shared<StateDescriptor>("Globals");
    Params &params = pkg->AllParams();
    // Current time in the simulation.  For ramping things up, ramping things down,
    // or preventing bad outcomes at known times
    params.Add("time", 0.0, true);
    // Last step's dt (Parthenon SimTime tm.dt), which must be preserved to output jcon
    params.Add("dt_last", 0.0, true);
    // Accumulator for maximum ctop within an MPI process
    // That is, this value does NOT generally reflect the actual maximum
    params.Add("ctop_max", 0.0, true);
    // Maximum between MPI processes, updated after each step; that is, always a maximum.
    params.Add("ctop_max_last", 0.0, true);
    // Whether we are computing initial outputs/timestep, or versions in the execution loop
    params.Add("in_loop", false, true);

    Flag("Initialized");
    return pkg;
}
void KHARMA::ResetGlobals(ParameterInput *pin, Mesh *pmesh)
{
    // The globals package was loaded & exists, retrieve it
    auto pkg = pmesh->packages.Get("Globals");
    Params &params = pkg->AllParams();
    // This needs to be reset to guarantee that EstimateTimestep doesn't try to
    // calculate a new dt from a blank 'ctop' variable,
    // just uses whatever the next step was going to be at reset
    params.Update("in_loop", false);

    // Everything else is a per-step variable, not per-run, so they're fine
    // to be restored by Parthenon
}

void KHARMA::FixParameters(std::unique_ptr<ParameterInput>& pin)
{
    Flag("Fixing parameters");
    // Parthenon sets 2 ghost zones as a default.
    // We can't override that default while allowing a file-specified value.
    // Fine for now because we crash with 2. (Flux CT)
    // TODO add under different name?  Better precedence/origin code?
    pin->SetInteger("parthenon/mesh", "nghost", 4);
    Globals::nghost = pin->GetInteger("parthenon/mesh", "nghost");
    // Warn if using less than 4 ghost zones in any circumstances, it's still not tested well
    // if (Globals::nghost < 4) {
    //     std::cerr << "WARNING: Using less than 4 ghost zones is untested!" << std::endl;
    // }

    // If we're restarting (not via Parthenon), read the restart file to get most parameters
    std::string prob = pin->GetString("parthenon/job", "problem_id");
    if (prob == "resize_restart") {
        ReadIharmRestartHeader(pin->GetString("resize_restart", "fname"), pin);
    }

    // Then handle coordinate systems and boundaries!
    std::string coordinate_base = pin->GetString("coordinates", "base");
    if (coordinate_base == "ks") coordinate_base = "spherical_ks";
    if (coordinate_base == "bl") coordinate_base = "spherical_bl";
    if (coordinate_base == "minkowski") coordinate_base = "cartesian_minkowski";
    std::string coordinate_transform = pin->GetOrAddString("coordinates", "transform", "null");
    if (coordinate_transform == "none") coordinate_transform = "null";
    if (coordinate_transform == "fmks") coordinate_transform = "funky";
    if (coordinate_transform == "mks") coordinate_transform = "modified";
    if (coordinate_transform == "exponential") coordinate_transform = "exp";
    if (coordinate_transform == "eks") coordinate_transform = "exp";
    // TODO any other synonyms
    if (coordinate_base == "spherical_ks" || coordinate_base == "spherical_bl" || coordinate_base == "spherical_minkowski") {
        pin->SetBoolean("coordinates", "spherical", true);
    } else {
        pin->SetBoolean("coordinates", "spherical", false);
    }

    // Spherical systems can specify r_out and optionally r_in,
    // instead of xNmin/max.
    // Other systems must specify x1min/max directly in the mesh region
    if (!pin->DoesParameterExist("parthenon/mesh", "x1min") ||
        !pin->DoesParameterExist("parthenon/mesh", "x1max")) {
        // TODO ask our coordinates about this rather than assuming exp()
        bool log_r = (coordinate_transform != "null");

        // Outer radius is always specified
        GReal Rout = pin->GetReal("coordinates", "r_out");
        GReal x1max = log_r ? log(Rout) : Rout;
        pin->GetOrAddReal("parthenon/mesh", "x1max", x1max);

        if (coordinate_base == "spherical_ks" || coordinate_base == "spherical_bl") {
            // Set inner radius if not specified
            if (pin->DoesParameterExist("coordinates", "r_in")) {
                GReal Rin = pin->GetReal("coordinates", "r_in");
                GReal x1min = log_r ? log(Rin) : Rin;
                pin->GetOrAddReal("parthenon/mesh", "x1min", x1min);
            } else {
                int nx1 = pin->GetInteger("parthenon/mesh", "nx1");
                Real a = pin->GetReal("coordinates", "a");
                GReal Rhor = 1 + sqrt(1 - a*a);
                GReal x1hor = log_r ? log(Rhor) : Rhor;

                // Set Rin such that we have 5 zones completely inside the event horizon
                // If xeh = log(Rhor), xin = log(Rin), and xout = log(Rout),
                // then we want xeh = xin + 5.5 * (xout - xin) / N1TOT:
                GReal x1min = (nx1 * x1hor / 5.5 - x1max) / (-1. + nx1 / 5.5);
                if (x1min < 0.0) {
                    throw std::invalid_argument("Not enough radial zones were specified to put 5 zones inside EH!");
                }
                pin->GetOrAddReal("parthenon/mesh", "x1min", x1min);
            }

            //cout << "Setting x1min: " << x1min << " x1max " << x1max << " based on BH with a=" << a << endl;

        } else if (coordinate_base == "spherical_minkowski") {
            // In Minkowski coordinates, require Rin so the singularity is at user option
            GReal Rin = pin->GetReal("coordinates", "r_in");
            GReal x1min = log_r ? log(Rin) : Rin;
            pin->GetOrAddReal("parthenon/mesh", "x1min", x1min);
        }
    }

    // Assumption: if we're in a spherical system...
    if (coordinate_base == "spherical_ks" || coordinate_base == "spherical_bl" || coordinate_base == "spherical_minkowski") {
        // ...then we definitely want KHARMA's spherical boundary conditions
        // These are inflow in x1 and reflecting in x2, but applied to *primitives* in
        // a custom operation, see boundaries.cpp
        pin->GetOrAddString("parthenon/mesh", "ix1_bc", "user");
        pin->GetOrAddString("parthenon/mesh", "ox1_bc", "user");
        pin->GetOrAddString("parthenon/mesh", "ix2_bc", "user");
        pin->GetOrAddString("parthenon/mesh", "ox2_bc", "user");
        pin->GetOrAddString("parthenon/mesh", "ix3_bc", "periodic");
        pin->GetOrAddString("parthenon/mesh", "ox3_bc", "periodic");

        // We also know the bounds for most transforms in spherical coords
        // Note we *only* set them here if they were not previously set/read!
        if (coordinate_transform == "null" || coordinate_transform == "exp") {
            pin->GetOrAddReal("parthenon/mesh", "x2min", 0.0);
            pin->GetOrAddReal("parthenon/mesh", "x2max", M_PI);
            pin->GetOrAddReal("parthenon/mesh", "x3min", 0.0);
            pin->GetOrAddReal("parthenon/mesh", "x3max", 2*M_PI);
        } else if (coordinate_transform == "modified" || coordinate_transform == "funky") {
            pin->GetOrAddReal("parthenon/mesh", "x2min", 0.0);
            pin->GetOrAddReal("parthenon/mesh", "x2max", 1.0);
            pin->GetOrAddReal("parthenon/mesh", "x3min", 0.0);
            pin->GetOrAddReal("parthenon/mesh", "x3max", 2*M_PI);
        } // TODO any other transforms/systems
    } else {
        // Most likely, Cartesian simulations will specify boundary conditions,
        // but we set defaults here.
        pin->GetOrAddString("parthenon/mesh", "ix1_bc", "periodic");
        pin->GetOrAddString("parthenon/mesh", "ox1_bc", "periodic");
        pin->GetOrAddString("parthenon/mesh", "ix2_bc", "periodic");
        pin->GetOrAddString("parthenon/mesh", "ox2_bc", "periodic");
        pin->GetOrAddString("parthenon/mesh", "ix3_bc", "periodic");
        pin->GetOrAddString("parthenon/mesh", "ox3_bc", "periodic");
        // Cartesian sims must specify the domain!
    }
    Flag("Fixed");
}

Packages_t KHARMA::ProcessPackages(std::unique_ptr<ParameterInput>& pin)
{
    // See above
    FixParameters(pin);

    Flag("Initializing packages");
    // Then put together what we're supposed to
    Packages_t packages;

    // Read all package enablements first so we can set their defaults here,
    // before any packages are initialized: thus they can know the full list
    std::string b_field_solver = pin->GetOrAddString("b_field", "solver", "flux_ct");

    // Enable b_cleanup package if we want it explicitly
    bool b_cleanup_package = pin->GetOrAddBoolean("b_cleanup", "on", false);
    // OR if we need it for resizing a dump
    bool is_resize = pin->GetString("parthenon/job", "problem_id") == "resize_restart";
    // OR if we want an initial cleanup pass for some other reason
    bool initial_cleanup = pin->GetOrAddBoolean("b_field", "initial_cleanup", false);
    // These were separated to make sure that the preference keys are initialized,
    // since short-circuiting prevented that when they were listed below
    bool b_cleanup = b_cleanup_package || is_resize || initial_cleanup;

    // TODO enable this iff jcon is in the list of outputs
    bool add_jcon = pin->GetOrAddBoolean("GRMHD", "add_jcon", true);
    bool do_electrons = pin->GetOrAddBoolean("electrons", "on", false);
    bool do_reductions = pin->GetOrAddBoolean("reductions", "on", true);
    bool do_emhd = pin->GetOrAddBoolean("emhd", "on", false);
    bool do_wind = pin->GetOrAddBoolean("wind", "on", false);

    // Set the default driver all the way up here, so packages know how to flag
    // prims vs cons (imex stepper syncs prims, but it's the packages' job to mark them)
    std::string driver_type;
    if (do_emhd) {
        // Default to implicit step for EMHD
        driver_type = pin->GetOrAddString("driver", "type", "imex");
    } else {
        driver_type = pin->GetOrAddString("driver", "type", "harm");
    }
    // Initialize the implicit timestepping package early so we can mark fields to be
    // updated implicitly vs explicitly
    if (driver_type == "imex") {
        packages.Add(Implicit::Initialize(pin.get()));
    }

    // Global variables "package."  Mutable global state Parthenon doesn't keep for us.
    // Always enable.
    packages.Add(KHARMA::InitializeGlobals(pin.get()));

    // Lots of common functions and variables are still in the GRMHD package,
    // always initialize it first among physics stuff
    packages.Add(GRMHD::Initialize(pin.get(), packages));

    // We'll also always want the floors package, even if floors are disabled
    packages.Add(Floors::Initialize(pin.get()));

    // B field solvers, to ensure divB == 0.
    if (b_field_solver == "none") {
        // Don't add a B field
    } else if (b_field_solver == "constraint_damping" || b_field_solver == "b_cd") {
        // Constraint damping, probably only useful for non-GR MHD systems
        packages.Add(B_CD::Initialize(pin.get(), packages));
    } else {
        // Don't even error on bad values.  This is probably what you want
        packages.Add(B_FluxCT::Initialize(pin.get(), packages));
    }
    // Additional cleanup on B field.
    // Can be enabled with or without a per-step solver, currently used for restart resizing
    if (b_cleanup) {
        packages.Add(B_Cleanup::Initialize(pin.get(), packages));
    }
    // Unless both a field solver and cleanup routine are disabled,
    // there is some form of B field present/declared.
    bool b_field_exists = !(b_field_solver == "none" && !b_cleanup);

    // Add jcon, so long as there's a field to calculate it from
    if (add_jcon && b_field_exists) {
        packages.Add(Current::Initialize(pin.get()));
    }

    // Electrons are boring but not impossible without a B field
    if (do_electrons) {
        packages.Add(Electrons::Initialize(pin.get(), packages));
    }

    if (do_reductions) {
        packages.Add(Reductions::Initialize(pin.get()));
    }

    if (do_emhd) {
        packages.Add(EMHD::Initialize(pin.get(), packages));
    }

    if (do_wind) {
        packages.Add(Wind::Initialize(pin.get()));
    }

    Flag("Finished initializing packages");
    return std::move(packages);
}


// TODO decide on a consistent implementation of foreach packages -> do X
void KHARMA::FillDerivedDomain(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, int coarse)
{
    Flag(rc.get(), "Filling derived variables on boundaries");
    // We need to re-fill the "derived" (primitive) variables on the physical boundaries,
    // since we already called "FillDerived" before the ghost zones were initialized
    // This does *not* apply to the GRMHD variables, as their primitive values are filled
    // during the boundary call
    auto pmb = rc->GetBlockPointer();
    // if (pmb->packages.AllPackages().count("GRMHD"))
    //     GRMHD::UtoP(rc.get(), domain, coarse);
    if (pmb->packages.AllPackages().count("B_FluxCT"))
        B_FluxCT::UtoP(rc.get(), domain, coarse);
    if (pmb->packages.AllPackages().count("B_CD"))
        B_CD::UtoP(rc.get(), domain, coarse);
    if (pmb->packages.AllPackages().count("Electrons"))
        Electrons::UtoP(rc.get(), domain, coarse);

    Flag(rc.get(), "Filled");
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
    // The order of operations after calling Step() is:
    // 1. Call PostStepUserWorkInLoop and PostStepDiagnostics (this function and following)
    // 2. Set the timestep tm.dt to the minimum from the EstimateTimestep calls
    // 3. Generate any outputs, e.g. jcon
    // Thus we preserve tm.dt (which has not yet been reset) as dt_last for Current::FillOutput
    pmesh->packages.Get("Globals")->UpdateParam<double>("dt_last", tm.dt);
    pmesh->packages.Get("Globals")->UpdateParam<double>("time", tm.time);

    // ctop_max has fewer rules. It's just convenient to set here since we're assured of no MPI hangs
    // Since it involves an MPI sync, we only keep track of this when we need it
    if (pmesh->packages.AllPackages().count("B_CD")) {
        static AllReduce<Real> ctop_max_last_r;
        ctop_max_last_r.val = pmesh->packages.Get("Globals")->Param<Real>("ctop_max");
        ctop_max_last_r.StartReduce(MPI_MAX);
        while (ctop_max_last_r.CheckReduce() == TaskStatus::incomplete);
        pmesh->packages.Get("Globals")->UpdateParam<Real>("ctop_max_last", ctop_max_last_r.val);
        pmesh->packages.Get("Globals")->UpdateParam<Real>("ctop_max", 0.0);
    }
}

void KHARMA::PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime &tm)
{
    // Parthenon's version of this has a bug, but I would probably subclass it anyway.
    // very useful to have a single per-step spot to control any routine print statements
    const auto& md = pmesh->mesh_data.GetOrAdd("base", 0).get();
    if (md->NumBlocks() > 0) {
        for (auto &package : pmesh->packages.AllPackages()) {
            package.second->PostStepDiagnostics(tm, md);
        }
    }
}

void KHARMA::FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    Flag("Filling output");
    // Rewrite this and the above as a callback registration
    if (pmb->packages.AllPackages().count("Current"))
        Current::FillOutput(pmb, pin);
    if (pmb->packages.AllPackages().count("B_FluxCT"))
        B_FluxCT::FillOutput(pmb, pin);
    if (pmb->packages.AllPackages().count("B_CD"))
        B_CD::FillOutput(pmb, pin);
    if (pmb->packages.AllPackages().count("Electrons"))
        Electrons::FillOutput(pmb, pin);
    Flag("Filled");
}

