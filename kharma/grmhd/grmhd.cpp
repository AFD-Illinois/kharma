/* 
 *  File: grmhd.cpp
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

#include "grmhd.hpp"

#include "decs.hpp"

// TODO eliminate when Parthenon gets reduction types
#include "Kokkos_Core.hpp"

#include "boundaries.hpp"
#include "current.hpp"
#include "debug.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "grmhd_functions.hpp"
#include "kharma.hpp"

#include <memory>


/**
 * GRMHD package.  Global operations on General Relativistic Magnetohydrodynamic systems.
 */
namespace GRMHD
{

std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    Flag("Initializing GRMHD");
    // This function builds and returns a "KHARMAPackage" object, which is a light
    // superset of Parthenon's "StateDescriptor" class for packages.
    // The most important part of this object is a member of type "Params",
    // which acts more or less like a Python dictionary:
    // it puts values into a map of names->objects, where "objects" are usually
    // floats, strings, and ints, but can be arbitrary classes.
    // This "dictionary" is mostly immutable, and should always be treated as immutable,
    // except in the "Globals" package.
    auto pkg = std::make_shared<KHARMAPackage>("GRMHD");
    Params &params = pkg->AllParams();

    // GRMHD PARAMETERS
    // Fluid gamma for ideal EOS.  Don't guess this.
    // Only ideal EOS are supported, though modifying gamma based on
    // local temperatures would be straightforward.
    double gamma = pin->GetReal("GRMHD", "gamma");
    params.Add("gamma", gamma);

    // Proportion of courant condition for timesteps
    double cfl = pin->GetOrAddReal("GRMHD", "cfl", 0.9);
    params.Add("cfl", cfl);

    // TIME PARAMETERS
    // These parameters are put in "parthenon/time" to match others, but ultimately we should
    // override the parthenon timestep chooser
    // Minimum timestep, if something about the sound speed goes wonky. Probably won't save you :)
    double dt_min = pin->GetOrAddReal("parthenon/time", "dt_min", 1.e-5);
    params.Add("dt_min", dt_min);
    // Starting timestep: guaranteed step 1 timestep returned by EstimateTimestep,
    // usually matters most for restarts
    double dt_start = pin->GetOrAddReal("parthenon/time", "dt", dt_min);
    params.Add("dt_start", dt_start);
    double max_dt_increase = pin->GetOrAddReal("parthenon/time", "max_dt_increase", 2.0);
    params.Add("max_dt_increase", max_dt_increase);

    // Alternatively, you can start with (or just always use) the light (phase) speed crossing time
    // of the smallest zone.  Useful when you're not sure of/modeling the characteristic velocities
    bool start_dt_light = pin->GetOrAddBoolean("parthenon/time", "start_dt_light", false);
    params.Add("start_dt_light", start_dt_light);
    bool use_dt_light = pin->GetOrAddBoolean("parthenon/time", "use_dt_light", false);
    params.Add("use_dt_light", use_dt_light);
    bool use_dt_light_phase_speed = pin->GetOrAddBoolean("parthenon/time", "use_dt_light_phase_speed", false);
    params.Add("use_dt_light_phase_speed", use_dt_light_phase_speed);

    // IMPLICIT PARAMETERS
    // The ImEx driver is necessary to evolve implicitly, but doesn't require it.  Using explicit
    // updates for GRMHD vars is useful for testing, or if adding just a couple of implicit variables
    // Doing EGRMHD requires implicit evolution of GRMHD variables, of course
    auto& driver = packages->Get("Driver")->AllParams();
    auto implicit_grmhd = (driver.Get<std::string>("type") == "imex") &&
                          (pin->GetBoolean("emhd", "on") || pin->GetOrAddBoolean("GRMHD", "implicit", false));
    params.Add("implicit", implicit_grmhd);

    // Update variable numbers
    if (implicit_grmhd) {
        int n_current = driver.Get<int>("n_implicit_vars");
        driver.Update("n_implicit_vars", n_current+5);
    } else {
        int n_current = driver.Get<int>("n_explicit_vars");
        driver.Update("n_explicit_vars", n_current+5);
    }

    // AMR PARAMETERS
    // Adaptive mesh refinement options
    // Only active if "refinement" and "numlevel" parameters allow
    Real refine_tol = pin->GetOrAddReal("GRMHD", "refine_tol", 0.5);
    params.Add("refine_tol", refine_tol);
    Real derefine_tol = pin->GetOrAddReal("GRMHD", "derefine_tol", 0.05);
    params.Add("derefine_tol", derefine_tol);

    // =================================== FIELDS ===================================

    // In addition to "params", the StateDescriptor/Package object carries "Fields"
    // These represent any variables we want to keep track of across the grid, and
    // generally inherit the size of the MeshBlock (for "Cell" fields) or some
    // closely-related size (for "Face" and "Edge" fields)

    // Add flags to distinguish groups of fields.
    // 1. One flag to mark the primitive variables specifically
    // (Parthenon has Metadata::Conserved already)
    Metadata::AddUserFlag("Primitive");
    // 2. And one for hydrodynamics (everything we directly handle in this package)
    Metadata::AddUserFlag("HD");
    // 3. And one for magnetohydrodynamics
    // (all HD fields plus B field, which we'll need to make use of)
    Metadata::AddUserFlag("MHD");
    // Mark whether to evolve our variables via the explicit or implicit step inside the driver
    MetadataFlag areWeImplicit = (implicit_grmhd) ? Metadata::GetUserFlag("Implicit")
                                                  : Metadata::GetUserFlag("Explicit");

    std::vector<MetadataFlag> flags_prim = {Metadata::Real, Metadata::Cell, Metadata::Derived, areWeImplicit,
                                            Metadata::Restart, Metadata::GetUserFlag("Primitive"), Metadata::GetUserFlag("HD"), Metadata::GetUserFlag("MHD")};
    std::vector<MetadataFlag> flags_cons = {Metadata::Real, Metadata::Cell, Metadata::Independent, areWeImplicit,
                                            Metadata::WithFluxes, Metadata::Conserved, Metadata::GetUserFlag("HD"), Metadata::GetUserFlag("MHD")};

    bool sync_prims = packages->Get("Driver")->Param<bool>("sync_prims");
    if (!sync_prims) { // Normal operation
        // As mentioned elsewhere, KHARMA treats the conserved variables as the independent ones,
        // and the primitives as "Derived"
        // Primitives are still used for reconstruction, physical boundaries, and output, and are
        // generally the easier to understand quantities
        // TODO can we not sync prims if we're using two_sync?
        flags_cons.push_back(Metadata::FillGhost);
        flags_prim.push_back(Metadata::FillGhost);
    } else { // Treat primitive vars as fundamental
        // When evolving (E)GRMHD implicitly, we just mark the primitive variables to be synchronized.
        // This won't work for AMR, but it fits much better with the implicit solver, which expects
        // primitive variable inputs and produces primitive variable results.
        flags_prim.push_back(Metadata::FillGhost);
    }

    // With the flags sorted & explained, actually declaring fields is easy.
    // These will be initialized & cleaned automatically for each meshblock
    auto m = Metadata(flags_prim);
    pkg->AddField("prims.rho", m);
    pkg->AddField("prims.u", m);
    // We add the "Vector" flag and a size to vectors
    auto flags_prim_vec(flags_prim);
    flags_prim_vec.push_back(Metadata::Vector);
    std::vector<int> s_vector({NVEC});
    m = Metadata(flags_prim_vec, s_vector);
    pkg->AddField("prims.uvec", m);

    m = Metadata(flags_cons);
    pkg->AddField("cons.rho", m);
    pkg->AddField("cons.u", m);
    auto flags_cons_vec(flags_cons);
    flags_cons_vec.push_back(Metadata::Vector);
    m = Metadata(flags_cons_vec, s_vector);
    pkg->AddField("cons.uvec", m);

    // Maximum signal speed (magnitude).
    // Needs to be cached from flux updates for calculating the timestep later
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
    pkg->AddField("ctop", m);

    // No magnetic fields here. KHARMA should operate fine in GRHD without them,
    // so they are allocated only by B field packages.

    // A KHARMAPackage also contains quite a few "callbacks," or functions called at
    // specific points in a step if the package is loaded.
    // Generally, see the headers for function descriptions.

    //pkg->BlockUtoP // Taken care of by the inverter package since it's hard to do
    // There's no "Flux" package, so we register the geometric (\Gamma*T) source here. I think it makes sense.
    pkg->AddSource = Flux::AddGeoSource;

    // Parthenon general callbacks
    pkg->CheckRefinementBlock    = GRMHD::CheckRefinement;
    pkg->EstimateTimestepBlock   = GRMHD::EstimateTimestep;
    pkg->PostStepDiagnosticsMesh = GRMHD::PostStepDiagnostics;

    // TODO TODO Reductions

    Flag("Initialized");
    return pkg;
}

Real EstimateTimestep(MeshBlockData<Real> *rc)
{
    Flag(rc, "Estimating timestep");
    auto pmb = rc->GetBlockPointer();
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto& G = pmb->coords;
    auto& ctop = rc->Get("ctop").data;

    // TODO: move timestep limiter into an override of SetGlobalTimestep
    // TODO: keep location of the max, or be able to look it up in diagnostics

    auto& globals = pmb->packages.Get("Globals")->AllParams();
    const auto& grmhd_pars = pmb->packages.Get("GRMHD")->AllParams();

    if (!globals.Get<bool>("in_loop")) {
        if (grmhd_pars.Get<bool>("start_dt_light") ||
            grmhd_pars.Get<bool>("use_dt_light")) {
            // Estimate based on light crossing time
            double dt = EstimateRadiativeTimestep(rc);
            // This records a per-rank minimum,
            // but Parthenon finds the global minimum anyway
            if (globals.hasKey("dt_light")) {
                if (dt < globals.Get<double>("dt_light"))
                    globals.Update<double>("dt_light", dt);
            } else {
                globals.Add<double>("dt_light", dt);
            }
            return dt;
        } else {
            // Or Just take from parameters
            double dt = grmhd_pars.Get<double>("dt_start");
            // Record this, since we'll use it to determine the max step increase
            globals.Update<double>("dt_last", dt);
            return dt;
        }
    }
    // If we're still using the light crossing time, skip the rest
    if (grmhd_pars.Get<bool>("use_dt_light")) {
        return globals.Get<double>("dt_light");
    }

    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb->par_reduce("ndt_min", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                      typename Kokkos::MinMax<Real>::value_type &lminmax) {
            double ndt_zone = 1 / (1 / (G.Dxc<1>(i) / ctop(0, k, j, i)) +
                                   1 / (G.Dxc<2>(j) / ctop(1, k, j, i)) +
                                   1 / (G.Dxc<3>(k) / ctop(2, k, j, i)));
            // Effective "max speed" used for the timestep
            double ctop_max_zone = m::min(G.Dxc<1>(i), m::min(G.Dxc<2>(j), G.Dxc<3>(k))) / ndt_zone;

            if (!m::isnan(ndt_zone) && (ndt_zone < lminmax.min_val))
                lminmax.min_val = ndt_zone;
            if (!m::isnan(ctop_max_zone) && (ctop_max_zone > lminmax.max_val))
                lminmax.max_val = ctop_max_zone;
        }
    , Kokkos::MinMax<Real>(minmax));
    // Keep dt to do some checks below
    const double min_ndt = minmax.min_val;
    const double nctop = minmax.max_val;

    // Apply limits
    const double cfl = grmhd_pars.Get<double>("cfl");
    const double dt_min = grmhd_pars.Get<double>("dt_min");
    const double dt_last = globals.Get<double>("dt_last");
    const double dt_max = grmhd_pars.Get<double>("max_dt_increase") * dt_last;
    const double ndt = clip(min_ndt * cfl, dt_min, dt_max);

    // Record max ctop, for constraint damping
    // TODO could probably use generic Max inside B_CD package
    if (pmb->packages.AllPackages().count("B_CD")) {
        auto& b_cd_params = pmb->packages.Get("B_CD")->AllParams();
        if (nctop > b_cd_params.Get<Real>("ctop_max"))
            b_cd_params.Update<Real>("ctop_max", nctop);
    }

    Flag(rc, "Estimated");
    return ndt;
}

Real EstimateRadiativeTimestep(MeshBlockData<Real> *rc)
{
    Flag(rc, "Estimating shortest light crossing time");
    auto pmb = rc->GetBlockPointer();
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto& G = pmb->coords;

    const auto& grmhd_pars = pmb->packages.Get("GRMHD")->AllParams();
    const bool phase_speed = grmhd_pars.Get<bool>("use_dt_light_phase_speed");

    const Real dx[GR_DIM] = {0., G.Dxc<1>(0), G.Dxc<2>(0), G.Dxc<3>(0)};

    // Leaving minmax in case the max phase speed is useful
    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb->par_reduce("ndt_min", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int& k, const int& j, const int& i,
                      typename Kokkos::MinMax<Real>::value_type& lminmax) {

            double light_phase_speed = SMALL;
            double dt_light_local = 0.;

            if (phase_speed) {
                for (int mu = 1; mu < GR_DIM; mu++) {
                    if(m::pow(G.gcon(Loci::center, j, i, 0, mu), 2) -
                        G.gcon(Loci::center, j, i, mu, mu)*G.gcon(Loci::center, j, i, 0, 0) >= 0.) {

                        double cplus = m::abs((-G.gcon(Loci::center, j, i, 0, mu) +
                                            m::sqrt(m::pow(G.gcon(Loci::center, j, i, 0, mu), 2) -
                                                G.gcon(Loci::center, j, i, mu, mu)*G.gcon(Loci::center, j, i, 0, 0)))/
                                            G.gcon(Loci::center, j, i, 0, 0));

                        double cminus = m::abs((-G.gcon(Loci::center, j, i, 0, mu) -
                                            m::sqrt(m::pow(G.gcon(Loci::center, j, i, 0, mu), 2) -
                                                G.gcon(Loci::center, j, i, mu, mu)*G.gcon(Loci::center, j, i, 0, 0)))/
                                            G.gcon(Loci::center, j, i, 0, 0));

                        light_phase_speed = m::max(cplus,cminus);
                    } else {
                        light_phase_speed = SMALL;
                    }

                    dt_light_local += 1./(dx[mu]/light_phase_speed);
                }
            } else {
                for (int mu = 1; mu < GR_DIM; mu++)
                    dt_light_local += 1./dx[mu];
            }
            dt_light_local = 1/dt_light_local;

            if (!m::isnan(dt_light_local) && (dt_light_local < lminmax.min_val))
                lminmax.min_val = dt_light_local;
            if (!m::isnan(light_phase_speed) && (light_phase_speed > lminmax.max_val))
                lminmax.max_val = light_phase_speed;
        }
    , Kokkos::MinMax<Real>(minmax));

    // Just spit out dt
    const double cfl = grmhd_pars.Get<double>("cfl");
    const double ndt = minmax.min_val * cfl;

    Flag(rc, "Estimated");
    return ndt;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    auto v = rc->Get("prims.rho").data;

    IndexDomain domain = IndexDomain::interior;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb->par_reduce("check_refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                      typename Kokkos::MinMax<Real>::value_type &lminmax) {
        lminmax.min_val =
            v(k, j, i) < lminmax.min_val ? v(k, j, i) : lminmax.min_val;
        lminmax.max_val =
            v(k, j, i) > lminmax.max_val ? v(k, j, i) : lminmax.max_val;
        }
    , Kokkos::MinMax<Real>(minmax));

    auto pkg = pmb->packages.Get("GRMHD");
    const auto &refine_tol   = pkg->Param<Real>("refine_tol");
    const auto &derefine_tol = pkg->Param<Real>("derefine_tol");

    if (minmax.max_val - minmax.min_val > refine_tol) return AmrTag::refine;
    if (minmax.max_val - minmax.min_val < derefine_tol) return AmrTag::derefine;
    return AmrTag::same;
}

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    Flag("Printing GRMHD diagnostics");
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int extra_checks = pars.Get<int>("extra_checks");
    Flag("Got pointers");

    // Check for a soundspeed (ctop) of 0 or NaN
    // This functions as a "last resort" check to stop a
    // simulation on obviously bad data
    if (extra_checks >= 1) {
        CheckNaN(md, X1DIR);
        if (pmesh->ndim > 1) CheckNaN(md, X2DIR);
        if (pmesh->ndim > 2) CheckNaN(md, X3DIR);
    }

    // Further checking for any negative values.  Floors should
    // prevent this, so we save it for dire debugging
    if (extra_checks >= 2) {
        Flag("Printing negative zones");
        CheckNegative(md, IndexDomain::interior);
    }

    Flag("Printed");
    return TaskStatus::complete;
}

} // namespace GRMHD
