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

#include <memory>

// Until Parthenon gets a reduce()
#include "Kokkos_Core.hpp"

#include <parthenon/parthenon.hpp>

#include "decs.hpp"

#include "boundaries.hpp"
#include "current.hpp"
#include "debug.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "grmhd_functions.hpp"
#include "U_to_P.hpp"

using namespace parthenon;


/**
 * GRMHD package.  Global operations on General Relativistic Magnetohydrodynamic systems.
 */
namespace GRMHD
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    // This function builds and returns a "StateDescriptor" or "Package" object.
    // The most important part of this object is a member of type "Params",
    // which acts more or less like a Python dictionary:
    // it puts values into a map of names->objects, where "objects" are usually
    // floats, strings, and ints, but can be arbitrary classes.
    // This "dictionary" is *not* totally immutable, but should be treated
    // as such in every package except "Globals".
    auto pkg = std::make_shared<StateDescriptor>("GRMHD");
    Params &params = pkg->AllParams();

    // =================================== PARAMETERS ===================================

    // Add the problem name, so we can be C++ noobs and special-case on string contents
    std::string problem_name = pin->GetString("parthenon/job", "problem_id");
    params.Add("problem", problem_name);

    // Fluid gamma for ideal EOS.  Don't guess this.
    // Only ideal EOS are supported, though modifying gamma based on
    // local temperatures would be straightforward.
    double gamma = pin->GetReal("GRMHD", "gamma");
    params.Add("gamma", gamma);

    // Proportion of courant condition for timesteps
    double cfl = pin->GetOrAddReal("GRMHD", "cfl", 0.9);
    params.Add("cfl", cfl);

    // Don't even error on this. LLF or bust, baby
    // TODO move this and recon options out of GRMHD package!
    std::string flux = pin->GetOrAddString("GRMHD", "flux", "llf");
    if (flux == "hlle") {
        params.Add("use_hlle", true);
    } else {
        params.Add("use_hlle", false);
    }

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

    // Reconstruction scheme: plm, weno5, ppm...
    std::string recon = pin->GetOrAddString("GRMHD", "reconstruction", "weno5");
    if (recon == "donor_cell") {
        params.Add("recon", ReconstructionType::donor_cell);
    } else if (recon == "linear_vl") {
        params.Add("recon", ReconstructionType::linear_vl);
    } else if (recon == "linear_mc") {
        params.Add("recon", ReconstructionType::linear_mc);
    } else if (recon == "weno5") {
        params.Add("recon", ReconstructionType::weno5);
    // } else if (recon == "weno5_lower_poles") {
    //     params.Add("recon", ReconstructionType::weno5_lower_poles);
    } else {
        std::cerr << "Reconstruction type not supported!  Supported reconstructions:" << std::endl;
        std::cerr << "donor_cell, linear_mc, linear_vl, weno5" << std::endl;
        throw std::invalid_argument("Unsupported reconstruction algorithm!");
    }

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // Option to disable checking the fluxes at boundaries:
    // Prevent inflow at outer boundaries
    bool check_inflow_inner = pin->GetOrAddBoolean("bounds", "check_inflow_inner", true);
    params.Add("check_inflow_inner", check_inflow_inner);
    bool check_inflow_outer = pin->GetOrAddBoolean("bounds", "check_inflow_outer", true);
    params.Add("check_inflow_outer", check_inflow_outer);
    // Ensure fluxes through the zero-size face at the pole are zero
    bool fix_flux_pole = pin->GetOrAddBoolean("bounds", "fix_flux_pole", true);
    params.Add("fix_flux_pole", fix_flux_pole);

    // Driver options
    // The two current drivers are "harm" or "imex", with the former being the usual KHARMA
    // driver, and the latter supporting implicit stepping of some or all variables
    auto driver_type = pin->GetString("driver", "type"); // This is set in kharma.cpp
    params.Add("driver_type", driver_type);
    // The ImEx driver is necessary to evolve implicitly, but doesn't require it.  Using explicit
    // updates for GRMHD vars is useful for testing, or if adding just a couple of implicit variables
    // Doing EGRMHD requires implicit evolution of GRMHD variables, of course
    auto implicit_grmhd = (driver_type == "imex") &&
                          (pin->GetBoolean("emhd", "on") || pin->GetOrAddBoolean("GRMHD", "implicit", false));
    params.Add("implicit", implicit_grmhd);
    // Synchronize boundary variables twice.  Ensures KHARMA is agnostic to the breakdown
    // of meshblocks, at the cost of twice the MPI overhead, for potentially much worse strong scaling.
    bool two_sync = pin->GetOrAddBoolean("perf", "two_sync", false) ||
                    pin->GetOrAddBoolean("driver", "two_sync", false);
    params.Add("two_sync", two_sync);

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
    // This is stretching what the "Params" object should really be carrying,
    // but the flag values are necessary in many places, and this was the
    // easiest way to ensure availability.
    // 1. One flag to mark the primitive variables specifically
    // (Parthenon has Metadata::Conserved already)
    MetadataFlag isPrimitive = Metadata::AllocateNewFlag("Primitive");
    params.Add("PrimitiveFlag", isPrimitive);
    // 2. And one for hydrodynamics (everything we directly handle in this package)
    MetadataFlag isHD = Metadata::AllocateNewFlag("HD");
    params.Add("HDFlag", isHD);
    // 3. And one for magnetohydrodynamics
    // (all HD fields plus B field, which we'll need to make use of)
    MetadataFlag isMHD = Metadata::AllocateNewFlag("MHD");
    params.Add("MHDFlag", isMHD);

    std::vector<MetadataFlag> flags_prim, flags_cons;
    if (driver_type == "harm") { // Normal operation
        // As mentioned elsewhere, KHARMA treats the conserved variables as the independent ones,
        // and the primitives as "Derived"
        // Primitives are still used for reconstruction, physical boundaries, and output, and are
        // generally the easier to understand quantities
        // Note especially their ghost zones are also filled! This is less efficient than syncing just
        // one or the other, but allows the most flexibility for reasons that should be clearer in harm_driver.cpp
        flags_prim = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Derived,
                                                Metadata::FillGhost, Metadata::Restart,
                                                isPrimitive, isHD, isMHD});
        // Conserved variables are actually rho*u^0 & T^0_mu, but are named after the prims for consistency
        // We will rarely need the conserved variables by name, we will mostly be treating them as a group
        flags_cons = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Independent,
                                                Metadata::WithFluxes, Metadata::FillGhost, Metadata::Restart,
                                                Metadata::Conserved, isHD, isMHD});
    } else if (driver_type == "imex") { // ImEx driver
        // When evolving (E)GRMHD implicitly, we instead mark the primitive variables to be synchronized.
        // This won't work for AMR, but it fits much better with the implicit solver, which expects
        // primitive variable inputs and produces primitive variable results.

        // Mark whether to evolve our variables via the explicit or implicit step inside the driver
        MetadataFlag areWeImplicit = (implicit_grmhd) ? packages.Get("Implicit")->Param<MetadataFlag>("ImplicitFlag")
                                                      : packages.Get("Implicit")->Param<MetadataFlag>("ExplicitFlag");

        flags_prim = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Derived, areWeImplicit,
                                                Metadata::FillGhost, Metadata::Restart, isPrimitive, isHD, isMHD});
        flags_cons = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Independent, areWeImplicit,
                                                Metadata::WithFluxes, Metadata::Conserved, isHD, isMHD});
    }

    // With the flags sorted & explained, actually declaring fields is easy.
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

    // No magnetic fields here. KHARMA should operate fine in GRHD without them,
    // so they are allocated only by B field packages.

    // Maximum signal speed (magnitude).
    // Needs to be cached from flux updates for calculating the timestep later
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
    pkg->AddField("ctop", m);

    // Flag denoting UtoP inversion failures
    // Only needed if we're actually calling UtoP, but always allocated as it's retrieved often
    // Needs boundary sync if treating primitive variables as fundamental
    if (driver_type == "imex" && !implicit_grmhd) {
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    } else {
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    }
    pkg->AddField("pflag", m);

    if (!implicit_grmhd) {
        // If we're using a step that requires calling UtoP, register it
        // Calling this messes up implicit stepping, so we only register it here
        pkg->FillDerivedBlock = GRMHD::FillDerivedBlock;
    }

    // Finally, the StateDescriptor/Package object determines the Callbacks Parthenon makes to
    // a particular package -- that is, some portion of the things that the package needs done
    // at each step, which must be done at specific times.
    // See the header files defining each of these functions for their purpose and call context.
    pkg->CheckRefinementBlock = GRMHD::CheckRefinement;
    pkg->EstimateTimestepBlock = GRMHD::EstimateTimestep;
    pkg->PostStepDiagnosticsMesh = GRMHD::PostStepDiagnostics;

    return pkg;
}

void UtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "Filling Primitives");
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    PackIndexMap prims_map, cons_map;
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    auto P = GRMHD::PackHDPrims(rc, prims_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    GridScalar pflag = rc->Get("pflag").data;

    // KHARMA uses only one boundary exchange, in the conserved variables
    // Except where FixUtoP has no neighbors, and must fix with bad zones, this is fully identical
    // between #s of MPI ranks, because we sync 4 ghost zones and only require 3 for reconstruction.
    // Thus as long as the last rank is not flagged, it will be inverted the same way on each process, and
    // used in the same way for fixups.  If it fails & thus might be different, it is ignored.

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    // Get the primitives from our conserved versions
    // Currently this returns *all* zones, including all ghosts, even
    // uninitialized zones which are still zero.  We select for initialized
    // zones only in the loop below, to avoid failures to converge while
    // calculating primtive vars over as much of the domain as possible
    // We could (did formerly) save some time here by running over
    // only zones with initialized conserved variables, but the domain
    // of such values is not rectangular in the current handling
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);
    const IndexRange ib_b = bounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jb_b = bounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kb_b = bounds.GetBoundsK(IndexDomain::interior);

    pmb->par_for("U_to_P", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            if (inside(k, j, i, kb_b, jb_b, ib_b) ||
                m::abs(P(m_p.RHO, k, j, i)) > SMALL || m::abs(P(m_p.UU, k, j, i)) > SMALL) {
                // Run over all interior zones and any initialized ghosts
                pflag(k, j, i) = GRMHD::u_to_p(G, U, m_u, gam, k, j, i, Loci::center, P, m_p);
            } else {
                // Don't *use* un-initialized zones for fixes, but also don't *fix* them
                pflag(k, j, i) = -1;
            }
        }
    );
    Flag(rc, "Filled");
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
    // TODO: move diagnostic printing to PostStepDiagnostics, now it's broken here

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
            double ndt_zone = 1 / (1 / (G.dx1v(i) / ctop(0, k, j, i)) +
                                   1 / (G.dx2v(j) / ctop(1, k, j, i)) +
                                   1 / (G.dx3v(k) / ctop(2, k, j, i)));
            // Effective "max speed" used for the timestep
            double ctop_max_zone = m::min(G.dx1v(i), m::min(G.dx2v(j), G.dx3v(k))) / ndt_zone;

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
    if (nctop > globals.Get<Real>("ctop_max")) {
        globals.Update<Real>("ctop_max", nctop);
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

    const Real dx[GR_DIM] = {0., G.dx1v(0), G.dx2v(0), G.dx3v(0)};

    // Leaving minmax in case the max phase speed is useful
    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb->par_reduce("ndt_min", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                      typename Kokkos::MinMax<Real>::value_type &lminmax) {

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
    const auto &refine_tol = pkg->Param<Real>("refine_tol");
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
    const auto& pars = pmesh->packages.Get("GRMHD")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");
    const int extra_checks = pars.Get<int>("extra_checks");

    // Debugging/diagnostic info about floor and inversion flags
    if (flag_verbose >= 1) {
        Flag("Printing flags");
        CountPFlags(md, IndexDomain::interior, flag_verbose);
        CountFFlags(md, IndexDomain::interior, flag_verbose);
    }

    // Check for a soundspeed (ctop) of 0 or NaN
    // This functions as a "last resort" check to stop a
    // simulation on obviously bad data
    // TODO also be able to print what zone dictated timestep
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
