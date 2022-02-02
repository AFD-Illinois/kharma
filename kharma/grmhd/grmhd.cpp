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

/**
 * GRMHD package.  Manipulations on GRMHD 
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
#include "fixup.hpp"
#include "floors.hpp"
#include "fluxes.hpp"
#include "gr_coordinates.hpp"
#include "kharma.hpp"
#include "mhd_functions.hpp"
#include "source.hpp"
#include "U_to_P.hpp"

using namespace parthenon;
// Need to access these directly for reductions
using namespace Kokkos;

namespace GRMHD
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin)
{
    // This function builds and returns a "StateDescriptor" or "Package" object.
    // The most important part of this object is a member of type "Params",
    // which acts more or less like a Python dictionary:
    // it puts values into a map of names->objects, where "objects" are usually
    // floats, strings, and ints, but can be arbitrary classes.
    // This "dictionary" is *not* immutable, but should be treated as such
    // in every package except "Globals".
    auto pkg = std::make_shared<StateDescriptor>("GRMHD");
    Params &params = pkg->AllParams();

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

    // Minimum timestep, if something about the sound speed goes wonky. Probably won't save you :)
    // know what we're doing modifying "parthenon/time" -- subclass 
    double dt_min = pin->GetOrAddReal("parthenon/time", "dt_min", 1.e-4);
    params.Add("dt_min", dt_min);
    // Starting timestep, in case we're restarting
    double dt_start = pin->GetOrAddReal("parthenon/time", "dt", dt_min);
    params.Add("dt_start", dt_start);
    double max_dt_increase = pin->GetOrAddReal("parthenon/time", "max_dt_increase", 2.0);
    params.Add("max_dt_increase", max_dt_increase);

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
        cerr << "Reconstruction type not supported!  Supported reconstructions:" << endl;
        cerr << "donor_cell, linear_mc, linear_vl, weno5" << endl;
        throw std::invalid_argument("Unsupported reconstruction algorithm!");
    }

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // Floor parameters
    // Floors for GRMHD quantities are handled in this package, since they
    // have a tendency to interact with fixing (& causing!) UtoP failures,
    // so we want to have control of the order in which floors/fixUtoP are
    // applied.
    double rho_min_geom, u_min_geom;
    if (pin->GetBoolean("coordinates", "spherical")) {
        // In spherical systems, floors drop as r^2, so set them higher by default
        rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-6);
        u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1.e-8);
    } else {
        rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-8);
        u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1.e-10);
    }
    params.Add("rho_min_geom", rho_min_geom);
    params.Add("u_min_geom", u_min_geom);
    double floor_r_char = pin->GetOrAddReal("floors", "r_char", 10);
    params.Add("floor_r_char", floor_r_char);

    double bsq_over_rho_max = pin->GetOrAddReal("floors", "bsq_over_rho_max", 1e20);
    params.Add("bsq_over_rho_max", bsq_over_rho_max);
    double bsq_over_u_max = pin->GetOrAddReal("floors", "bsq_over_u_max", 1e20);
    params.Add("bsq_over_u_max", bsq_over_u_max);
    double u_over_rho_max = pin->GetOrAddReal("floors", "u_over_rho_max", 1e20);
    params.Add("u_over_rho_max", u_over_rho_max);
    double ktot_max = pin->GetOrAddReal("floors", "ktot_max", 1e20);
    params.Add("ktot_max", ktot_max);

    double gamma_max = pin->GetOrAddReal("floors", "gamma_max", 50.);
    params.Add("gamma_max", gamma_max);

    bool temp_adjust_u = pin->GetOrAddBoolean("floors", "temp_adjust_u", false);
    params.Add("temp_adjust_u", temp_adjust_u);
    bool fluid_frame = pin->GetOrAddBoolean("floors", "fluid_frame", false);
    params.Add("fluid_frame", fluid_frame);
    bool adjust_k = pin->GetOrAddBoolean("floors", "adjust_k", true);
    params.Add("adjust_k", adjust_k);

    // Disable all floors.  It is obviously tremendously inadvisable to
    // set this option to true
    bool disable_floors = pin->GetOrAddBoolean("floors", "disable_floors", false);
    params.Add("disable_floors", disable_floors);

    // Option to disable checking the fluxes at boundaries
    bool check_inflow_inner = pin->GetOrAddBoolean("bounds", "check_inflow_inner", true);
    params.Add("check_inflow_inner", check_inflow_inner);
    bool check_inflow_outer = pin->GetOrAddBoolean("bounds", "check_inflow_outer", true);
    params.Add("check_inflow_outer", check_inflow_outer);
    bool fix_flux_pole = pin->GetOrAddBoolean("bounds", "fix_flux_pole", true);
    params.Add("fix_flux_pole", fix_flux_pole);


    // Performance options
    // Packed communications kernels, exchanging all boundary buffers of an MPI process
    // together.  Useful if # MeshBlocks is > # MPI ranks
    bool pack_comms = pin->GetOrAddBoolean("perf", "pack_comms", true);
    params.Add("pack_comms", pack_comms);

    // Adaptive mesh refinement options
    // Only active if "refinement" and "numlevel" parameters allow
    Real refine_tol = pin->GetOrAddReal("GRMHD", "refine_tol", 0.5);
    params.Add("refine_tol", refine_tol);
    Real derefine_tol = pin->GetOrAddReal("GRMHD", "derefine_tol", 0.05);
    params.Add("derefine_tol", derefine_tol);

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

    // In addition to "params", the StateDescriptor/Package object carries "Fields"
    // These represent any variables we want to keep track of across the grid, and
    // generally inherit the size of the MeshBlock (for "Cell" fields) or some
    // closely-related size (for "Face" and "Edge" fields)

    // As mentioned elsewhere, KHARMA treats the conserved variables as the independent ones,
    // and the primitives as "Derived"
    // Primitives are still used for reconstruction, physical boundaries, and output, and are
    // generally the easier to understand quantities
    std::vector<int> s_vector({3});
    auto flags_prim = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Derived,
                                                      Metadata::Restart, isPrimitive, isHD, isMHD});
    auto m = Metadata(flags_prim);
    pkg->AddField("prims.rho", m);
    pkg->AddField("prims.u", m);
    auto flags_prim_vec(flags_prim);
    flags_prim_vec.push_back(Metadata::Vector);
    m = Metadata(flags_prim_vec, s_vector);
    pkg->AddField("prims.uvec", m);

    // Conserved variables are actualy rho*u^0 & T^0_mu, but are named after the prims for consistency
    // We will rarely need the conserved variables by name, we will mostly be treating them as a group
    auto flags_cons = std::vector<MetadataFlag>({Metadata::Real, Metadata::Cell, Metadata::Independent,
                                                      Metadata::WithFluxes, Metadata::FillGhost, Metadata::Restart,
                                                      Metadata::Conserved, isHD, isMHD});
    m = Metadata(flags_cons);
    pkg->AddField("cons.rho", m);
    pkg->AddField("cons.u", m);
    auto flags_cons_vec(flags_cons);
    flags_cons_vec.push_back(Metadata::Vector);
    m = Metadata(flags_cons_vec, s_vector);
    pkg->AddField("cons.uvec", m);

    bool use_b = (pin->GetString("b_field", "solver") != "none");
    params.Add("use_b", use_b);
    if (!use_b) {
        // Declare placeholder fields only if not using another package providing B field.
        // This should be redundant w/using the "Overridable" flag but has caused problems in the past.
        // The ultimate goal is to support never defining these fields in the first place, i.e. true GRHD
        // without memory or computation penalties.

        // Remove the "HD" flag from B, since it is not that
        flags_prim_vec.erase(std::remove(flags_prim_vec.begin(), flags_prim_vec.end(), isHD), flags_prim_vec.end());
        // Remove the "Restart" flag, since unlike the fluid prims, prims.B is fully redundant
        flags_prim_vec.erase(std::remove(flags_prim_vec.begin(), flags_prim_vec.end(), Metadata::Restart), flags_prim_vec.end());
        flags_prim_vec.push_back(Metadata::Overridable);
        m = Metadata(flags_prim_vec, s_vector);
        pkg->AddField("prims.B", m);
        flags_cons_vec.erase(std::remove(flags_cons_vec.begin(), flags_cons_vec.end(), isHD), flags_cons_vec.end());
        flags_cons_vec.push_back(Metadata::Overridable);
        m = Metadata(flags_cons_vec, s_vector);
        pkg->AddField("cons.B", m);
    }

    // Maximum signal speed (magnitude).
    // Needs to be cached from flux updates for calculating the timestep later
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_vector);
    pkg->AddField("ctop", m);

    // Temporary fix just for being able to save field values
    // Should switch these to "Integer" fields when Parthenon supports it
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("pflag", m);
    pkg->AddField("fflag", m);

    // Finally, the StateDescriptor/Package object determines the Callbacks Parthenon makes to
    // a particular package -- that is, some portion of the things that the package needs done
    // at each step, which must be done at specific times.
    // See the documentation on each of these functions for their purpose and call context.
    pkg->FillDerivedBlock = GRMHD::UtoP;
    pkg->PostFillDerivedBlock = GRMHD::PostUtoP;
    pkg->CheckRefinementBlock = GRMHD::CheckRefinement;
    pkg->EstimateTimestepBlock = GRMHD::EstimateTimestep;
    pkg->PostStepDiagnosticsMesh = GRMHD::PostStepDiagnostics;

    return pkg;
}

void UtoP(MeshBlockData<Real> *rc)
{
    Flag(rc, "Filling Primitives");
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

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
    const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);
    pmb->par_for("U_to_P", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            if (P(m_p.RHO, k, j, i) != 0. || P(m_p.UU, k, j, i) != 0.) {
                pflag(k, j, i) = GRMHD::u_to_p(G, U, m_u, gam, k, j, i, Loci::center, P, m_p);
            } else {
                // Don't *use* un-initialized zones for fixes, but also don't *fix* them
                pflag(k, j, i) = -1;
            }
        }
    );
    Flag(rc, "Filled");
}

void PostUtoP(MeshBlockData<Real> *rc)
{
    Flag(rc, "Fixing Derived");

    // Apply floors
    if (!rc->GetBlockPointer()->packages.Get("GRMHD")->Param<bool>("disable_floors")) {
        GRMHD::ApplyFloors(rc);
    }

    // Fix inversion errors computing P from U, by averaging adjacent zones
    GRMHD::FixUtoP(rc);

    Flag(rc, "Fixed");
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

    if (!pmb->packages.Get("Globals")->Param<bool>("in_loop")) {
        double dt = pmb->packages.Get("GRMHD")->Param<double>("dt_start");
        // Record this, since we'll use it to determine the max step next
        pmb->packages.Get("Globals")->UpdateParam<double>("dt_last", dt);
        return dt;
    }

    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb->par_reduce("ndt_min", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                      typename Kokkos::MinMax<Real>::value_type &lminmax) {
            double ndt_zone = 1 / (1 / (G.dx1v(i) / ctop(0, k, j, i)) +
                                   1 / (G.dx2v(j) / ctop(1, k, j, i)) +
                                   1 / (G.dx3v(k) / ctop(2, k, j, i)));
            // Effective "max speed" used for the timestep
            double ctop_max_zone = min(G.dx1v(i), min(G.dx2v(j), G.dx3v(k))) / ndt_zone;

            if (!isnan(ndt_zone) && (ndt_zone < lminmax.min_val))
                lminmax.min_val = ndt_zone;
            if (!isnan(ctop_max_zone) && (ctop_max_zone > lminmax.max_val))
                lminmax.max_val = ctop_max_zone;
        }
    , Kokkos::MinMax<Real>(minmax));
    // Keep dt to do some checks below
    const double min_ndt = minmax.min_val;
    const double nctop = minmax.max_val;

    // Apply limits
    const double cfl = pmb->packages.Get("GRMHD")->Param<double>("cfl");
    const double dt_min = pmb->packages.Get("GRMHD")->Param<double>("dt_min");
    const double dt_last = pmb->packages.Get("Globals")->Param<double>("dt_last");
    const double dt_max = pmb->packages.Get("GRMHD")->Param<double>("max_dt_increase") * dt_last;
    const double ndt = clip(min_ndt * cfl, dt_min, dt_max);

    // Record max ctop, for constraint damping
    if (nctop > pmb->packages.Get("Globals")->Param<Real>("ctop_max")) {
        pmb->packages.Get("Globals")->UpdateParam<Real>("ctop_max", nctop);
    }

    Flag(rc, "Estimated");
    return ndt;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    auto v = rc->Get("prims.rho").data;

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

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
