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
    double dt_min = pin->GetOrAddReal("parthenon/time", "dt_min", 1.e-4);
    params.Add("dt_min", dt_min);
    // Starting timestep.  Important for consistent restarts, otherwise just the minimum
    double dt = pin->GetOrAddReal("parthenon/time", "dt", dt_min);
    params.Add("dt", dt);
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
        throw std::invalid_argument(string_format("Unsupported reconstruction algorithm %s!", recon));
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
    bool disable_floors = pin->GetOrAddBoolean("floors", "disable_floors", false);
    params.Add("disable_floors", disable_floors);

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

    // Boundary options
    // Note these only trigger on "user" boundary conditions i.e. polar x2, outflow x1
    // Thus you pretty much always want both true.
    bool fix_flux_inflow = pin->GetOrAddBoolean("bounds", "fix_flux_inflow", true);
    params.Add("fix_flux_inflow", fix_flux_inflow);
    bool fix_flux_pole = pin->GetOrAddBoolean("bounds", "fix_flux_pole", true);
    params.Add("fix_flux_pole", fix_flux_pole);


    // Performance options
    // Boundary buffers.  Packing is experimental in Parthenon
    bool buffer_send_pack = pin->GetOrAddBoolean("perf", "buffer_send_pack", false);
    params.Add("buffer_send_pack", buffer_send_pack);
    bool buffer_recv_pack = pin->GetOrAddBoolean("perf", "buffer_recv_pack", false);
    params.Add("buffer_recv_pack", buffer_recv_pack);
    bool buffer_set_pack = pin->GetOrAddBoolean("perf", "buffer_set_pack", false);
    params.Add("buffer_set_pack", buffer_set_pack);
    bool combine_flux_source = pin->GetOrAddBoolean("perf", "combine_flux_source", true);
    params.Add("combine_flux_source", combine_flux_source);

    // Refinement options
    Real refine_tol = pin->GetOrAddReal("GRMHD", "refine_tol", 0.5);
    params.Add("refine_tol", refine_tol);
    Real derefine_tol = pin->GetOrAddReal("GRMHD", "derefine_tol", 0.05);
    params.Add("derefine_tol", derefine_tol);

    // And a flags to keep these fields apart
    // One for primitives specifically
    MetadataFlag isPrimitive = Metadata::AllocateNewFlag("Primitive");
    params.Add("PrimitiveFlag", isPrimitive);
    // And one for HydroDynamics (all of these fields)
    MetadataFlag isHD = Metadata::AllocateNewFlag("HD");
    params.Add("HDFlag", isHD);
    // And one for MagnetoHydroDynamics (all of these plus B)
    MetadataFlag isMHD = Metadata::AllocateNewFlag("MHD");
    params.Add("MHDFlag", isMHD);

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
        // This should be redundant w/ "Overridable" flag...
        // See B field packages for details
        // TODO unmark B and all other primitives from being "Restart" since we don't need to seed UtoP with them
        flags_prim_vec.push_back(Metadata::Overridable);
        m = Metadata(flags_prim_vec, s_vector);
        pkg->AddField("prims.B", m);
        flags_cons_vec.push_back(Metadata::Overridable);
        m = Metadata(flags_cons_vec, s_vector);
        pkg->AddField("cons.B", m);
    }

    // Maximum signal speed (magnitude).  Calculated in flux updates but needed for deciding timestep
    m = Metadata({Metadata::Real, Metadata::Face, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("ctop", m);

    // Temporary fix just for being able to save field values
    // I wish they were really integers, but that's still unsupported
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("pflag", m);
    pkg->AddField("fflag", m);

    pkg->FillDerivedBlock = GRMHD::UtoP;
    pkg->PostFillDerivedBlock = GRMHD::PostUtoP;
    pkg->CheckRefinementBlock = GRMHD::CheckRefinement;
    pkg->EstimateTimestepBlock = GRMHD::EstimateTimestep;
    pkg->PostStepDiagnosticsMesh = GRMHD::PostStepDiagnostics;
    // TODO historyfile output w/new reduction operations
    return pkg;
}

void UtoP(MeshBlockData<Real> *rc)
{
    FLAG("Filling Primitives");
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
    // Note this covers ghost zones!  This is intentional, as primitives in
    // ghost zones are needed for reconstruction
    IndexRange ib = GetPhysicalZonesI(pmb->boundary_flag, pmb->cellbounds);
    IndexRange jb = GetPhysicalZonesJ(pmb->boundary_flag, pmb->cellbounds);
    IndexRange kb = GetPhysicalZonesK(pmb->boundary_flag, pmb->cellbounds);

    pmb->par_for("U_to_P", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            pflag(k, j, i) = GRMHD::u_to_p(G, U, m_u, gam, k, j, i, Loci::center, P, m_p);
        }
    );
    FLAG("Filled");
}

void PostUtoP(MeshBlockData<Real> *rc)
{
    FLAG("Fixing Derived");

    // Apply floors
    if (!rc->GetBlockPointer()->packages.Get("GRMHD")->Param<bool>("disable_floors")) {
        GRMHD::ApplyFloors(rc);
    }

    // Fix inversion errors computing P from U, by averaging adjacent zones
    GRMHD::FixUtoP(rc);

    FLAG("Fixed");
}

Real EstimateTimestep(MeshBlockData<Real> *rc)
{
    FLAG("Estimating timestep");
    auto pmb = rc->GetBlockPointer();
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    const auto& G = pmb->coords;
    auto& ctop = rc->GetFace("ctop").data;

    // TODO: move timestep limiter into an override of SetGlobalTimestep?
    // TODO: move diagnostic printing to PostStepDiagnostics?

    if (!pmb->packages.Get("Globals")->Param<bool>("in_loop")) {
        return pmb->packages.Get("GRMHD")->Param<double>("dt");
    }


    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb->par_reduce("ndt_min", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                      typename Kokkos::MinMax<Real>::value_type &lminmax) {
            double ndt_zone = 1 / (1 / (G.dx1v(i) / ctop(1, k, j, i)) +
                                   1 / (G.dx2v(j) / ctop(2, k, j, i)) +
                                   1 / (G.dx3v(k) / ctop(3, k, j, i)));
            // Effective "max speed" used for the timestep
            double ctop_max_zone = min(G.dx1v(i), min(G.dx2v(j), G.dx3v(k))) / ndt_zone;

            if (ndt_zone < lminmax.min_val) lminmax.min_val = ndt_zone;
            if (ctop_max_zone > lminmax.max_val) lminmax.max_val = ctop_max_zone;
        }
    , Kokkos::MinMax<Real>(minmax));
    // Keep dt to do some checks below
    double min_ndt = minmax.min_val;
    double nctop = minmax.max_val;

    // Apply limits
    double cfl = pmb->packages.Get("GRMHD")->Param<double>("cfl");
    double dt_min = pmb->packages.Get("GRMHD")->Param<double>("dt_min");
    double dt_last = pmb->packages.Get("Globals")->Param<double>("dt_last");
    double dt_max = pmb->packages.Get("GRMHD")->Param<double>("max_dt_increase") * dt_last;
    double ndt = clip(min_ndt * cfl, dt_min, dt_max);

    // Record max ctop, for constraint damping
    if (nctop > pmb->packages.Get("Globals")->Param<Real>("ctop_max")) {
        pmb->packages.Get("Globals")->UpdateParam<Real>("ctop_max", nctop);
    }

#if 0
    // Print stuff about the zone that set the timestep
    // Warning that triggering this code slows the code down pretty badly (~1/3)
    // TODO Fix this to work with multi-mesh, SYCL
    if (pmb->packages.Get("GRMHD")->Param<int>("verbose") >= 3) {
        auto fflag = rc->Get("fflag").data;
        auto pflag = rc->Get("pflag").data;
        pmb->par_for("ndt_min", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA_3D {
                double ndt_zone = 1 / (1 / (G.dx1v(i) / ctop(1, k, j, i)) +
                                       1 / (G.dx2v(j) / ctop(2, k, j, i)) +
                                       1 / (G.dx3v(k) / ctop(3, k, j, i)));
                if (ndt_zone == min_ndt) {
                    printf("Timestep set by %d %d %d: pflag was %f and fflag was %f\n",
                           i, j, k, pflag(k, j, i), fflag(k, j, i));
                }
            }
        );
    }
#endif

    FLAG("Estimated");
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
    FLAG("Printing GRMHD diagnostics");
    auto pmesh = md->GetMeshPointer();
    auto pmb = md->GetBlockData(0)->GetBlockPointer();

    if (md->NumBlocks() > 0) {
        // Debugging/diagnostic info about floor and inversion flags
        int flag_verbose = pmesh->packages.Get("GRMHD")->Param<int>("flag_verbose");
        if (flag_verbose >= 1) {
            FLAG("Printing flags");
            CountPFlags(md, IndexDomain::interior, flag_verbose);
            CountFFlags(md, IndexDomain::interior, flag_verbose);
        }

        // TODO move CheckNaN here?  Do we preserve ctop to end of step correctly?

        if (pmb->packages.Get("GRMHD")->Param<int>("extra_checks") >= 1) {
            CheckNaN(md, 1);
            if (pmesh->ndim > 1) CheckNaN(md, 2);
            if (pmesh->ndim > 2) CheckNaN(md, 3);
        }

        // Extra checking for negative values.  Floors should definitely prevent this,
        // so we save it for dire debugging
        if (pmb->packages.Get("GRMHD")->Param<int>("extra_checks") >= 2) {
            FLAG("Printing negative zones");
            CheckNegative(md, IndexDomain::interior);
        }
    }

    FLAG("Printed");
    return TaskStatus::complete;
}

} // namespace GRMHD
