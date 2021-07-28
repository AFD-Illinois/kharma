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

// Some programmers just want to watch the world burn
// TODO not this, ever again
double last_dt, ctop_max;

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
    double gamma = pin->GetReal("GRMHD", "gamma");
    params.Add("gamma", gamma);
    // Sometimes, you have to leak a few bytes in the name of science
    EOS* eos = CreateEOS(gamma);
    params.Add("eos", eos);

    // Proportion of courant condition for timesteps
    double cfl = pin->GetOrAddReal("GRMHD", "cfl", 0.7);
    params.Add("cfl", cfl);

    // Don't even error on this. LLF or bust, baby
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
    } else if (recon == "weno5_lower_poles") {
        params.Add("recon", ReconstructionType::weno5_lower_poles);
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
    // TODO construct/add a floors struct here instead?
    double rho_min_geom, u_min_geom;
    if (!pin->GetBoolean("coordinates", "spherical")) {
        // In spherical systems, floors drop as r^2, so set them higher by default
        rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-5);
        u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1.e-7);
    } else {
        rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-7);
        u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1.e-9);
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


    bool fix_flux_inflow = pin->GetOrAddBoolean("bounds", "fix_flux_inflow", true);
    params.Add("fix_flux_inflow", fix_flux_inflow);
    bool fix_flux_pole = pin->GetOrAddBoolean("bounds", "fix_flux_pole", true);
    params.Add("fix_flux_pole", fix_flux_pole);

    bool wind_term = pin->GetOrAddBoolean("wind", "on", false);
    params.Add("wind_term", wind_term);
    Real wind_n = pin->GetOrAddReal("wind", "ne", 2.e-4);
    params.Add("wind_n", wind_n);
    Real wind_Tp = pin->GetOrAddReal("wind", "Tp", 10.);
    params.Add("wind_Tp", wind_Tp);
    int wind_pow = pin->GetOrAddInteger("wind", "pow", 4);
    params.Add("wind_pow", wind_pow);
    Real wind_ramp_start = pin->GetOrAddReal("wind", "ramp_start", 0.);
    params.Add("wind_ramp_start", wind_ramp_start);
    Real wind_ramp_end = pin->GetOrAddReal("wind", "ramp_end", 0.);
    params.Add("wind_ramp_end", wind_ramp_end);

    // Performance options
    // Boundary buffers.  Packing is experimental in Parthenon
    bool buffer_send_pack = pin->GetOrAddBoolean("perf", "buffer_send_pack", true);
    params.Add("buffer_send_pack", buffer_send_pack);
    bool buffer_recv_pack = pin->GetOrAddBoolean("perf", "buffer_recv_pack", true);
    params.Add("buffer_recv_pack", buffer_recv_pack);
    bool buffer_set_pack = pin->GetOrAddBoolean("perf", "buffer_set_pack", true);
    params.Add("buffer_set_pack", buffer_set_pack);

    // Refinement options
    Real refine_tol = pin->GetOrAddReal("GRMHD", "refine_tol", 0.5);
    params.Add("refine_tol", refine_tol);
    Real derefine_tol = pin->GetOrAddReal("GRMHD", "derefine_tol", 0.05);
    params.Add("derefine_tol", derefine_tol);

    // Custom sizes
    std::vector<int> s_vector({3});
    std::vector<int> s_fourvector({4});
    std::vector<int> s_prims({NPRIM});
    // And a flag to denote primitives
    MetadataFlag isPrimitive = Metadata::AllocateNewFlag("Primitive");
    params.Add("PrimitiveFlag", isPrimitive);

    // As mentioned elsewhere, KHARMA treats the conserved variables as the independent ones,
    // and the primitives as "Derived"
    // Primitives are still used for reconstruction, physical boundaries, and output, and are
    // generally the easier to understand quantities
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                           Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes}, s_prims);
    pkg->AddField("c.c.bulk.cons", m);
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
                  Metadata::Restart, isPrimitive}, s_prims);
    pkg->AddField("c.c.bulk.prims", m);

    bool use_b = (pin->GetOrAddString("b_field", "solver", "flux_ct") != "none");
    params.Add("use_b", use_b);
    if (!use_b) {
        // Declare placeholder fields only if not using another package providing B field.
        // This should be redundant w/ "Overridable" flag
        // Magnetic field placeholder. "Primitive" and "conserved" forms are the strength and flux respectively
        m = Metadata({Metadata::Overridable,
                    Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                    Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes, Metadata::Vector}, s_vector);
        pkg->AddField("c.c.bulk.B_con", m);
        m = Metadata({Metadata::Overridable,
                    Metadata::Real, Metadata::Cell, Metadata::Derived,
                    Metadata::Restart, isPrimitive, Metadata::Vector}, s_vector);
        pkg->AddField("c.c.bulk.B_prim", m);
    }

    // Maximum signal speed (magnitude).  Calculated in flux updates but needed for deciding timestep
    m = Metadata({Metadata::Real, Metadata::Face, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("f.f.bulk.ctop", m);

    // 4-current jcon. Calculated only for output
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_fourvector);
    pkg->AddField("c.c.bulk.jcon", m);

    // Temporary fix just for being able to save field values
    // I wish they were really integers, but that's still unsupported despite the flag I think
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("c.c.bulk.pflag", m);
    pkg->AddField("c.c.bulk.fflag", m);

    pkg->FillDerivedBlock = GRMHD::UtoP;
    pkg->PostFillDerivedBlock = GRMHD::PostUtoP;
    pkg->CheckRefinementBlock = GRMHD::CheckRefinement;
    pkg->EstimateTimestepBlock = GRMHD::EstimateTimestep;
    //pkg->PostStepUserDiagnosticsInLoop = GRMHD::PostStepDiagnostics;
    return pkg;
}

void UtoP(MeshBlockData<Real> *rc)
{
    FLAG("Filling Primitives");
    auto pmb = rc->GetBlockPointer();
    auto& G = pmb->coords;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVector B_U = rc->Get("c.c.bulk.B_con").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    GridScalar pflag = rc->Get("c.c.bulk.pflag").data;

    // KHARMA uses only one boundary exchange, in the conserved variables
    // Except where FixUtoP has no neighbors, and must fix with bad zones, this is fully identical
    // between #s of MPI ranks, because we sync 4 ghost zones and only require 3 for reconstruction.
    // Thus as long as the last rank is not flagged, it will be inverted the same way on each process, and
    // used in the same way for fixups.  If it fails & thus might be different, it is ignored.

    EOS* eos = pmb->packages.Get("GRMHD")->Param<EOS*>("eos");

    // Get the primitives from our conserved versions
    // Note this covers ghost zones!  This is intentional, as primitives in
    // ghost zones are needed for reconstruction
    int is = is_physical_bound(pmb->boundary_flag[BoundaryFace::inner_x1]) ?
                pmb->cellbounds.is(IndexDomain::interior) : pmb->cellbounds.is(IndexDomain::entire);
    int ie = is_physical_bound(pmb->boundary_flag[BoundaryFace::outer_x1]) ?
                pmb->cellbounds.ie(IndexDomain::interior) : pmb->cellbounds.ie(IndexDomain::entire);
    int js = is_physical_bound(pmb->boundary_flag[BoundaryFace::inner_x2]) ?
                pmb->cellbounds.js(IndexDomain::interior) : pmb->cellbounds.js(IndexDomain::entire);
    int je = is_physical_bound(pmb->boundary_flag[BoundaryFace::outer_x2]) ?
                pmb->cellbounds.je(IndexDomain::interior) : pmb->cellbounds.je(IndexDomain::entire);
    int ks = is_physical_bound(pmb->boundary_flag[BoundaryFace::inner_x3]) ?
                pmb->cellbounds.ks(IndexDomain::interior) : pmb->cellbounds.ks(IndexDomain::entire);
    int ke = is_physical_bound(pmb->boundary_flag[BoundaryFace::outer_x3]) ?
                pmb->cellbounds.ke(IndexDomain::interior) : pmb->cellbounds.ke(IndexDomain::entire);

    pmb->par_for("U_to_P", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            pflag(k, j, i) = GRMHD::u_to_p(G, U, B_U, eos, k, j, i, Loci::center, P);
        }
    );
    FLAG("Filled");
}

void PostUtoP(MeshBlockData<Real> *rc)
{
    FLAG("Fixing Derived");

    // Apply floors
    ApplyFloors(rc);

    // Fix inversion errors computing P from U, by averaging adjacent zones
    // TODO entropy advection version
    FixUtoP(rc);

    FLAG("Fixed");
}

Real EstimateTimestep(MeshBlockData<Real> *rc)
{
    FLAG("Estimating timestep");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto coords = pmb->coords;
    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;

    // TODO: move timestep limiter into an override of the integrator code, keep last_dt somewhere formal/accessible
    // TODO: move diagnostic printing to PostStepDiagnostics. Preserve location without double equality jank?
    // TODO: timestep choices seem to be MPI or meshblock-dependent currently. That's bad.
    //static double last_dt; // For when we no longer need this in post-step output

    // If we're taking the first step, return the default dt and zero speed
    // TODO this needs a more reliable selector
    // This assumes we are called once per block given to this process,
    // which is fragile if PostInit or Parthenon's structure changes.
    // TODO just call fluxes to get ctop and continue?  Could reduce initial jump in divB under CD
    static int ncall = 0;
    int nmb = pmb->pmy_mesh->GetNumMeshBlocksThisRank();
    if (ncall < nmb) {
        last_dt = pmb->packages.Get("GRMHD")->Param<Real>("dt");
        ctop_max = 0.0;
    }
    ncall++;


    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb->par_reduce("ndt_min", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                      typename Kokkos::MinMax<Real>::value_type &lminmax) {
            double ndt_zone = 1 / (1 / (coords.dx1v(i) / ctop(1, k, j, i)) +
                                   1 / (coords.dx2v(j) / ctop(2, k, j, i)) +
                                   1 / (coords.dx3v(k) / ctop(3, k, j, i)));
            // Effective "max speed" used for the timestep
            double ctop_max_zone = min(coords.dx1v(i), min(coords.dx2v(j),coords.dx3v(k))) / ndt_zone;

            if (ndt_zone < lminmax.min_val) lminmax.min_val = ndt_zone;
            if (ctop_max_zone > lminmax.max_val) lminmax.max_val = ctop_max_zone;
        }
    , Kokkos::MinMax<Real>(minmax));
    // Keep dt to do some checks below
    double ndt = minmax.min_val;
    double nctop = minmax.max_val;

    // Print stuff about the zone that set the timestep
    // Warning that triggering this code slows the code down pretty badly (~1/3)
    // TODO: Fix for SYCL, which disallows device printf. Generally solve host/diagnostic versions... HostExec?
    if (pmb->packages.Get("GRMHD")->Param<int>("verbose") > 2) {
        auto fflag = rc->Get("c.c.bulk.fflag").data;
        auto pflag = rc->Get("c.c.bulk.pflag").data;
        pmb->par_for("ndt_min", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA_3D {
                double ndt_zone = 1 / (1 / (coords.dx1v(i) / ctop(1, k, j, i)) +
                                    1 / (coords.dx2v(j) / ctop(2, k, j, i)) +
                                    1 / (coords.dx3v(k) / ctop(3, k, j, i)));
                if (ndt_zone == ndt) {
                    //printf("Timestep set by %d %d %d: pflag was %f and fflag was %f\n",
                    //        i, j, k, pflag(k, j, i), fflag(k, j, i));
                }
            }
        );
    }

    // Nominal timestep
    ndt *= pmb->packages.Get("GRMHD")->Param<Real>("cfl");

    // TODO Subclass integrator to do these checks globally, could fix block-dependence above
    // Sometimes we come out with a silly timestep. Try to salvage it
    double dt_min = pmb->packages.Get("GRMHD")->Param<Real>("dt_min");
    if (ndt < dt_min || isnan(ndt) || ndt > 10000) {
        // Note this still prints for every mesh on rank 0
        if (MPIRank0()) cerr << "ndt was unsafe: " << ndt << "! Using dt_min" << std::endl;
        ndt = dt_min;
        nctop *= 0.01;
    }

    // Only allow dt to increase to e.g. last_dt*1.3 at most
    Real dt_limit = pmb->packages.Get("GRMHD")->Param<Real>("max_dt_increase") * last_dt;
    if (ndt > dt_limit) {
        if(MPIRank0() && pmb->packages.Get("GRMHD")->Param<int>("verbose") > 1) {
            cout << "Limiting dt: " << dt_limit << endl;
        }
        ndt = dt_limit;
        //nctop *= 0.01;
    }

    // Record to some no good very bad file-scope globals
    ctop_max = MPIMax(nctop);
    last_dt = MPIMin(ndt);

    FLAG("Estimated");
    return last_dt;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    auto v = rc->Get("c.c.bulk.prims").data;

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

    typename Kokkos::MinMax<Real>::value_type minmax;
    pmb->par_reduce("check_refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i,
                      typename Kokkos::MinMax<Real>::value_type &lminmax) {
        lminmax.min_val =
            v(prims::rho, k, j, i) < lminmax.min_val ? v(prims::rho, k, j, i) : lminmax.min_val;
        lminmax.max_val =
            v(prims::rho, k, j, i) > lminmax.max_val ? v(prims::rho, k, j, i) : lminmax.max_val;
        }
    , Kokkos::MinMax<Real>(minmax));

    auto pkg = pmb->packages.Get("GRMHD");
    const auto &refine_tol = pkg->Param<Real>("refine_tol");
    const auto &derefine_tol = pkg->Param<Real>("derefine_tol");

    if (minmax.max_val - minmax.min_val > refine_tol) return AmrTag::refine;
    if (minmax.max_val - minmax.min_val < derefine_tol) return AmrTag::derefine;
    return AmrTag::same;
}

TaskStatus PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime& tm)
{
    FLAG("Printing diagnostics");

    // TODO move most of this into debug.cpp for calling on demand/more flexibly
    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();

        auto& G = pmb->coords;
        GridVars P = rc->Get("c.c.bulk.prims").data;
        GridVars U = rc->Get("c.c.bulk.cons").data;
        GridScalar pflag = rc->Get("c.c.bulk.pflag").data;
        GridScalar fflag = rc->Get("c.c.bulk.fflag").data;

        // Debugging/diagnostic info about floor and inversion flags
        int print_flags = pmb->packages.Get("GRMHD")->Param<int>("flag_verbose");
        if (print_flags >= 1) {
            auto fflag_host = fflag.GetHostMirrorAndCopy();
            auto pflag_host = pflag.GetHostMirrorAndCopy();
            int npflags = CountPFlags(pmb, pflag_host, IndexDomain::interior, print_flags);
            int nfflags = CountFFlags(pmb, fflag_host, IndexDomain::interior, print_flags);
        }

        // TODO move checking ctop here?  Have to preserve it without overwriting

        // Extra checking for negative values
        if (pmb->packages.Get("GRMHD")->Param<int>("extra_checks") > 1) {
            IndexDomain domain = IndexDomain::interior;
            int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
            int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
            int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);

            // Check for negative values in the conserved vars
            int nless;
            Kokkos::Sum<int> sum_reducer(nless);
            pmb->par_reduce("count_negative_U", ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA_3D_REDUCE_INT {
                    if (U(prims::rho, k, j, i) < 0.) ++local_result;
                }
            , sum_reducer);
            nless = MPISum(nless);
            if (MPIRank0() && nless > 0) {
                cout << "Number of negative conserved rho: " << nless << endl;
            }
            pmb->par_reduce("count_negative_P", ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA_3D_REDUCE_INT {
                    if (P(prims::rho, k, j, i) < 0.) ++local_result;
                    if (P(prims::u, k, j, i) < 0.) ++local_result;
                }
            , sum_reducer);
            nless = MPISum(nless);
            if (MPIRank0() && nless > 0) {
                cout << "Number of negative primitive rho, u: " << nless << endl;
            }
        }
    }

    return TaskStatus::complete;
}

void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // See issues with determining the first step, above.
    FLAG("Adding output fields");

    auto& rc1 = pmb->meshblock_data.Get();
    auto& rc0 = pmb->meshblock_data.Get("preserve");

    Real dt = last_dt;
    GRMHD::CalculateCurrent(rc0.get(), rc1.get(), dt);

    FLAG("Added");
    //cout << "Filled" << endl;
}

} // namespace GRMHD
