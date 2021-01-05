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

#include "parthenon/parthenon.hpp"

#include "decs.hpp"

#include "boundaries.hpp"
#include "current.hpp"
#include "debug.hpp"
#include "fixup.hpp"
#include "floors.hpp"
#include "fluxes.hpp"
#include "gr_coordinates.hpp"
#include "phys.hpp"
#include "source.hpp"
#include "U_to_P.hpp"

using namespace parthenon;
// Need to access these directly for reductions
using namespace Kokkos;

namespace GRMHD
{

/**
 * Declare fields
 *
 * TODO:
 * Check metadata flags, esp ctop
 * Add pflag as an integer "field", or at least something sync-able
 * Split out B fields to option them face-centered
 */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin)
{
    auto fluid_state = std::make_shared<StateDescriptor>("GRMHD");
    Params &params = fluid_state->AllParams();

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
    double cfl = pin->GetOrAddReal("GRMHD", "cfl", 0.9);
    params.Add("cfl", cfl);

    // Option to omit jcon calculation before dumps.  For optimum speed I guess?
    // TODO make this and next automatic by reading outputs
    bool add_jcon = pin->GetOrAddBoolean("GRMHD", "add_jcon", false);
    params.Add("add_jcon", add_jcon);
    bool flag_save = pin->GetOrAddBoolean("GRMHD", "add_flags", false);
    params.Add("flag_save", flag_save);
    bool add_divB = pin->GetOrAddBoolean("GRMHD", "add_divB", false);
    params.Add("add_divB", add_divB);

    // Minimum timestep, if something about the sound speed goes wonky. Probably won't save you :)
    double dt_min = pin->GetOrAddReal("parthenon/time", "dt_min", 1.e-6);
    params.Add("dt_min", dt_min);
    // Starting timestep.  Important for consistent restarts, otherwise just the minimum
    double dt = pin->GetOrAddReal("parthenon/time", "dt", dt_min);
    params.Add("dt", dt);
    double max_dt_increase = pin->GetOrAddReal("parthenon/time", "max_dt_increase", 1.3);
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
    double rho_min_geom = pin->GetOrAddReal("floors", "rho_min_geom", 1.e-5);
    params.Add("rho_min_geom", rho_min_geom);
    double u_min_geom = pin->GetOrAddReal("floors", "u_min_geom", 1.e-7);
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
    bool fix_flux_inflow = pin->GetOrAddBoolean("floors", "fix_flux_inflow", true);
    params.Add("fix_flux_inflow", fix_flux_inflow);
    bool fix_flux_B = pin->GetOrAddBoolean("floors", "fix_flux_B", true);
    params.Add("fix_flux_B", fix_flux_B);

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

    std::vector<int> s_vector({3});
    std::vector<int> s_fourvector({4});
    std::vector<int> s_prims({NPRIM});

    // As mentioned elsewhere, KHARMA treats the conserved variables as the independent ones,
    // and the primitives as "Derived."
    // They're still necessary for reconstruction, and generally are the quantities in output files
    Metadata m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                    Metadata::Restart, Metadata::Conserved}, s_prims);
    fluid_state->AddField("c.c.bulk.cons", m);
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Restart}, s_prims);
    fluid_state->AddField("c.c.bulk.prims", m);

    // Maximum signal speed (magnitude).  Calculated in flux updates but needed for deciding timestep
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy});
    fluid_state->AddField("f.f.bulk.ctop", m);

    // Add jcon as an output-only calculation, likely overriding MeshBlock::UserWorkBeforeOutput
    if (add_jcon) {
        m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_fourvector);
        fluid_state->AddField("c.c.bulk.jcon", m);
    }

    if (flag_save) {
        m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, 1);
        fluid_state->AddField("c.c.bulk.pflag", m);
        fluid_state->AddField("c.c.bulk.fflag", m);
    }

    fluid_state->FillDerivedBlock = GRMHD::UtoP;
    fluid_state->CheckRefinementBlock = nullptr;
    fluid_state->EstimateTimestepBlock = GRMHD::EstimateTimestep;
    return fluid_state;
}

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived".
 * Also applies floors to the calculated primitives, and fixes up any inversion errors
 *
 * input: U, whatever form
 * output: U and P match with inversion errors corrected, and obey floors
 */
void UtoP(MeshBlockData<Real> *rc)
{
    FLAG("Filling Derived");
    auto pmb = rc->GetBlockPointer();
    auto& G = pmb->coords;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    // KHARMA uses only one boundary exchange, in the conserved variables
    // Except where FixUtoP has no neighbors, and must fix with bad zones, this is fully identical
    // between #s of MPI ranks, because we sync 4 ghost zones and only require 3 for reconstruction.
    // Thus if the last rank is not flagged, it will be inverted the same way on each process, and
    // used in the same way for fixups.  If it fails & thus might be different, it is ignored.
    GridInt pflag("pflag", n3, n2, n1);
    GridInt fflag("fflag", n3, n2, n1);

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    // Pull out a struct of just the actual floor values for speed
    FloorPrescription floors = FloorPrescription(pmb->packages["GRMHD"]->AllParams());

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
            pflag(k, j, i) = u_to_p(G, U, eos, k, j, i, Loci::center, P);

            // Apply the floors in the same pass (if we can trust P!)
            // Note we do this even for flagged cells!  In the worst case, we may have to use
            // flagged cells to fix flagged cells, so they should at least obey floors
            fflag(k, j, i) = 0;

            // Fixup_floor involves another U_to_P call.  Hide the pflag in bottom 5 bits and retrieve both
            int comboflag = apply_floors(G, P, U, eos, k, j, i, floors);
            fflag(k, j, i) |= (comboflag / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;

            // Apply ceilings *after* floors, to make the temperature ceiling better-behaved
            // Ceilings don't involve a U_to_P call
            fflag(k, j, i) |= apply_ceilings(G, P, U, eos, k, j, i, floors);

            // Optionally record the floor inversion failures & average over them
            // Floors are still applied in fluid frame if this isn't done, but it might help?
            // if (pflag(k, j, i) == InversionStatus::success) {
            //     pflag(k, j, i) = comboflag % HIT_FLOOR_GEOM_RHO;
            // }
        }
    );
    FLAG("Filled and Floored");

    // Note I could separate this into a new step if pflag/fflag could be moved out
    FixUtoP(rc, pflag, fflag);

    // Debugging/diagnostic info about floor and inversion flags.
    // Also should be separate...
    // Note we only print flags in the interior, borders are covered either by other blocks,
    // or by the outflow/polar conditions later
    int print_flags = pmb->packages["GRMHD"]->Param<int>("flag_verbose");
    if (print_flags > 0) {
        auto fflag_host = fflag.GetHostMirrorAndCopy();
        auto pflag_host = pflag.GetHostMirrorAndCopy();
        int npflags = CountPFlags(pmb, pflag_host, IndexDomain::interior, print_flags);
        int nfflags = CountFFlags(pmb, fflag_host, IndexDomain::interior, print_flags);
    }
    if (pmb->packages["GRMHD"]->Param<bool>("flag_save")) {
        GridScalar pflag_save = rc->Get("c.c.bulk.pflag").data;
        GridScalar fflag_save = rc->Get("c.c.bulk.fflag").data;
        pmb->par_for("save_flags", ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA_3D {
                pflag_save(k,j,i) = pflag(k,j,i);
                fflag_save(k,j,i) = fflag(k,j,i);
            }
        );
    }
}

/**
 * Calculate dU/dt from a set of fluxes.
 * Shortcut for Parthenon's FluxDivergence & GRMHD's AddSourceTerm
 *
 * @param rc is the current stage's container
 * @param base is the base container containing the global dUdt term
 */
TaskStatus ApplyFluxes(SimTime tm, MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt)
{
    FLAG("Applying fluxes");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    int ndim = 3;
    if (js == je) {
        ndim = 1;
    } else if (ks == ke) {
        ndim = 2;
    }

    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars F1, F2, F3;
    F1 = rc->Get("c.c.bulk.cons").flux[X1DIR];
    if (ndim > 1) F2 = rc->Get("c.c.bulk.cons").flux[X2DIR];
    if (ndim > 2) F3 = rc->Get("c.c.bulk.cons").flux[X3DIR];
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    bool wind_term = pmb->packages["GRMHD"]->Param<bool>("wind_term");
    Real wind_n = pmb->packages["GRMHD"]->Param<Real>("wind_n");
    Real wind_Tp = pmb->packages["GRMHD"]->Param<Real>("wind_Tp");
    int wind_pow = pmb->packages["GRMHD"]->Param<int>("wind_pow");
    Real wind_ramp_start = pmb->packages["GRMHD"]->Param<Real>("wind_ramp_start");
    Real wind_ramp_end = pmb->packages["GRMHD"]->Param<Real>("wind_ramp_end");
    Real current_wind_n;
    if (wind_ramp_end > 0.0) {
        current_wind_n = min((tm.time - wind_ramp_start) / (wind_ramp_end - wind_ramp_start), 1.0) * wind_n;
    } else {
        current_wind_n = wind_n;
    }
    //cerr << "Winding at " << current_wind_n << endl;

    // Unpack for kernel
    auto dUdt = dudt->Get("c.c.bulk.cons").data;

    pmb->par_for("apply_fluxes", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Calculate the source term and apply it in 1 go (since it's stencil-1)
            FourVectors Dtmp;
            Real dU[NPRIM] = {0};
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            get_fluid_source(G, P, Dtmp, eos, k, j, i, dU);
            if (wind_term) add_wind(G, eos, k, j, i, current_wind_n, wind_pow, wind_Tp, dU);

            PLOOP {
                dUdt(p, k, j, i) = (F1(p, k, j, i) - F1(p, k, j, i+1)) / G.dx1v(i) + dU[p];
                if (ndim > 1) dUdt(p, k, j, i) += (F2(p, k, j, i) - F2(p, k, j+1, i)) / G.dx2v(j);
                if (ndim > 2) dUdt(p, k, j, i) += (F3(p, k, j, i) - F3(p, k+1, j, i)) / G.dx3v(k);
            }
        }
    );
    FLAG("Applied");

    return TaskStatus::complete;
}

/**
 * Returns the minimum CFL timestep among all zones in the block,
 * multiplied by a proportion "cfl" for safety.
 *
 * This is just for a particular MeshBlock/package, so don't rely on it
 * Parthenon will take the minimum and put it in pmy_mesh->dt
 */
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

    static double last_dt;

    // Overall TODO: Override timestep class in the driver, and move
    // this stuff there.  

    // Use parameter "dt" on the first step
    static bool first_call = true;
    if (first_call) {
        first_call = false;
        last_dt = pmb->packages["GRMHD"]->Param<Real>("dt") * pmb->packages["GRMHD"]->Param<Real>("cfl");
        return last_dt;
    }

    // TODO preserve location, needs custom (?) Kokkos Index type for 3D
    double ndt;
    Kokkos::Min<double> min_reducer(ndt);
    pmb->par_reduce("ndt_min", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            double ndt_zone = 1 / (1 / (coords.dx1v(i) / ctop(1, k, j, i)) +
                                   1 / (coords.dx2v(j) / ctop(2, k, j, i)) +
                                   1 / (coords.dx3v(k) / ctop(3, k, j, i)));
            if (ndt_zone < local_result) {
                local_result = ndt_zone;
            }
        }
    , min_reducer);

    // Warning that triggering this code slows the code down pretty badly (~1/3)
    if (pmb->packages["GRMHD"]->Param<int>("verbose") > 1) {
        if (pmb->packages["GRMHD"]->Param<bool>("flag_save")) {
            auto fflag = rc->Get("c.c.bulk.fflag").data;
            auto pflag = rc->Get("c.c.bulk.pflag").data;
            pmb->par_for("ndt_min", ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA_3D {
                    double ndt_zone = 1 / (1 / (coords.dx1v(i) / ctop(1, k, j, i)) +
                                        1 / (coords.dx2v(j) / ctop(2, k, j, i)) +
                                        1 / (coords.dx3v(k) / ctop(3, k, j, i)));
                    if (ndt_zone == ndt) {
                        printf("Timestep set by %d %d %d: pflag was %f and fflag was %f\n",
                                i, j, k, pflag(k, j, i), fflag(k, j, i));
                    }
                }
            );
        } else {
            pmb->par_for("ndt_min", ks, ke, js, je, is, ie,
                KOKKOS_LAMBDA_3D {
                    double ndt_zone = 1 / (1 / (coords.dx1v(i) / ctop(1, k, j, i)) +
                                        1 / (coords.dx2v(j) / ctop(2, k, j, i)) +
                                        1 / (coords.dx3v(k) / ctop(3, k, j, i)));
                    if (ndt_zone == ndt) {
                        printf("Timestep set by %d %d %d\n", i, j, k);
                    }
                }
            );
        }
    }
    ndt *= pmb->packages["GRMHD"]->Param<Real>("cfl");

    // Sometimes we come out with a silly timestep. Try to salvage it
    double dt_min = pmb->packages["GRMHD"]->Param<Real>("dt_min");
    if (ndt < dt_min || isnan(ndt) || ndt > 10000) {
        cerr << "ndt was unsafe: " << ndt << "! Using dt_min" << std::endl;
        ndt = dt_min;
    }

    // Only allow the local dt to increase by a certain amount.  This also caps the global
    // one, but preserves local values in case we substep I guess?
    Real dt_limit = pmb->packages["GRMHD"]->Param<Real>("max_dt_increase") * last_dt;
    if (ndt > dt_limit) {
        if(pmb->packages["GRMHD"]->Param<int>("verbose") > 0) {
            cout << "Limiting dt: " << dt_limit << endl;
        }
        ndt = dt_limit;
    }

    last_dt = ndt;

    FLAG("Estimated");
    return ndt;
}

void FillOutput(MeshBlock *pmb, double dt)
{
    FLAG("Adding output fields");

    auto& rc0 = pmb->meshblock_data.Get("preserve");
    auto& rc1 = pmb->meshblock_data.Get();
    CalculateCurrent(rc0.get(), rc1.get(), dt);

    // TODO divB!


    FLAG("Added");
}

} // namespace GRMHD
