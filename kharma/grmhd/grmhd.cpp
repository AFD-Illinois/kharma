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
#include "debug.hpp"
#include "fixup.hpp"
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

    // Omit jcon calculation before dumps.  For optimum speed I guess?
    double no_jcon = pin->GetOrAddBoolean("GRMHD", "no_jcon", false);
    params.Add("no_jcon", no_jcon);

    // Starting/minimum timestep, if something about the sound speed goes wonky
    double dt_min = pin->GetOrAddReal("parthenon/time", "dt_min", 1.e-5);
    params.Add("dt_min", dt_min);
    double max_dt_increase = pin->GetOrAddReal("parthenon/time", "max_dt_increase", 1.1);
    params.Add("max_dt_increase", max_dt_increase);

    // Reconstruction scheme: plm, weno5, ppm...
    std::string recon = pin->GetOrAddString("GRMHD", "reconstruction", "weno5");
    if (recon == "linear_mc") {
        params.Add("recon", ReconstructionType::linear_mc);
    } else if (recon == "ppm") {
        params.Add("recon", ReconstructionType::ppm);
    } else if (recon == "weno5") {
        params.Add("recon", ReconstructionType::weno5);
    } else if (recon == "mp5") {
        params.Add("recon", ReconstructionType::mp5);
    } else {
        throw std::invalid_argument("Reconstruction must be one of linear_mc, ppm, weno5, mp5!");
    }

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    bool flag_save = pin->GetOrAddBoolean("debug", "flag_save", false);
    params.Add("flag_save", flag_save);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // Floor parameters
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

    double gamma_max = pin->GetOrAddReal("floors", "gamma_max", 50);
    params.Add("gamma_max", gamma_max);

    bool temp_adjust_u = pin->GetOrAddBoolean("floors", "temp_adjust_u", false);
    params.Add("temp_adjust_u", temp_adjust_u);
    bool fluid_frame = pin->GetOrAddBoolean("floors", "fluid_frame", false);
    params.Add("fluid_frame", fluid_frame);

    // TODO separate block?
    bool wind_term = pin->GetOrAddBoolean("floors", "wind_term", false);
    params.Add("wind_term", wind_term);
    Real wind_n = pin->GetOrAddReal("floors", "wind_n", 2.e-4);
    params.Add("wind_n", wind_n);
    Real wind_Tp = pin->GetOrAddReal("floors", "wind_Tp", 10.);
    params.Add("wind_Tp", wind_Tp);
    int wind_pow = pin->GetOrAddInteger("floors", "wind_pow", 4);
    params.Add("wind_pow", wind_pow);
    Real wind_ramp = pin->GetOrAddReal("floors", "wind_ramp", 0.);
    params.Add("wind_ramp", wind_ramp);

    std::vector<int> s_vector({3});
    std::vector<int> s_fourvector({4});
    std::vector<int> s_prims({NPRIM});

    // As mentioned elsewhere, KHARMA treats the conserved variables as the independent ones,
    // and the primitives as "Derived."
    // They're still necessary for reconstruction, and generally are the quantities in output files
    Metadata m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                    Metadata::Restart, Metadata::Conserved}, s_prims);
    fluid_state->AddField("c.c.bulk.cons", m, DerivedOwnership::shared);
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Intensive, Metadata::Restart}, s_prims);
    fluid_state->AddField("c.c.bulk.prims", m, DerivedOwnership::shared);

    // Maximum signal speed (magnitude).  Calculated in flux updates but needed for deciding timestep
    // TODO figure out how to preserve either this or the timestep in restart files
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy});
    fluid_state->AddField("f.f.bulk.ctop", m, DerivedOwnership::unique);

    // Add jcon as an output-only calculation, likely overriding MeshBlock::UserWorkBeforeOutput
    if (!no_jcon) {
        m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_fourvector);
        fluid_state->AddField("c.c.bulk.jcon", m, DerivedOwnership::unique);
    }

    if (flag_save) {
        m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, 1);
        fluid_state->AddField("c.c.bulk.pflag", m, DerivedOwnership::unique);
        fluid_state->AddField("c.c.bulk.fflag", m, DerivedOwnership::unique);
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

    // I don't think the flags need a separate sync if I run U_to_P redundantly over ghost zones --
    // it will just produce the same flags in the same zones for each process
    // TODO verify
    GridInt pflag("pflag", n3, n2, n1);
    GridInt fflag("fflag", n3, n2, n1);

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    // Pull out a struct of just the actual floor values for speed
    FloorPrescription floors = FloorPrescription(pmb->packages["GRMHD"]->AllParams());

    //Diagnostic(rc, IndexDomain::entire);

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

            // Apply the floors in the same pass
            fflag(k, j, i) = 0;

            // Fixup_floor involves another U_to_P call.  Hide the pflag in bottom 5 bits and retrieve both
            int comboflag = apply_floors(G, P, U, eos, k, j, i, floors);
            fflag(k, j, i) |= (comboflag / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;
            // If the floor was applied (i.e. did an inversion), overwrite the original flag
            if (fflag(k, j, i)) pflag(k, j, i) = comboflag % HIT_FLOOR_GEOM_RHO;

            // Apply ceilings *after* floors, to make the temperature ceiling better-behaved
            int fflag_c = apply_ceilings(G, P, U, eos, k, j, i, floors);
            fflag(k, j, i) |= fflag_c;
            // Since ceilings are applied without even a U_to_P call, avoid fixups
            if (fflag_c) pflag(k, j, i) = InversionStatus::success;
        }
    );
    FLAG("Filled and Floored");

    // Done inline above
    //ApplyFloors(rc);

    // Note I could separate this into a new step if pflag/fflag could be moved out
    FixUtoP(rc, pflag, fflag);

    // Debugging/diagnostic info about floor and inversion flags.
    // Also should be separate...
    // Note we only print flags in the interior, borders are covered either by other blocks,
    // or by the outflow/polar conditions later
    int print_flags = pmb->packages["GRMHD"]->Param<int>("flag_verbose");
    bool save_flags = pmb->packages["GRMHD"]->Param<bool>("flag_save");
    if (print_flags > 0) {
        auto fflag_host = fflag.GetHostMirrorAndCopy();
        auto pflag_host = pflag.GetHostMirrorAndCopy();
        int npflags = CountPFlags(pmb, pflag_host, IndexDomain::interior, print_flags);
        int nfflags = CountFFlags(pmb, fflag_host, IndexDomain::interior, print_flags);

        // Anything you want here
        //cerr << string_format("UtoP domain: %d-%d,%d-%d,%d-%d",is,ie,js,je,ks,ke) << endl;
    }
    if (save_flags) {
        GridScalar pflag_save = rc->Get("c.c.bulk.pflag").data;
        GridScalar fflag_save = rc->Get("c.c.bulk.pflag").data;
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
TaskStatus ApplyFluxes2D(SimTime tm, MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt)
{
    FLAG("Applying fluxes");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars F1 = rc->Get("c.c.bulk.cons").flux[X1DIR];
    GridVars F2 = rc->Get("c.c.bulk.cons").flux[X2DIR];
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    bool wind_term = pmb->packages["GRMHD"]->Param<bool>("wind_term");
    Real wind_n = pmb->packages["GRMHD"]->Param<Real>("wind_n");
    Real wind_Tp = pmb->packages["GRMHD"]->Param<Real>("wind_Tp");
    int wind_pow = pmb->packages["GRMHD"]->Param<int>("wind_pow");
    Real wind_ramp = pmb->packages["GRMHD"]->Param<Real>("wind_ramp");
    Real current_wind_n = (wind_ramp > 0.0) ? min(tm.time / wind_ramp, 1.0) * wind_n : wind_n;
    //cerr << "Winding at " << current_wind_n << endl;

    // Unpack for kernel
    auto dUdt = dudt->Get("c.c.bulk.cons").data;

    pmb->par_for("apply_fluxes", js, je, is, ie,
        KOKKOS_LAMBDA_2D {
            // Calculate the source term and apply it in 1 go (since it's stencil-1)
            FourVectors Dtmp;
            Real dU[NPRIM] = {0};
            get_state(G, P, 0, j, i, Loci::center, Dtmp);
            get_fluid_source(G, P, Dtmp, eos, 0, j, i, dU);
            if (wind_term) add_wind(G, eos, 0, j, i, current_wind_n, wind_pow, wind_Tp, dU);

            PLOOP dUdt(p, 0, j, i) = (F1(p, 0, j, i) - F1(p, 0, j, i+1)) / G.dx1v(i) +
                                     (F2(p, 0, j, i) - F2(p, 0, j+1, i)) / G.dx2v(j) +
                                     dU[p];
        }
    );
    FLAG("Applied");

    return TaskStatus::complete;
}
TaskStatus ApplyFluxes(SimTime tm, MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt)
{
    FLAG("Applying fluxes");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    if (ks == ke) return ApplyFluxes2D(tm, rc, dudt);
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars F1 = rc->Get("c.c.bulk.cons").flux[X1DIR];
    GridVars F2 = rc->Get("c.c.bulk.cons").flux[X2DIR];
    GridVars F3 = rc->Get("c.c.bulk.cons").flux[X3DIR];
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;

    EOS* eos = pmb->packages["GRMHD"]->Param<EOS*>("eos");

    bool wind_term = pmb->packages["GRMHD"]->Param<bool>("wind_term");
    Real wind_n = pmb->packages["GRMHD"]->Param<Real>("wind_n");
    Real wind_Tp = pmb->packages["GRMHD"]->Param<Real>("wind_Tp");
    int wind_pow = pmb->packages["GRMHD"]->Param<int>("wind_pow");
    Real wind_ramp = pmb->packages["GRMHD"]->Param<Real>("wind_ramp");
    Real current_wind_n = (wind_ramp > 0.0) ? min(tm.time / wind_ramp, 1.0) * wind_n : wind_n;
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
            if (wind_term) add_wind(G, eos, 0, j, i, current_wind_n, wind_pow, wind_Tp, dU);

            PLOOP dUdt(p, k, j, i) = (F1(p, k, j, i) - F1(p, k, j, i+1)) / G.dx1v(i) +
                                     (F2(p, k, j, i) - F2(p, k, j+1, i)) / G.dx2v(j) +
                                     (F3(p, k, j, i) - F3(p, k+1, j, i)) / G.dx3v(k) +
                                     dU[p];
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

    // TODO parthenon takes a dt, or we could namespace ourselves into the driver to get integrator access...
    static double last_dt = pmb->packages["GRMHD"]->Param<Real>("dt_min");

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

    // Sometimes we come out with a silly timestep. Try to salvage it
    // TODO don't allow the *overall* timestep to be >1, while still allowing *blocks* to have larger steps
    if (ndt <= 0.0 || isnan(ndt) || ndt > 10000) {
        cerr << "ndt was unsafe: " << ndt << "! Using dt_min" << std::endl;
        ndt = pmb->packages["GRMHD"]->Param<Real>("dt_min");
    } else {
        ndt *= pmb->packages["GRMHD"]->Param<Real>("cfl");
    }

    // Only allow the local dt to increase by a certain amount.  This also caps the global
    // one, but preserves local values in case we substep I guess?
    Real dt_limit = pmb->packages["GRMHD"]->Param<Real>("max_dt_increase") * last_dt;
    if (ndt > dt_limit) {
        if(pmb->packages["GRMHD"]->Param<int>("verbose") > 0){
            cerr << "Limiting dt: " << dt_limit;
        }
        ndt = dt_limit;
    }

    last_dt = ndt;

    FLAG("Estimated");
    return ndt;
}



} // namespace GRMHD
