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

    // Fluid gamma for EOS.  Don't guess this.
    double gamma = pin->GetReal("GRMHD", "gamma");
    params.Add("gamma", gamma);

    // Proportion of courant condition for timesteps
    double cfl = pin->GetOrAddReal("GRMHD", "cfl", 0.9);
    params.Add("cfl", cfl);

    // Starting/minimum timestep, if something about the sound speed goes wonky
    // Parthenon allows up to 2x higher dt per step, so it climbs to CFL quite fast
    double dt_min = pin->GetOrAddReal("parthenon/time", "dt_min", 1.e-5);
    params.Add("dt_min", dt_min);

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
    }  // TODO error on bad value...

    // Whether to split or merge recon kernels
    bool merge_recon = pin->GetOrAddBoolean("GRMHD", "merge_reconstruction", true);
    params.Add("merge_recon", merge_recon);
    bool simple_recon = pin->GetOrAddBoolean("GRMHD", "simple_reconstruction", false);
    params.Add("simple_recon", simple_recon);

    // Diagnostic data
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);

    // Magnetic field centering option.  HARM traditionally uses cell-centered fields,
    // but KHARMA is branching into face-centered.  Latter is required for SMR/AMR.
    // TODO Split GRHD from B fields?  Would go a long way toward new centering, etc, etc, etc.
    bool face_fields = false;
    std::string centering = pin->GetOrAddString("GRMHD", "field_centering", "cell");
    if (centering == "face") {
        face_fields = true;
        params.Add("face_fields", true);
    } else {
        face_fields = false;
        params.Add("face_fields", false);
        // TODO if SMR/AMR throw a fit
    }

    // We generally carry around the conserved versions of varialbles, treating them as the fundamental ones
    // However, since most analysis tooling expects the primitives, we *output* those.
    std::vector<int> s_vector({3});
    std::vector<int> s_fourvector({4});
    std::vector<int> s_prims({NPRIM});

    Metadata m;
    m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                    Metadata::Restart, Metadata::Conserved}, s_prims);
    fluid_state->AddField("c.c.bulk.cons", m, DerivedOwnership::shared);
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Intensive}, s_prims);
    fluid_state->AddField("c.c.bulk.prims", m, DerivedOwnership::shared);

#if 0
    // These versions split out the fields.  Someday...

    Metadata m;
    m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                    Metadata::Restart, Metadata::Conserved});
    fluid_state->AddField("c.c.bulk.rho_C", m, DerivedOwnership::shared);
    fluid_state->AddField("c.c.bulk.u_C", m, DerivedOwnership::shared);

    m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                    Metadata::Restart, Metadata::Conserved, Metadata::Vector}, s_vector);
    fluid_state->AddField("c.c.bulk.v_C", m, DerivedOwnership::shared);
    if (!face_fields)
    fluid_state->AddField("c.c.bulk.B_C", m, DerivedOwnership::shared);

    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Intensive});
    fluid_state->AddField("c.c.bulk.rho_P", m, DerivedOwnership::shared);
    fluid_state->AddField("c.c.bulk.u_P", m, DerivedOwnership::shared);

    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Intensive, Metadata::Vector}, s_vector);
    fluid_state->AddField("c.c.bulk.v_P", m, DerivedOwnership::shared);
    fluid_state->AddField("c.c.bulk.B_P", m, DerivedOwnership::shared);
#endif

    // Max (i.e. positive) sound speed vector.  Easiest to keep here due to needing it for EstimateTimestep
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy, Metadata::Vector});
    fluid_state->AddField("f.f.bulk.ctop", m, DerivedOwnership::unique);

    // Add jcon as an output-only calculation, likely overriding MeshBlock::UserWorkBeforeOutput
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, s_fourvector);
    fluid_state->AddField("c.c.bulk.jcon", m, DerivedOwnership::unique);

    fluid_state->FillDerived = GRMHD::FillDerived;
    fluid_state->CheckRefinement = nullptr;
    fluid_state->EstimateTimestep = GRMHD::EstimateTimestep;
    return fluid_state;
}

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived"
 *
 * Note that this step also applies the floors and fixups. Basically it is:
 * input: U, whatever form
 * output: U and P match with inversion errors corrected, and obey floors
 */
void FillDerived(std::shared_ptr<MeshBlockData<Real>>& rc)
{
    FLAG("Filling Derived");
    auto pmb = rc->GetBlockPointer();
    auto& G = pmb->coords;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    // TODO I don't think the flags need a separate sync if I run U_to_P redundantly over ghost zones --
    // it will just produce the same flags in the same zones for each process
    GridInt pflag("pflag", n3, n2, n1);
    GridInt fflag("fflag", n3, n2, n1);

    // TODO See other EOS todos
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    //Diagnostic(rc, IndexDomain::entire);

    // Get the primitives from our conserved versions
    // Note this covers ghost zones!  This is intentional, as primitives in
    // ghost zones are needed for reconstruction
    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("U_to_P", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            pflag(k, j, i) = U_to_P(G, U, eos, k, j, i, Loci::center, P);

            // Apply the floors in the same pass
            fflag(k, j, i) = 0;
            fflag(k, j, i) |= gamma_ceiling(G, P, U, eos, k, j, i);
            // Fixup_floor involves another U_to_P call.  Hide the pflag in bottom 5 bits and retrieve both
            int comboflag = fixup_floor(G, P, U, eos, k, j, i);
            fflag(k, j, i) |= (comboflag / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;

            int pflag_floor = comboflag % HIT_FLOOR_GEOM_RHO;
            if (pflag_floor != 0) {
                pflag(k, j, i) = pflag_floor;
            }
        }
    );
    FLAG("Filled and Floored");

    // Done inline above
    //ApplyFloors(rc);

    // Note I could separate this into a new step if pflag/fflag could be moved out
    FixUtoP(rc, pflag, fflag);

    // Debugging/diagnostic info about floor and inversion flags. Also could be separate if pflag/fflag are state variables.
    int print_flags = pmb->packages["GRMHD"]->Param<int>("flag_verbose");
    if (print_flags) {
        auto fflag_host = fflag.GetHostMirrorAndCopy();
        auto pflag_host = pflag.GetHostMirrorAndCopy();

        if (print_flags == 2) {
            CountPFlags(pmb, pflag_host, IndexDomain::interior, true);
            CountFFlags(pmb, fflag_host, IndexDomain::interior, true);

            // TODO verbose divb?
            if (n3 > 1) {
                cout << "DivB: " << MaxDivB(rc, IndexDomain::interior) << endl;
            }
        } else if (print_flags == 1) {
            // TODO option for entire?
            cout << "PFLAGS: " << CountPFlags(pmb, pflag_host, IndexDomain::interior, false) << endl;
            cout << "FFLAGS: " << CountFFlags(pmb, fflag_host, IndexDomain::interior, false) << endl;

            // TODO for divB: 2D version, conserved variables version
            if (n3 > 1) {
                cout << "DivB: " << MaxDivB(rc, IndexDomain::interior) << endl;
            }
        }
    }

    DelEOS(eos);
}

/**
 * Calculate the LLF flux in a direction
 */
TaskStatus CalculateFlux(std::shared_ptr<MeshBlockData<Real>>& rc, const int& dir)
{
    FLAG(string_format("Calculating flux %d", dir));
    auto pmb = rc->GetBlockPointer();

    bool merge_recon = pmb->packages["GRMHD"]->Param<bool>("merge_recon");
    bool simple_recon = pmb->packages["GRMHD"]->Param<bool>("simple_recon");

    if (merge_recon) {
        return LLF::ReconAndFlux(rc, dir);
    } else {
        int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
        int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
        int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
        GridVars pl("pl", NPRIM, n3, n2, n1);
        GridVars pr("pr", NPRIM, n3, n2, n1);
        GridVars Flux = rc->Get("c.c.bulk.cons").flux[dir];

        ReconstructionType recon = pmb->packages["GRMHD"]->Param<ReconstructionType>("recon");

        // Reconstruct primitives on the faces
        if (simple_recon) {
            Reconstruction::ReconstructLRSimple(rc, pl, pr, dir, recon);
        } else {
            Reconstruction::ReconstructLR(rc, pl, pr, dir, recon);
        }

        // Calculate flux from values at left & right of face
        LLF::LRToFlux(rc, pr, pl, dir, Flux);

        FLAG(string_format("Calculated flux %d", dir));

        return TaskStatus::complete;
    }
}

/**
 * Add HARM source term to RHS
 */
TaskStatus AddSourceTerm(std::shared_ptr<MeshBlockData<Real>>& rc, std::shared_ptr<MeshBlockData<Real>>& dudt)
{
    FLAG("Adding source term");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    GridVars P = rc->Get("c.c.bulk.prims").data;
    auto& G = pmb->coords;

    // TODO *sigh*
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    // Unpack for kernel
    auto dUdt = dudt->Get("c.c.bulk.cons").data;

    pmb->par_for("source_term", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Calculate the source term and apply it in 1 go (since it's stencil-1)
            FourVectors Dtmp;
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            add_fluid_source(G, P, Dtmp, eos, k, j, i, dUdt);
        }
    );

    DelEOS(eos);

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
Real EstimateTimestep(std::shared_ptr<MeshBlockData<Real>>& rc)
{
    FLAG("Estimating timestep");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto coords = pmb->coords;
    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;

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

    FLAG("Estimated");
    return ndt;
}



} // namespace GRMHD
