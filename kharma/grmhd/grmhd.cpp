// Functions defining the evolution of GRMHD fluid

#include <memory>

// Until Parthenon gets a reduce()
#include "Kokkos_Core.hpp"

#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"

#include "decs.hpp"

#include "boundaries.hpp"
#include "coordinate_embedding.hpp"
#include "coordinate_systems.hpp"
#include "debug.hpp"
#include "fixup.hpp"
#include "fluxes.hpp"
#include "grmhd.hpp"
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

    // Fluid gamma for EOS (TODO separate EOS class to make this broader)
    double gamma = pin->GetOrAddReal("GRMHD", "gamma", 4. / 3);
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
    }

    // Whether to split or merge recon kernels
    bool merge_recon = pin->GetOrAddBoolean("GRMHD", "merge_reconstruction", true);
    params.Add("merge_recon", merge_recon);

    // Magnetic field centering option.  HARM traditionally uses cell-centered fields,
    // but KHARMA is branching into face-centered.  Latter is required for SMR/AMR.
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

    // TODO integer/flag fields in Parthenon
    // Or hack it with doubles/casts
    // m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy}, s_fourvector);
    // fluid_state->AddField("c.c.bulk.pflag", m, DerivedOwnership::unique);
    // fluid_state->AddField("c.c.bulk.fflag", m, DerivedOwnership::unique);

    fluid_state->FillDerived = GRMHD::FillDerived;
    fluid_state->CheckRefinement = nullptr;
    fluid_state->EstimateTimestep = GRMHD::EstimateTimestep;
    return fluid_state;
}

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived"
 * TODO check if this is done again before output...
 * Note that this step also applies the floors and fixups. Basically it is:
 * input: U, whatever form
 * output: U and P match with inversion errors corrected, and obey floors
 */
void FillDerived(std::shared_ptr<Container<Real>>& rc)
{
    FLAG("Filling Derived");
    MeshBlock *pmb = rc->pmy_block;
    GRCoordinates G = pmb->coords;

    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVars P = rc->Get("c.c.bulk.prims").data;

    //GridVars pflag = rc->Get("bulk.pflag").data; // TODO parthenon int variables
    GridInt pflag("pflag", n3, n2, n1);
    GridInt fflag("fflag", n3, n2, n1);

    // TODO See other EOS todos
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    //Diagnostic(rc, IndexDomain::entire);

    // Get the primitives from our conserved versions
    // Note this covers ghost zones!  This is intentional, as primitives in
    // ghost zones are needed for reconstruction
    pmb->par_for("U_to_P", 0, n3-1, 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA_3D {
            pflag(k, j, i) = U_to_P(G, U, eos, k, j, i, Loci::center, P);
            fflag(k, j, i) = 0;
            fflag(k, j, i) |= fixup_ceiling(G, P, U, eos, k, j, i);
            // Fixup_floor involves another U_to_P call.  Hide the pflag in bottom 5 bits and retrieve both
            int comboflag = fixup_floor(G, P, U, eos, k, j, i);
            fflag(k, j, i) |= (comboflag / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;

            int pflag_floor = comboflag % HIT_FLOOR_GEOM_RHO;
            if (pflag_floor != 0) {
                pflag(k, j, i) = pflag_floor;
            }
        }
    );
    FLAG("Filled");

    //ApplyFloors(rc);
    //FLAG("Floored");

    // We expect primitives all the way out to 3 ghost zones on all sides.  But we can only fix primitives with their neighbors.
    // This may actually mean we require the 4 ghost zones Parthenon "wants" us to have, if we need to use only fixed zones.
    // TODO alternatively do a bounds check in fix_U_to_P
    ClearCorners(pmb, pflag); // Don't use zones in physical corners. TODO persist this?
    FLAG("Cleared corner flags");
    pmb->par_for("fix_U_to_P", 1, n3-2, 1, n2-2, 1, n1-2,
        KOKKOS_LAMBDA_3D {
            // Ignore pflags incurred syncing after applying floors to averaged zones.  TODO why does such a thing even exist
            fflag(k, j, i) |= (fix_U_to_P(G, P, U, eos, pflag, k, j, i) / HIT_FLOOR_GEOM_RHO) * HIT_FLOOR_GEOM_RHO;
        }
    );
    FLAG("Fixed failed inversions");

#if DEBUG
    // TODO this does the calculation from primitives, but we could also
    // use the conserved vars
    double maxDivB = MaxDivB(rc);
    fprintf(stderr, "Maximum divB: %g\n", maxDivB);

    auto fflag_host = fflag.GetHostMirrorAndCopy();
    auto pflag_host = pflag.GetHostMirrorAndCopy();
    CountFFlags(pmb, fflag_host, IndexDomain::interior);
    CountPFlags(pmb, pflag_host, IndexDomain::interior);
#elif 0
    auto fflag_host = fflag.GetHostMirrorAndCopy();
    auto pflag_host = pflag.GetHostMirrorAndCopy();
    cout << "PFLAGS interior: " << CountPFlags(pmb, pflag_host, IndexDomain::interior, false)
         << " entire: " << CountPFlags(pmb, pflag_host, IndexDomain::entire, false) << endl;
    cout << "FFLAGS interior: " << CountFFlags(pmb, fflag_host, IndexDomain::interior, false)
         << " entire: " << CountFFlags(pmb, fflag_host, IndexDomain::entire, false) << endl;
    cout << "DivB interior: " << MaxDivB(rc, IndexDomain::interior)
         << " entire: " << MaxDivB(rc, IndexDomain::entire) << endl;
#endif

    DelEOS(eos);
}

/**
 * Calculate the LLF flux in a direction
 * Note this is the sequential recon/LR version -- 
 * an interleaved version is also available as ReconAndFlux
 * TODO Make async?
 */
TaskStatus CalculateFlux(std::shared_ptr<Container<Real>>& rc, const int& dir)
{
    FLAG(string_format("Calculating flux %d", dir));
    MeshBlock *pmb = rc->pmy_block;
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    GridVars pl("pl", NPRIM, n3, n2, n1);
    GridVars pr("pr", NPRIM, n3, n2, n1);
    GridVars Flux = rc->Get("c.c.bulk.cons").flux[dir];

    ReconstructionType recon = pmb->packages["GRMHD"]->Param<ReconstructionType>("recon");

    // Reconstruct primitives on the faces
    Reconstruction::ReconstructLR(rc, pl, pr, dir, recon);

    // Calculate flux from values at left & right of face
    LRToFlux(rc, pr, pl, dir, Flux);

    FLAG(string_format("Calculated flux %d", dir));

    return TaskStatus::complete;
}

/**
 * Reconstruct primitives at a face, *and* turn them into the LLF flux
 *
 * Also fills the "ctop" vector with the highest magnetosonic speed mhd_vchar -- used to estimate timestep later.
 *
 */
TaskStatus ReconAndFlux(std::shared_ptr<Container<Real>>& rc, const int& dir)
{
    FLAG(string_format("Recon and flux X%d", dir));
    MeshBlock *pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain)-1, ie = pmb->cellbounds.ie(domain)+1;
    int js = pmb->cellbounds.js(domain)-1, je = pmb->cellbounds.je(domain)+1;
    int ks = pmb->cellbounds.ks(domain)-1, ke = pmb->cellbounds.ke(domain)+1;
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);

    auto& P = rc->Get("c.c.bulk.prims").data;
    auto& flux = rc->Get("c.c.bulk.cons").flux[dir];

    GRCoordinates G = pmb->coords;
    // TODO *sigh*
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);
    ReconstructionType recon = pmb->packages["GRMHD"]->Param<ReconstructionType>("recon");

    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;

    // So far we don't need fluxes that don't match faces
    Loci loc;
    switch (dir) {
    case X1DIR:
        loc = Loci::face1;
        break;
    case X2DIR:
        loc = Loci::face2;
        break;
    case X3DIR:
        loc = Loci::face3;
        break;
    }

    const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
    size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(NPRIM, n1);

    pmb->par_for_outer(string_format("uberkernel_x%d", dir), 3 * scratch_size_in_bytes, scratch_level, ks, ke, js, je,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int& k, const int& j) {
            ScratchPad2D<Real> ql(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> qr(member.team_scratch(scratch_level), NPRIM, n1);
            ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), NPRIM, n1);

            // get reconstructed state on faces
            // TODO switch statements are fast... right? This dispatch table is a pain, but so is another variant
            switch (recon) {
            case ReconstructionType::linear_mc:
                switch (dir) {
                case X1DIR:
                    Reconstruction::PiecewiseLinearX1(member, k, j, is, ie, P, ql, qr);
                    break;
                case X2DIR:
                    Reconstruction::PiecewiseLinearX2(member, k, j - 1, is, ie, P, ql, q_unused);
                    Reconstruction::PiecewiseLinearX2(member, k, j, is, ie, P, q_unused, qr);
                    break;
                case X3DIR:
                    Reconstruction::PiecewiseLinearX3(member, k - 1, j, is, ie, P, ql, q_unused);
                    Reconstruction::PiecewiseLinearX3(member, k, j, is, ie, P, q_unused, qr);
                    break;
                }
                break;
            case ReconstructionType::weno5:
                switch (dir) {
                case X1DIR:
                    Reconstruction::WENO5X1(member, k, j, is, ie, P, ql, qr);
                    break;
                case X2DIR:
                    Reconstruction::WENO5X2l(member, k, j - 1, is, ie, P, ql);
                    Reconstruction::WENO5X2r(member, k, j, is, ie, P, qr);
                    break;
                case X3DIR:
                    Reconstruction::WENO5X3l(member, k - 1, j, is, ie, P, ql);
                    Reconstruction::WENO5X3r(member, k, j, is, ie, P, qr);
                    break;
                }
                break;
            }

            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            parthenon::par_for_inner(member, is, ie, [&](const int i) {
                // Reverse the fluxes so that "left" and "right" are w.r.t. the *face*s
                Real pl[NPRIM], pr[NPRIM];
                PLOOP {
                    pl[p] = ql(p, i);
                    pr[p] = qr(p, i);
                }

                // LR -> flux
                FourVectors Dtmp;
                Real cmaxL, cmaxR, cminL, cminR;
                Real cmin, cmax, ctop_loc;

                Real fluxL[8], fluxR[8];
                Real Ul[8], Ur[8];

                // Left
                get_state(G, pl, k, j, i, loc, Dtmp);
                prim_to_flux(G, pl, Dtmp, eos, k, j, i, loc, 0, Ul); // dir==0 -> U instead of F in direction
                prim_to_flux(G, pl, Dtmp, eos, k, j, i, loc, dir, fluxL);
                mhd_vchar(G, pl, Dtmp, eos, k, j, i, loc, dir, cmaxL, cminL);

                // Right
                get_state(G, pr, k, j, i, loc, Dtmp);
                // Note: these three can be done simultaneously if we want to get real fancy
                prim_to_flux(G, pr, Dtmp, eos, k, j, i, loc, 0, Ur);
                prim_to_flux(G, pr, Dtmp, eos, k, j, i, loc, dir, fluxR);
                mhd_vchar(G, pr, Dtmp, eos, k, j, i, loc, dir, cmaxR, cminR);

                cmax = fabs(max(max(0.,  cmaxL),  cmaxR));
                cmin = fabs(max(max(0., -cminL), -cminR));
                ctop_loc = max(cmax, cmin);

                ctop(dir, k, j, i) = ctop_loc;
                PLOOP flux(p, k, j, i) = 0.5 * (fluxL[p] + fluxR[p] - ctop_loc * (Ur[p] - Ul[p]));
            });
        }
    );
    
    DelEOS(eos);

    FLAG(string_format("Finished recon and flux X%d", dir));
    return TaskStatus::complete;
}

/**
 * Add HARM source term to RHS
 */
TaskStatus AddSourceTerm(std::shared_ptr<Container<Real>>& rc, std::shared_ptr<Container<Real>>& dudt)
{
    FLAG("Adding source term");
    MeshBlock *pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GRCoordinates G = pmb->coords;

    // TODO *sigh*
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    // Unpack for kernel
    auto dUdt = dudt->Get("c.c.bulk.cons").data;

    pmb->par_for("source_term", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Calculate the source term and apply it in 1 go (since it's stencil-1)
            // TODO pass global dU so as not to copy?
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
Real EstimateTimestep(std::shared_ptr<Container<Real>>& rc)
{
    FLAG("Estimating timestep");
    MeshBlock *pmb = rc->pmy_block;
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto coords = pmb->coords;
    auto& ctop = rc->GetFace("f.f.bulk.ctop").data;

    // TODO is there a parthenon par_reduce yet?
    double ndt;
    Kokkos::Min<double> min_reducer(ndt);
    Kokkos::parallel_reduce("ndt_min", MDRangePolicy<Rank<3>>({ks, js, is}, {ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA_3D_REDUCE {
            double ndt_zone = 1 / (1 / (coords.dx1v(i) / ctop(1, k, j, i)) +
                                   1 / (coords.dx2v(j) / ctop(2, k, j, i)) +
                                   1 / (coords.dx3v(k) / ctop(3, k, j, i)));
            if (ndt_zone < local_result) local_result = ndt_zone;
        }
    , min_reducer);

    // Sometimes we come out with a silly timestep. Try to salvage it
    // TODO don't allow the *overall* timestep to be large, while allowing *blocks* to have large steps
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
