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

    // There are only 2 parameters related to fluid evolution:
    // 1. Fluid gamma for EOS (TODO separate EOS class to make this broader)
    // 2. Proportion of courant condition for timesteps
    double gamma = pin->GetOrAddReal("GRMHD", "gamma", 4. / 3);
    params.Add("gamma", gamma);
    double cfl = pin->GetOrAddReal("GRMHD", "cfl", 0.9);
    params.Add("cfl", cfl);
    // Starting/minimum timestep, if something about the sound speed goes wonky
    // Parthenon allows up to 2x higher dt per step, so it climbs to CFL quite fast
    double dt_min = pin->GetOrAddReal("parthenon/time", "dt_min", 1.e-5);
    params.Add("dt_min", dt_min);

    // We generally carry around the conserved versions of varialbles, treating them as the fundamental ones
    // However, since most analysis tooling expects the primitives, we *output* those.
    std::vector<int> s_vector({3});
    std::vector<int> s_fourvector({4});
    std::vector<int> s_fluid({NPRIM-3});
    std::vector<int> s_prims({NPRIM});
    // TODO arrange names/metadata to more accurately reflect e.g. variable locations and relations
    // for now cells are too darn easy and I don't use any fancy face-centered features
    Metadata m = Metadata({Metadata::Cell, Metadata::FillGhost, Metadata::Independent,
                    Metadata::Restart, Metadata::Conserved}, s_prims);
    fluid_state->AddField("c.c.bulk.cons", m, DerivedOwnership::shared);
    // initialize metadata the same but length s_vector
    // fluid_state->AddField("c.c.bulk.cons_B", m, DerivedOwnership::shared);

    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Intensive}, s_prims);
    fluid_state->AddField("c.c.bulk.prims", m, DerivedOwnership::shared);
    // metadata!
    // fluid_state->AddField("c.c.bulk.prims_B", m, DerivedOwnership::shared);

    // Max (i.e. positive) sound speed vector.  Easiest to keep here due to needing it for EstimateTimestep
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy});
    fluid_state->AddField("f.f.bulk.ctop", m, DerivedOwnership::unique);

    // TODO Add jcon as an output-only calculation, likely overriding MeshBlock::UserWorkBeforeOutput


    // TODO integer/flag fields in Parthenon
    // Then move pflag and fflag here

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

    // Get the primitives from our conserved versions
    // Note this covers ghost zones!  This is intentional, as primitives in
    // ghost zones are needed for reconstruction
    pmb->par_for("U_to_P", 0, n3-1, 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA_3D {
            pflag(k, j, i) = U_to_P(G, U, eos, k, j, i, Loci::center, P);
            fflag(k, j, i) = 0;
            fflag(k, j, i) |= fixup_ceiling(G, P, U, eos, k, j, i);
            fflag(k, j, i) |= fixup_floor(G, P, U, eos, k, j, i); // TODO this can generate a pflag
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
            fflag(k, j, i) |= fix_U_to_P(G, P, U, eos, pflag, k, j, i);
        }
    );
    FLAG("Fixed failed inversions");

#if DEBUG
    // TODO this does the calculation from primitives, but we could also
    // use the conserved vars
    double maxDivB = MaxDivB(rc);
    fprintf(stderr, "Maximum divB: %g\n", maxDivB);

    auto pflag_host = pflag.GetHostMirrorAndCopy();
    CountFFlags(pmb, pflag_host, IndexDomain::interior);
    CountPFlags(pmb, pflag_host, IndexDomain::interior);
#endif

    DelEOS(eos);
}

/**
 * Calculate the LLF flux in 1 direction
 * TODO this doesn't require full 3D pl/pr. Could merge recon and LR steps to save time *and* memory
 * TODO Make this async by returning TaskStatus::running
 */
TaskStatus CalculateFlux1(std::shared_ptr<Container<Real>>& rc)
{
    FLAG("Calculating flux 1");
    MeshBlock *pmb = rc->pmy_block;
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    GridVars pl("pl", NPRIM, n3, n2, n1);
    GridVars pr("pr", NPRIM, n3, n2, n1);
    GridVars F1 = rc->Get("c.c.bulk.cons").flux[X1DIR];

    // Reconstruct primitives at left and right sides of faces
    //Reconstruction::WENO5X1(rc, pl, pr);
    Reconstruction::LinearX1(rc, pl, pr);
    // Calculate flux from values at left & right of face
    LRToFlux(rc, pr, pl, 1, F1);

    FLAG("Calculated flux 1");

    return TaskStatus::complete;
}
/**
 * Calculate the LLF flux in 2 direction
 */
TaskStatus CalculateFlux2(std::shared_ptr<Container<Real>>& rc)
{
    FLAG("Calculating flux 2");
    MeshBlock *pmb = rc->pmy_block;
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    GridVars pl("pl", NPRIM, n3, n2, n1);
    GridVars pr("pr", NPRIM, n3, n2, n1);
    GridVars F2 = rc->Get("c.c.bulk.cons").flux[X2DIR];

    //Reconstruction::WENO5X2(rc, pl, pr);
    Reconstruction::LinearX2(rc, pl, pr);
    LRToFlux(rc, pr, pl, 2, F2);

    FLAG("Calculated flux 2");

    return TaskStatus::complete;
}
/**
 * Calculate the LLF flux in 3 direction
 */
TaskStatus CalculateFlux3(std::shared_ptr<Container<Real>>& rc)
{
    FLAG("Calculating flux 3");
    MeshBlock *pmb = rc->pmy_block;
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    GridVars pl("pl", NPRIM, n3, n2, n1);
    GridVars pr("pr", NPRIM, n3, n2, n1);
    GridVars F3 = rc->Get("c.c.bulk.cons").flux[X3DIR];

    //Reconstruction::WENO5X3(rc, pl, pr);
    Reconstruction::LinearX3(rc, pl, pr);
    LRToFlux(rc, pr, pl, 3, F3);

    FLAG("Calculated flux 3");

    return TaskStatus::complete;
}

/**
 * Add HARM source term to RHS
 */
TaskStatus SourceTerm(std::shared_ptr<Container<Real>>& rc, std::shared_ptr<Container<Real>>& dudt)
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
            Real dU[NPRIM] = {0};
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            get_fluid_source(G, P, Dtmp, eos, k, j, i, dU);

            PLOOP dUdt(p, k, j, i) += dU[p];
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

    // Sometimes this is called before ctop is initialized.  Catch weird dts and play it safe.
    if (ndt <= 0.0 || isnan(ndt) || ndt > 1) {
        cerr << "ndt was unsafe: " << ndt << "! Using dt_min" << std::endl;
        ndt = pmb->packages["GRMHD"]->Param<Real>("dt_min");
    } else {
        ndt *= pmb->packages["GRMHD"]->Param<Real>("cfl");
    }
    FLAG("Estimated");
    return ndt;
}



} // namespace GRMHD