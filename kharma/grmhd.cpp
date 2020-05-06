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

    // Coordinate options for building Grids per-mesh
    // It is probably easier to create a global CoordinateEmbedding pointer
    // TODO defaults should probably be KS
    std::string base_str = pin->GetOrAddString("coordinates", "base", "cartesian_minkowski");
    params.Add("c_base", base_str);
    std::string transform_str = pin->GetOrAddString("coordinates", "transform", "cartesian_null");
    params.Add("c_transform", transform_str);
    GReal startx1 = pin->GetOrAddReal("mesh", "x1min", 0);
    params.Add("c_startx1", startx1);
    GReal a = pin->GetOrAddReal("coordinates", "a", 0.0);
    params.Add("c_a", a);
    GReal hslope = pin->GetOrAddReal("coordinates", "hslope", 0.3);
    params.Add("c_hslope", hslope);
    GReal mks_smooth = pin->GetOrAddReal("coordinates", "mks_smooth", 0.5);
    params.Add("c_mks_smooth", mks_smooth);
    GReal poly_xt = pin->GetOrAddReal("coordinates", "poly_xt", 0.82);
    params.Add("c_poly_xt", poly_xt);
    GReal poly_alpha = pin->GetOrAddReal("coordinates", "poly_alpha", 14.0);
    params.Add("c_poly_alpha", poly_alpha);

    cerr << "GRMHD using " << base_str << " base coordiantes with " << transform_str << " transform" << std::endl;

    // We generally carry around the conserved versions of varialbles, treating them as the fundamental ones
    // However, since most analysis tooling expects the primitives, we *output* those.
    Metadata m;
    std::vector<int> s_vector({3});
    std::vector<int> s_fourvector({4});
    std::vector<int> s_fluid({NPRIM-3});
    std::vector<int> s_prims({NPRIM});
    // TODO arrange names/metadata to more accurately reflect e.g. variable locations and relations
    // for now cells are too darn easy and I don't use any fancy face-centered features
    m = Metadata({m.Cell, m.FillGhost, m.Independent, m.Restart, m.Conserved}, s_prims);
    fluid_state->AddField("c.c.bulk.cons", m, DerivedOwnership::shared);
    // initialize metadata the same but length s_vector
    // fluid_state->AddField("c.c.bulk.cons_B", m, DerivedOwnership::shared);

    m = Metadata({m.Cell, m.Derived, m.OneCopy, m.Intensive}, s_prims);
    fluid_state->AddField("c.c.bulk.prims", m, DerivedOwnership::shared);
    // metadata!
    // fluid_state->AddField("c.c.bulk.prims_B", m, DerivedOwnership::shared);

    // Max (i.e. positive) sound speed vector.  Easiest to keep here due to needing it for EstimateTimestep
    m = Metadata({m.Face, m.Derived, m.OneCopy});
    fluid_state->AddField("f.f.bulk.ctop", m, DerivedOwnership::unique);

    // TODO Add jcon as an output-only calculation, likely overriding MeshBlock::UserWorkBeforeOutput


    // Flags for patching inverter errors
    // TODO integer fields in Parthenon? Flags need to be sync'd with FillGhost,
    // and would be nice to include in dumps/restarts as well
    // m = Metadata({m.Cell, m.OneCopy, m.FillGhost, m.Independent, m.Restart});
    // fluid_state->AddField("bulk.pflag", m, DerivedOwnership::shared);

    fluid_state->FillDerived = GRMHD::FillDerived;
    fluid_state->CheckRefinement = nullptr;
    fluid_state->EstimateTimestep = GRMHD::EstimateTimestep;
    return fluid_state;
}

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived"
 * Derived variables are updated before output and during the step so that we can work with them
 */
void FillDerived(Container<Real>& rc)
{
    FLAG("Filling Derived");
    MeshBlock *pmb = rc.pmy_block;

    GridVars U = rc.Get("c.c.bulk.cons").data;
    GridVars P = rc.Get("c.c.bulk.prims").data;

    //GridVars pflag = rc.Get("bulk.pflag").data; // TODO remember to switch to ints
    GridInt pflag("pflag", pmb->ncells3, pmb->ncells2, pmb->ncells1);

    // TODO how do I carry this around per-block and just update it when needed?
    // (or if not the Grid, then at least a coordinate system and EOS...)
    Grid G(pmb);
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = new GammaLaw(gamma);

    // Get the primitives from our conserved versions
    // Note this covers ghost zones!  This is intentional, as primitives in
    // ghost zones are needed for reconstruction
    pmb->par_for("U_to_P", 0, pmb->ncells3-1, 0, pmb->ncells2-1, 0, pmb->ncells1-1,
        KOKKOS_LAMBDA_3D {
            pflag(k, j, i) = U_to_P(G, U, eos, k, j, i, Loci::center, P);
        }
    );
    FLAG("Filled");

    // We expect primitives all the way out to 3 ghost zones on all sides.  But we can only fix primitives with their neighbors.
    // This may actually mean we require the 4 ghost zones Parthenon "wants" us to have, if we need to use only fixed zones.
    // TODO alternatively do a bounds check in fix_U_to_P
    ClearCorners(pmb, pflag); // Don't use zones in physical corners. TODO persist this?
    pmb->par_for("fix_U_to_P", 1, pmb->ncells3-2, 1, pmb->ncells2-2, 1, pmb->ncells1-2,
        KOKKOS_LAMBDA_3D {
            fix_U_to_P(G, P, U, eos, pflag, k, j, i); // Returns fflag, whether any floors were hit.  TODO record...
        }
    );
    FLAG("Corrected");

#if DEBUG
    // TODO this is actually a lot easier to calculate in the conserved vars,
    // due to not needing metric.  But this was the routine I could copy.
    double maxDivB = MaxDivB(rc);
    fprintf(stderr, "Maximum divB: %g\n", maxDivB);

    count_print_pflags(pmb, pflag);
#endif
}

/**
 * Calculate the LLF fluxes
 */
TaskStatus CalculateFluxes(Container<Real>& rc)
{
    FLAG("Calculating Fluxes");
    MeshBlock *pmb = rc.pmy_block;
    GridVars pl("pl", NPRIM, pmb->ncells3, pmb->ncells2, pmb->ncells1);
    GridVars pr("pr", NPRIM, pmb->ncells3, pmb->ncells2, pmb->ncells1);
    GridVars F1 = rc.Get("c.c.bulk.cons").flux[0];
    GridVars F2 = rc.Get("c.c.bulk.cons").flux[1];
    GridVars F3 = rc.Get("c.c.bulk.cons").flux[2];

    // Reconstruct primitives at left and right sides of faces
    WENO5X1(rc, pl, pr);
    // Calculate flux from values at left & right of face
    LRToFlux(rc, pr, pl, 1, F1);

    WENO5X2(rc, pl, pr);
    LRToFlux(rc, pr, pl, 2, F2);

    WENO5X3(rc, pl, pr);
    LRToFlux(rc, pr, pl, 3, F3);

    // TODO necessary?  Definitely messes with Bondi problem currently, needs nuance
    //FixFlux(rc);

    // Constrained transport for B must be applied after everything, including fixing boundary fluxes
    FluxCT(rc);
    FLAG("Calculated fluxes");

    return TaskStatus::complete;
}

/**
 * Add HARM source term to RHS
 */
TaskStatus SourceTerm(Container<Real>& rc, Container<Real>& dudt)
{
    FLAG("Adding source term");
    MeshBlock *pmb = rc.pmy_block;
    auto is = pmb->is, js = pmb->js, ks = pmb->ks;
    auto ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    GridVars P = rc.Get("c.c.bulk.prims").data;

    // TODO *sigh*
    Grid G(pmb);
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = new GammaLaw(gamma);

    // Unpack a bunch of variables for the kernel below
    auto dUdt = dudt.Get("c.c.bulk.cons").data;

    pmb->par_for("source_term", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Calculate the source term and apply it in 1 go (since it's stencil-1)
            FourVectors Dtmp;
            Real dU[NPRIM] = {0};
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            get_fluid_source(G, P, Dtmp, eos, k, j, i, dU);

            PLOOP dUdt(p, k, j, i) += dU[p];
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
Real EstimateTimestep(Container<Real>& rc)
{
    FLAG("Estimating timestep");
    MeshBlock *pmb = rc.pmy_block;
    auto is = pmb->is, js = pmb->js, ks = pmb->ks;
    auto ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    auto dx1v = pmb->pcoord->dx1v;
    auto dx2v = pmb->pcoord->dx2v;
    auto dx3v = pmb->pcoord->dx3v;
    auto& ctop = rc.GetFace("f.f.bulk.ctop").data;

    // TODO is there a parthenon par_reduce yet?
    double ndt;
    Kokkos::Min<double> min_reducer(ndt);
    Kokkos::parallel_reduce("ndt_min", MDRangePolicy<Rank<3>>({ks, js, is}, {ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA_3D_REDUCE {
            double ndt_zone = 1 / (1 / (dx1v(i) / ctop(1, k, j, i)) +
                                   1 / (dx2v(j) / ctop(2, k, j, i)) +
                                   1 / (dx3v(k) / ctop(3, k, j, i)));
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
    //fprintf(stderr, "dt = %g\n", ndt);
    return ndt;
}



} // namespace GRMHD