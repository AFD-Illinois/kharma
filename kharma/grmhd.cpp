// Functions defining the evolution of GRMHD fluid

#include <memory>

// Until Parthenon gets a reduce()
#include "Kokkos_Core.hpp"

#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"

#include "coordinate_embedding.hpp"
#include "coordinate_systems.hpp"
#include "decs.hpp"
#include "fluxes.hpp"
#include "grmhd.hpp"
#include "phys.hpp"
#include "U_to_P.hpp"

using namespace parthenon;

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

    // HARM is non-negotiably RK2 a.k.a. predictor-corrector for now
    params.Add("order", 2);

    // There are only 2 parameters related to fluid evolution:
    // 1. Fluid gamma for EOS (TODO separate EOS class to make this broader)
    // 2. Proportion of courant condition for timesteps
    double gamma = pin->GetOrAddReal("Hydro", "gamma", 4. / 3);
    params.Add("gamma", gamma);
    double cfl = pin->GetOrAddReal("Hydro", "cfl", 0.9);
    params.Add("cfl", cfl);

    // We generally carry around the conserved versions of varialbles, treating them as the fundamental ones
    // However, since most analysis tooling expects the primitives, we *output* those.
    Metadata m;
    std::vector<int> s_vector({3});
    std::vector<int> s_fluid({NPRIM-3});
    std::vector<int> s_prims({NPRIM});
    m = Metadata({m.Cell, m.Intensive, m.Vector, m.OneCopy, m.FillGhost, m.Conserved}, s_prims);
    fluid_state->AddField("c.c.bulk.cons", m, DerivedOwnership::shared);
    // initialize metadata the same but length s_vector
    // fluid_state->AddField("c.c.bulk.cons_B", m, DerivedOwnership::shared);

    m = Metadata({m.Cell, m.Intensive, m.Derived, m.Vector, m.OneCopy, m.Graphics}, s_prims);
    fluid_state->AddField("c.c.bulk.prims", m, DerivedOwnership::shared);
    // metadata!
    // fluid_state->AddField("c.c.bulk.prims_B", m, DerivedOwnership::shared);

    // Sound speed.  Easiest to keep here
    m = Metadata({m.Face, m.Intensive, m.Derived, m.Vector, m.OneCopy}, s_vector);
    fluid_state->AddField("f.f.bulk.ctop", m, DerivedOwnership::unique);

    // TODO Probably jcon will go here too eventually. Does Parthenon have output-only calculations?

    // Flags for patching inverter errors
    // TODO integer fields?  Some better way to do boundaries?
    //m = Metadata({m.Cell, m.Intensive, m.Derived, m.OneCopy});
    //fluid_state->AddField("bulk.pflag", m, DerivedOwnership::shared);

    fluid_state->FillDerived = GRMHD::FillDerived;
    fluid_state->CheckRefinement = nullptr;
    fluid_state->EstimateTimestep = GRMHD::EstimateTimestep;
    return fluid_state;
}

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived"
 * Derived variables are updated before output and during the step so that we can work with them
 */
void FillDerived(Container<Real>& rc) {
    MeshBlock *pmb = rc.pmy_block;
    Coordinates *pcoord = pmb->pcoord.get();
    auto is = pmb->is, js = pmb->js, ks = pmb->ks;
    auto ie = pmb->ie, je = pmb->je, ke = pmb->ke;

    auto U = rc.Get("c.c.bulk.cons").data;
    auto P = rc.Get("c.c.bulk.prims").data;

    //auto pflag = rc.Get("bulk.pflag"); // TODO this will need to be sync'd!
    ParArrayND<int> pflag("pflag", pmb->ncells1, pmb->ncells2, pmb->ncells3);

    // TODO how do I carry this around per-block and just update it when needed?
    // (or if not the Grid, then at least a coordinate system and EOS...)
    Grid G(pmb);
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("cfl");
    EOS* eos = new GammaLaw(gamma);

    // Get the primitives from our conserved versions
    pmb->par_for("cons_to_prim", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            pflag(i, j, k) = U_to_P(G, U, eos, i, j, k, Loci::center, P);
        }
    );
    FLAG("Get primitives");

    // TODO if DEBUG check pflags
}

/**
 * Calculate dU/dt from a set of fluxes.
 * Needs prims and cons.flux components filled, by FillDerived and CalculateFluxes respectively
 *
 * @param rc is the current stage's container
 * @param base is the base container containing the global dUdt term
 */
TaskStatus ApplyFluxes(Container<Real>& rc, Container<Real>& base) {
    MeshBlock *pmb = rc.pmy_block;
    Coordinates *pcoord = pmb->pcoord.get();
    auto is = pmb->is, js = pmb->js, ks = pmb->ks;
    auto ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    auto ni = ie-is, nj = je-js, nk = ke-ks;
    GridVars U = rc.Get("c.c.bulk.cons").data;
    GridVars F1 = rc.Get("c.c.bulk.cons").flux[0];
    GridVars F2 = rc.Get("c.c.bulk.cons").flux[1];
    GridVars F3 = rc.Get("c.c.bulk.cons").flux[2];
    GridVars P = rc.Get("c.c.bulk.prims").data;

    // TODO *sigh*
    Grid G(pmb);
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("cfl");
    EOS* eos = new GammaLaw(gamma);

    // Unpack a bunch of variables for the kernel below
    auto dUdt = base.Get("dUdt").data;
    // We don't otherwise support irregular grids...
    auto dx1v = pcoord->dx1v;
    auto dx2v = pcoord->dx2v;
    auto dx3v = pcoord->dx3v;
    auto dt = pmb->pmy_mesh->dt;

    pmb->par_for("uber_diff", is, ie, js, je, ks, ke,
        KOKKOS_LAMBDA_3D {
            // Calculate the source term and apply it in 1 go (since it's stencil-1)
            FourVectors Dtmp;
            Real dU[NPRIM] = {0};
            get_state(G, P, i, j, k, Loci::center, Dtmp);
            //get_fluid_source(G, P, Dtmp, eos, i, j, k, dU);

            for(int p=0; p < NPRIM; ++p)
                dUdt(i, j, k, p) += (F1(i, j, k, p) - F1(i+1, j, k, p)) / dx1v(i,j,k) +
                                  (F2(i, j, k, p) - F2(i, j+1, k, p)) / dx2v(i,j,k) +
                                  (F3(i, j, k, p) - F3(i, j, k+1, p)) / dx3v(i,j,k) +
                                  dU[p];
        }
    );
    FLAG("Uber finite diff");

    return TaskStatus::complete;
}

/**
 * Returns the minimum CFL timestep among all zones in the block,
 * multiplied by a proportion "cfl" for safety.
 *
 * TODO pretty sure this needs to be added to if there are new packages e.g. bhlight
 */
Real EstimateTimestep(Container<Real>& rc)
{
    MeshBlock *pmb = rc.pmy_block;
    auto is = pmb->is, js = pmb->js, ks = pmb->ks;
    auto ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    auto dx1v = pmb->pcoord->dx1v;
    auto dx2v = pmb->pcoord->dx2v;
    auto dx3v = pmb->pcoord->dx3v;
    auto ctop = rc.Get("f.f.bulk.ctop").data;

    double ndt;
    Kokkos::Min<double> min_reducer(ndt);
    Kokkos::parallel_reduce("ndt_min", MDRangePolicy<Rank<3>>({is, js, ks}, {ie, je, ke}),
        KOKKOS_LAMBDA (const int &i, const int &j, const int &k, double &local_min) {
            double ndt_zone = 1 / (1 / (dx1v(i,j,k) / ctop(i, j, k, 0)) +
                                   1 / (dx2v(i,j,k) / ctop(i, j, k, 1)) +
                                   1 / (dx3v(i,j,k) / ctop(i, j, k, 2)));
            if (ndt_zone < local_min) local_min = ndt_zone;
        }
    , min_reducer);

    // TODO MPI reduce, record zone and/or coordinate of the true minimum

    return ndt * pmb->packages["GRMHD"]->Param<Real>("cfl");
}



} // namespace GRMHD