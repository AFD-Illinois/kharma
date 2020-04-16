// Functions defining the evolution of GRMHD fluid

#include <memory>

#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"

#include "coordinate_embedding.hpp"
#include "coordinate_systems.hpp"
#include "decs.hpp"
#include "grmhd.hpp"
#include "U_to_P.hpp"

using namespace parthenon;

namespace GRMHD
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin)
{
    auto fluid_state = std::make_shared<StateDescriptor>("GRMHD");
    Params &params = fluid_state->AllParams();

    // HARM is non-negotiably RK2 a.k.a. predictor-corrector for now
    params.Add("order", 2);

    // Only 2 fluid-related parameters:
    // 1. Fluid gamma for EOS (TODO separate EOS class to make this broader)
    // 2. Proportion of courant condition for timesteps
    double gamma = pin->GetOrAddReal("Hydro", "gamma", 5. / 3);
    params.Add("gamma", gamma);
    double cfl = pin->GetOrAddReal("Hydro", "cfl", 0.9);
    params.Add("cfl", cfl);

    // We generally carry around the conserved versions of varialbles, treating them as the fundamental ones
    // However, since most analysis tooling expects the primitives, we *output* those.
    Metadata m;
    std::vector<int> s_vector({3}); // TODO that's what these are right?
    std::vector<int> s_fluid({NPRIM-3});
    std::vector<int> s_prims({NPRIM});
    m = Metadata({m.Cell, m.Intensive, m.Vector, m.OneCopy, m.FillGhost, m.Conserved}, s_prims);
    fluid_state->AddField("c.c.bulk.cons", m, DerivedOwnership::shared);
    // initialize metadata the same but length s_vector
    // fluid_state->AddField("c.c.bulk.cons_B", m, DerivedOwnership::shared);

    m = Metadata({m.Cell, m.Intensive, m.Derived, m.Vector, m.OneCopy, m.Graphics}, s_prims);
    fluid_state->AddField("c.c.bulk.prims", m, DerivedOwnership::shared);
    // m = Metadata({m.Cell, m.Intensive, m.Derived, m.Vector, m.OneCopy}, s_vector);
    // fluid_state->AddField("c.c.bulk.prims_B", m, DerivedOwnership::shared);

    // Flags for patching inverter errors
    // TODO integer fields?  Some better way to do boundaries?
    //m = Metadata({m.Cell, m.Intensive, m.Derived, m.OneCopy});
    //fluid_state->AddField("bulk.pflag", m, DerivedOwnership::shared);


    fluid_state->FillDerived = nullptr; // TODO make this U_to_P
    fluid_state->CheckRefinement = nullptr;
    fluid_state->EstimateTimestep = GRMHD::EstimateTimestep;
    return fluid_state;
}

/**
 * Get the primitive variables, which in Parthenon's nomenclature are "derived"
 * Derived variables are updated before output and during the step so that we can work with them
 */
TaskStatus FillDerived(Container<Real>& rc) {
    MeshBlock *pmb = rc.pmy_block;
    Coordinates *pcoord = pmb->pcoord.get();
    int is = pmb->is, js = pmb->js, ks = pmb->ks;
    int ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    int ni = ie-is, nj = je-js, nk = ke-ks;
    CellVariable<Real>& Ui = rc.Get("c.c.bulk.cons");
    CellVariable<Real>& Pi = rc.Get("c.c.bulk.prims");

    //CellVariable<int>& pflag = rc.Get("bulk.pflag");
    ParArrayND<int> pflag("pflag", ni, nj, nk);

    CartMinkowskiCoords base_in;
    CartNullTransform transform_in;
    CoordinateEmbedding C(base_in, transform_in);

    // Get the primitives from our conserved versions
    pmb->par_for("cons_to_prim", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            pflag(i, j, k) = U_to_P(C, Ui, i, j, k, 0, Pi);
        }
    );
    FLAG("Get primitives");
}

TaskStatus AdvanceFluid(Container<Real>& rc) {
    MeshBlock *pmb = rc.pmy_block;
    Coordinates *pcoord = pmb->pcoord.get();
    int is = pmb->is, js = pmb->js, ks = pmb->ks;
    int ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    int ni = ie-is, nj = je-js, nk = ke-ks;
    CellVariable<Real>& Ui = rc.Get("c.c.bulk.cons");
    CellVariable<Real>& Pi = rc.Get("c.c.bulk.prims");

    CartMinkowskiCoords base_in;
    CartNullTransform transform_in;
    CoordinateEmbedding C(base_in, transform_in);

    // Get the fluxes in each direction on the zone faces
    CalculateFluxes(pmb);
    FLAG("Get flux");

    // Fix boundary fluxes
//    fix_flux(F1, F2, F3);
//    FLAG("Fix flux");

    // Constrained transport for B
    // flux_ct(pmb);
    // FLAG("Flux CT");

    pmb->par_for("uber_diff", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            FourVectors Dtmp;

            get_state(G, Pi, i, j, k, Loci::center, Dtmp);
            prim_to_flux(G, Pi, Dtmp, eos, i, j, k, Loci::center, 0, Ui); // (i, j, k, p)

            if (Ps != Pi)
                get_state(G, Ps, i, j, k, Loci::center, Dtmp);
            get_fluid_source(G, Ps, Dtmp, eos, i, j, k, dU);

            for(int p=0; p < 5; ++p)
                Uf(i, j, k, p) = Ui(i, j, k, p) +
                                    pmb->pmy_mesh->dt * ((F1(i, j, k, p) - F1(i+1, j, k, p)) / pcoord->dx1v(i,j,k) +
                                        (F2(i, j, k, p) - F2(i, j+1, k, p)) / pcoord->dx2v(i,j,k) +
                                        (F3(i, j, k, p) - F3(i, j, k+1, p)) / pcoord->dx3v(i,j,k) +
                                        dU(i, j, k, p));
        }
    );
    FLAG("Uber finite diff");

    return TaskStatus::complete;
}

Real EstimateTimestep(Container<Real>& rc)
{
    return 0.017;
}

/**
 * Returns the minimum CFL timestep among all zones in the block,
 * multiplied by a proportion "cfl" for safety.
 * 
 * TODO does this need to be amended for e.g. bhlight?
 */
// Real EstimateTimestep(Container<Real>& rc)
// {
//     MeshBlock *pmb = rc.pmy_block;
//     Coordinates pcoords = pmb->pcoords.get();

//     double ndt;
//     Kokkos::Min<double> min_reducer(ndt);
//     Kokkos::parallel_reduce("ndt_min", G.bulk_ng(),
//         KOKKOS_LAMBDA (const int &i, const int &j, const int &k, double &local_min) {
//             double ndt_zone = 1 / (1 / (pcoords->dx1 / ctop(i, j, k, 1)) +
//                                  1 / (pcoords->dx2 / ctop(i, j, k, 2)) +
//                                  1 / (pcoords->dx3 / ctop(i, j, k, 3)));
//             if (ndt_zone < local_min) local_min = ndt_zone;
//         }
//     , min_reducer);

//     // TODO MPI, record & optionally print zone of the minimum

//     return ndt * pmb->packages["GRMHD"]->Param<Real>("cfl");
// }



} // namespace GRMHD