// Dispatch for all problems which can be generated

#include "bvals/boundary_conditions.hpp"
#include "mesh/mesh.hpp"

#include "grid.hpp"
#include "phys.hpp"

#include "mhdmodes.hpp"
#include "bondi.hpp"

using namespace parthenon;

/**
 * Override a parthenon method to generate the section of a problem redisiding in a given mesh block.
 *
 * This takes care of calling a problem initialization method with the correct parameters, then initializing the
 * conserved versions of variables from the problem's primitives, and syncing ghost zones for the first time.
 */
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
    auto pmb = this;
    auto rc = pmb->real_containers.Get();
    GridVars P = rc.Get("c.c.bulk.prims").data;
    GridVars U = rc.Get("c.c.bulk.cons").data;

    Grid G(pmb);
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = new GammaLaw(gamma);

    auto prob = pin->GetString("job", "problem_id"); // Required
    if (prob == "mhdmodes") {
        int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
        int dir = pin->GetOrAddInteger("mhdmodes", "dir", 0);

        InitializeMHDModes(pmb, G, P, nmode, dir);

    } else if (prob == "bondi") {
        Real mdot = pin->GetOrAddInteger("bondi", "mdot", 1.0);
        Real rs = pin->GetOrAddInteger("bondi", "rs", 8.0);
        // Add these to package properties, since they continue to be needed on boundaries
        pmb->packages["GRMHD"]->AddParam<Real>("mdot", mdot);
        pmb->packages["GRMHD"]->AddParam<Real>("rs", rs);

        InitializeBondi(pmb, G, P, eos, mdot, rs);

    } else if (prob == "torus") {

    } else if (prob == "legacy_restart") {

    }

    // Initialize U
    pmb->par_for("first_flux", 0, pmb->ncells3-1, 0, pmb->ncells2-1, 0, pmb->ncells1-1,
        KOKKOS_LAMBDA_3D {
            FourVectors Dtmp;
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
        }
    );

    // TODO only define prims on physical zones, then sync boundaries here. Doesn't work for some reason...
    // rc.SendBoundaryBuffers();
    // rc.ReceiveBoundaryBuffers();
    // rc.SetBoundaries();
    // rc.ClearBoundary(BoundaryCommSubset::all);
    // pmb->pbval->ProlongateBoundaries(0.0, 0.0);
    // ApplyBoundaryConditions(rc);

    // Then re-calculate primitives
    // FillDerivedVariables::FillDerived(rc);

    FLAG("Initialized problem"); // TODO this called in every meshblock.  Avoid the spam somehow
}