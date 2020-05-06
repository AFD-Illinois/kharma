// Dispatch for all problems which can be generated

#include "bvals/boundary_conditions.hpp"
#include "mesh/mesh.hpp"

#include "grid.hpp"
#include "phys.hpp"

// Problem initialization headers
#include "mhdmodes.hpp"
#include "bondi.hpp"
#include "fm_torus.hpp"
#include "seed_B.hpp"

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

    auto prob = pin->GetString("parthenon/job", "problem_id"); // Required
    if (prob == "mhdmodes") {
        int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
        int dir = pin->GetOrAddInteger("mhdmodes", "dir", 0);

        Real tf = InitializeMHDModes(pmb, G, P, nmode, dir);
        pin->SetReal("parthenon/time", "tlim", tf); // TODO if this doesn't work push it upstream as an issue

    } else if (prob == "bondi") {
        Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
        Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);
        // Add these to package properties, since they continue to be needed on boundaries
        pmb->packages["GRMHD"]->AddParam<Real>("mdot", mdot);
        pmb->packages["GRMHD"]->AddParam<Real>("rs", rs);

        InitializeBondi(pmb, G, P, eos, mdot, rs);

    } else if (prob == "torus") {
        Real rin = pin->GetOrAddReal("torus", "rin", 6.0);
        Real rmax = pin->GetOrAddReal("torus", "rmax", 12.0);
        InitializeFMTorus(pmb, G, P, eos, rin, rmax);

        // TODO these are actually kind of separate, split them out.
        Real u_jitter = pin->GetOrAddReal("torus", "u_jitter", 0.1);
        int rng_seed = pin->GetOrAddInteger("torus", "rng_seed", 31337);
        PerturbU(pmb, P, u_jitter, rng_seed);

        Real beta = pin->GetOrAddReal("torus", "beta", 100.0);
        Real min_rho_q = pin->GetOrAddReal("torus", "min_rho_q", 0.2);
        std::string b_field_type = pin->GetOrAddString("torus", "b_field_type", "sane");
        SeedBField(pmb, G, P, rin, min_rho_q, b_field_type);

    } else if (prob == "legacy_restart") {

    }
    // TODO if BHflux seed here

    // Initialize U
    pmb->par_for("first_U", 0, pmb->ncells3-1, 0, pmb->ncells2-1, 0, pmb->ncells1-1,
        KOKKOS_LAMBDA_3D {
            FourVectors Dtmp;
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
        }
    );

    FLAG("Initialized MeshBlock"); // TODO this called in every meshblock.  Avoid the spam somehow
}