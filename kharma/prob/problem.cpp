// Dispatch for all problems which can be generated

#include "bvals/boundary_conditions.hpp"
#include "mesh/mesh.hpp"

#include "fixup.hpp"
#include "floors.hpp"
#include "gr_coordinates.hpp"
#include "phys.hpp"

// Problem initialization headers
#include "mhdmodes.hpp"
#include "bondi.hpp"
#include "fm_torus.hpp"
#include "seed_B.hpp"
//#include "bh_flux.hpp"

using namespace parthenon;

/**
 * Generate the initial condition on a meshblock
 *
 * This takes care of calling a problem initialization method with the correct parameters, then initializing the
 * conserved versions of variables from the problem's primitives, and syncing ghost zones for the first time.
 */
void InitializeProblem(ParameterInput *pin, MeshBlock *pmb)
{
    auto& rc = pmb->real_containers.Get();
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;

    GRCoordinates G = pmb->coords;
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    auto prob = pin->GetString("parthenon/job", "problem_id"); // Required parameter
    if (prob == "mhdmodes") {
        int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
        int dir = pin->GetOrAddInteger("mhdmodes", "dir", 0);

        Real tf = InitializeMHDModes(pmb, G, P, nmode, dir);
        pin->SetReal("parthenon/time", "tlim", tf);

    } else if (prob == "bondi") {
        Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
        Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);
        // Add these to package properties, since they continue to be needed on boundaries
        if(! (pmb->packages["GRMHD"]->AllParams().hasKey("mdot")))
            pmb->packages["GRMHD"]->AddParam<Real>("mdot", mdot);
        if(! (pmb->packages["GRMHD"]->AllParams().hasKey("rs")))
            pmb->packages["GRMHD"]->AddParam<Real>("rs", rs);

        InitializeBondi(pmb, G, P, eos, mdot, rs);

    } else if (prob == "torus") {
        Real rin = pin->GetOrAddReal("torus", "rin", 6.0);
        Real rmax = pin->GetOrAddReal("torus", "rmax", 12.0);
        FLAG("Initializing torus");
        InitializeFMTorus(pmb, G, P, eos, rin, rmax);

    } else if (prob == "legacy_restart") {

    }

    // TODO namespace these outside "torus"
    if (prob != "legacy_restart") {
        // TODO randomness is deterministic per-run I think (at least on OpenMP),
        // but not per-mesh-geometry/MPI geometry
        Real u_jitter = pin->GetOrAddReal("torus", "u_jitter", 0.0);
        int rng_seed = pin->GetOrAddInteger("torus", "rng_seed", 31337);
        if (u_jitter > 0.0) {
            FLAG("Applying U perturbation");
            PerturbU(pmb, P, u_jitter, rng_seed + pmb->gid);
        }

        Real rin = pin->GetOrAddReal("torus", "rin", 6.0); // Needed for MAD initializations
        Real min_rho_q = pin->GetOrAddReal("torus", "min_rho_q", 0.2);
        std::string b_field_type = pin->GetOrAddString("torus", "b_field_type", "none");
        if (b_field_type != "none") {
            FLAG("Seeding magnetic field");
            SeedBField(pmb, G, P, rin, min_rho_q, b_field_type);
        }
        
        Real BHflux = pin->GetOrAddReal("torus", "bhflux", 0.0);
        if (BHflux > 0.) {
            //SeedBHFlux(pmb, G, P, BHflux);
        }
    }

    // Initialize U
    FLAG("First P->U");
    pmb->par_for("first_U", 0, n3-1, 0, n2-1, 0, n1-1,
        KOKKOS_LAMBDA_3D {
            FourVectors Dtmp;
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
        }
    );

    // Make sure any zero zones get floored before beginning driver
    FLAG("First Floors");
    ApplyFloors(rc);

    DelEOS(eos);

    FLAG("Initialized Problem"); // TODO this called in every meshblock.  Avoid the spam somehow
}