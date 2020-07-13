// Dispatch for all problems which can be generated

#include "problem.hpp"

#include "boundaries.hpp"
#include "debug.hpp"
#include "fixup.hpp"
#include "floors.hpp"
#include "gr_coordinates.hpp"
#include "phys.hpp"

// Problem initialization headers
#include "bondi.hpp"
#include "fm_torus.hpp"
#include "iharm_restart.hpp"
#include "mhdmodes.hpp"
#include "seed_B.hpp"

#include "bvals/boundary_conditions.hpp"
#include "mesh/mesh.hpp"

using namespace parthenon;

void InitializeMesh(ParameterInput *pin, Mesh *pmesh)
{
    // TODO this is *instead* of defining MeshBlock::ProblemGenerator,
    // which means that initial auto-refinement will *NOT* work.
    // we're a ways from AMR yet so we'll cross that bridge etc.

    // Note this first step spits out valid P and U on *physical* zones
    // TODO make a list of pmb pointers like Initialize does
    MeshBlock *pmb = pmesh->pblock;
    while (pmb != nullptr) {
        // Initialize the base container with problem values
        // This could be switched into task format without too much issue
        auto& rc = pmb->real_containers.Get();
        InitializeProblem(rc, pin);
        pmb = pmb->next;
    }
    FLAG("Initialized Fluid");

    // Add the field for torus problems as a second pass
    // Preserves P==U and ends with all physical zones fully defined
    if (pin->GetOrAddString("b_field", "type", "none") != "none") {
        // Calculating B has a stencil outside physical zones
        FLAG("Extra boundary sync for B");
        SyncAllBounds(pmesh);
        //pmesh->Initialize(0, pin);

        FLAG("Seeding magnetic field");
        // Seed the magnetic field and find the minimum beta
        Real beta_min = 1e100;
        pmb = pmesh->pblock;
        while (pmb != nullptr) {
            auto& rc = pmb->real_containers.Get();
            SeedBField(rc, pin);
        
            // TODO add this after normalization instead?
            // TODO options to add to horizon/renormalize B during run?
            Real BHflux = pin->GetOrAddReal("b_field", "bhflux", 0.0);
            if (BHflux > 0.) {
                //SeedBHFlux(rc, BHflux);
            }

            Real beta_local = GetLocalBetaMin(rc);
            if(beta_local < beta_min) beta_min = beta_local;
            pmb = pmb->next;
        }
        beta_min = MPIMin(beta_min);

        // Then normalize B by sqrt(beta/beta_min)
        FLAG("Normalizing magnetic field");
        Real beta = pin->GetOrAddReal("b_field", "beta_min", 100.);
        Real factor = sqrt(beta/beta_min);
        pmb = pmesh->pblock;
        while (pmb != nullptr) {
            auto& rc = pmb->real_containers.Get();
            NormalizeBField(rc, factor);
            pmb = pmb->next;
        }
    }
    FLAG("Added B Field");

    // Sync to fill the ghost zones
    FLAG("Boundary sync");
    SyncAllBounds(pmesh);
    //pmesh->Initialize(0, pin);

    //Diagnostic(rc, IndexDomain::entire);


    FLAG("Initialized Mesh");
}

TaskStatus InitializeProblem(std::shared_ptr<Container<Real>>& rc, ParameterInput *pin)
{
    MeshBlock *pmb = rc->pmy_block;
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;

    GRCoordinates G = pmb->coords;
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    auto prob = pin->GetString("parthenon/job", "problem_id"); // Required parameter
    if (prob == "mhdmodes") {
        int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
        int dir = pin->GetOrAddInteger("mhdmodes", "dir", 0);

        double tf = InitializeMHDModes(pmb, G, P, nmode, dir);
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

    } else if (prob == "iharm_restart") {
        auto fname = pin->GetString("iharm_restart", "fname"); // Require this, don't guess
        bool use_tf = pin->GetOrAddBoolean("iharm_restart", "use_tf", false);
        double tf = ReadIharmRestart(pmb, G, P, fname);
        if (use_tf) {
            pin->SetReal("parthenon/time", "tlim", tf);
        }
    }

    // TODO namespace this outside "torus," it could be added to anything
    Real u_jitter = pin->GetOrAddReal("torus", "u_jitter", 0.0);
    int rng_seed = pin->GetOrAddInteger("torus", "rng_seed", 31337);
    if (u_jitter > 0.0) {
        FLAG("Applying U perturbation");
        PerturbU(pmb, P, u_jitter, rng_seed + pmb->gid);
    }

    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    // Initialize U
    FLAG("First P->U");
    pmb->par_for("first_U", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            FourVectors Dtmp;
            get_state(G, P, k, j, i, Loci::center, Dtmp);
            prim_to_flux(G, P, Dtmp, eos, k, j, i, Loci::center, 0, U);
        }
    );

    //Diagnostic(rc, IndexDomain::entire);

    // Apply any floors. Floors preserve P<->U so why not test that?
    FLAG("First Floors");
    ApplyFloors(rc);

    //Diagnostic(rc, IndexDomain::entire);


    DelEOS(eos);
    FLAG("Initialized Block");
    return TaskStatus::complete;
}

void SyncAllBounds(Mesh *pmesh)
{
    // Update ghost cells. Only performed on U
    MeshBlock *pmb = pmesh->pblock;
    while (pmb != nullptr) {
        auto& rc = pmb->real_containers.Get();
        rc->ClearBoundary(BoundaryCommSubset::mesh_init);
        rc->StartReceiving(BoundaryCommSubset::mesh_init);
        rc->SendBoundaryBuffers();

        pmb = pmb->next;
    }

    pmb = pmesh->pblock;
    while (pmb != nullptr) {
        auto& rc = pmb->real_containers.Get();
        rc->ReceiveAndSetBoundariesWithWait();
        rc->ClearBoundary(BoundaryCommSubset::mesh_init);
        //pmb->pbval->ProlongateBoundaries(0.0, 0.0);

        // Physical boundary conditions
        parthenon::ApplyBoundaryConditions(rc);
        ApplyCustomBoundaries(rc);

        // Fill P again, including ghost zones
        parthenon::FillDerivedVariables::FillDerived(rc);

        pmb = pmb->next;
    }
}