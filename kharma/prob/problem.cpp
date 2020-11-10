/* 
 *  File: problem.cpp
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
#include "orszag_tang.hpp"
#include "seed_B.hpp"

#include "bvals/boundary_conditions.hpp"
#include "mesh/mesh.hpp"

using namespace parthenon;

// Dispatch for all problems which can be generated

void InitializeMesh(ParameterInput *pin, Mesh *pmesh)
{
    // TODO this is *instead* of defining MeshBlock::ProblemGenerator,
    // which means that initial auto-refinement will *NOT* work.
    // we're a ways from AMR yet so we'll cross that bridge etc.
    FLAG("Initializing Mesh");

    // Note this first step spits out valid P and U on *physical* zones
    int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
    for (int i = 0; i < nmb; ++i) {
        // Initialize the base container with problem values
        // This could be switched into task format without too much issue
        auto& pmb = pmesh->block_list[i];
        auto& rc = pmb->meshblock_data.Get();
        InitializeProblem(rc, pin);
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
        for (int i = 0; i < nmb; ++i) {
            auto& pmb = pmesh->block_list[i];
            auto& rc = pmb->meshblock_data.Get();
            SeedBField(rc, pin);
        
            // TODO should this be added after normalization?
            // TODO option to add flux slowly during the run?
            Real BHflux = pin->GetOrAddReal("b_field", "bhflux", 0.0);
            if (BHflux > 0.) {
                //SeedBHFlux(rc, BHflux);
            }

            Real beta_local = GetLocalBetaMin(rc);
            if(beta_local < beta_min) beta_min = beta_local;
        }
        beta_min = MPIMin(beta_min);
#if DEBUG
        cerr << "Beta min pre-norm: " << beta_min << endl;
#endif

        // Then normalize B by sqrt(beta/beta_min)
        FLAG("Normalizing magnetic field");
        Real beta = pin->GetOrAddReal("b_field", "beta_min", 100.);
        Real norm = sqrt(beta_min/beta);
        for (int i = 0; i < nmb; ++i) {
            auto& pmb = pmesh->block_list[i];
            auto& rc = pmb->meshblock_data.Get();
            NormalizeBField(rc, norm);
        }
#if DEBUG
        // Do it again to check
        beta_min = 1e100;
        for (int i = 0; i < nmb; ++i) {
            auto& pmb = pmesh->block_list[i];
            auto& rc = pmb->meshblock_data.Get();
            Real beta_local = GetLocalBetaMin(rc);
            if(beta_local < beta_min) beta_min = beta_local;
        }
        cerr << "Beta min post-norm: " << beta_min << endl;
#endif
    }
    FLAG("Added B Field");

    // Sync to fill the ghost zones
    FLAG("Boundary sync");
    SyncAllBounds(pmesh);
    //pmesh->Initialize(0, pin);

    //Diagnostic(rc, IndexDomain::entire);


    FLAG("Initialized Mesh");
}

TaskStatus InitializeProblem(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    FLAG("Initializing Block");
    auto pmb = rc->GetBlockPointer();
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;

    auto& G = pmb->coords;
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = CreateEOS(gamma);

    auto prob = pin->GetString("parthenon/job", "problem_id"); // Required parameter
    if (prob == "mhdmodes") {
        int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
        int dir = pin->GetOrAddInteger("mhdmodes", "dir", 0);

        double tf = InitializeMHDModes(pmb, G, P, nmode, dir);
        pin->SetReal("parthenon/time", "tlim", tf);

    } else if (prob == "orszag_tang") {
        InitializeOrszagTang(pmb, G, P);

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

    // Apply any floors. Floors preserve P<->U so why not test that?
    FLAG("First Floors");
    ApplyFloors(rc);


    DelEOS(eos);
    FLAG("Initialized Block");
    return TaskStatus::complete;
}

void SyncAllBounds(Mesh *pmesh)
{
    // Update ghost cells. Only performed on U
    int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
    for (int i = 0; i < nmb; ++i) {
        auto& pmb = pmesh->block_list[i];
        auto& rc = pmb->meshblock_data.Get();
        rc->ClearBoundary(BoundaryCommSubset::mesh_init);
        rc->StartReceiving(BoundaryCommSubset::mesh_init);
        rc->SendBoundaryBuffers();
    }

    for (int i = 0; i < nmb; ++i) {
        auto& pmb = pmesh->block_list[i];
        auto& rc = pmb->meshblock_data.Get();
        rc->ReceiveAndSetBoundariesWithWait();
        rc->ClearBoundary(BoundaryCommSubset::mesh_init);
        //pmb->pbval->ProlongateBoundaries(0.0, 0.0);

        // Physical boundary conditions
        parthenon::ApplyBoundaryConditions(rc);
        ApplyCustomBoundaries(rc);

        // Fill P again, including ghost zones
        parthenon::FillDerivedVariables::FillDerived(rc);
    }
}