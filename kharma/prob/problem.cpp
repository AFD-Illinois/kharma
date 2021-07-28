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
#include "fluxes.hpp"
#include "gr_coordinates.hpp"
#include "b_field_tools.hpp"

// Problem initialization headers
#include "bondi.hpp"
#include "explosion.hpp"
#include "fm_torus.hpp"
#include "iharm_restart.hpp"
#include "kelvin_helmholtz.hpp"
#include "mhdmodes.hpp"
#include "orszag_tang.hpp"
#include "b_field_tools.hpp"

// Package headers
#include "mhd_functions.hpp"
#include "seed_B_ct.hpp"
#include "seed_B_cd.hpp"

#include "b_flux_ct.hpp"
#include "b_cd.hpp"
#include "b_functions.hpp"

#include "bvals/boundary_conditions.hpp"
#include "mesh/mesh.hpp"

using namespace parthenon;

void KHARMA::ProblemGenerator(MeshBlock *pmb, ParameterInput *pin)
{
    FLAG("Initializing Block");
    auto rc = pmb->meshblock_data.Get();
    GridVars P = rc->Get("c.c.bulk.prims").data;
    GridVector B_P = rc->Get("c.c.bulk.B_prim").data;
    GridVars U = rc->Get("c.c.bulk.cons").data;
    GridVector B_U = rc->Get("c.c.bulk.B_con").data;
    const bool use_b = pmb->packages.AllPackages().count("B_FluxCT") ||
                        pmb->packages.AllPackages().count("B_CD");

    GRCoordinates G = pmb->coords;
    EOS* eos = pmb->packages.Get("GRMHD")->Param<EOS*>("eos");

    auto prob = pin->GetString("parthenon/job", "problem_id"); // Required parameter
    if (prob == "mhdmodes") {
        int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
        int dir = pin->GetOrAddInteger("mhdmodes", "dir", 0);

        double tf = InitializeMHDModes(pmb, G, P, B_P, nmode, dir);
        if(tf > 0.) pin->SetReal("parthenon/time", "tlim", tf);

    } else if (prob == "orszag_tang") {
        Real tscale = pin->GetOrAddReal("orszag_tang", "tscale", 0.05);
        InitializeOrszagTang(pmb, G, P, B_P, tscale);

    } else if (prob == "explosion") {
        // Generally includes constant B field Bx = 0.1,1
        InitializeExplosion(pmb, G, P);

    } else if (prob == "kelvin_helmholtz") {
        // Generally includes constant B field Bx = 0.1,1
        Real tscale = pin->GetOrAddReal("kelvin_helmholtz", "tscale", 0.05);
        InitializeKelvinHelmholtz(pmb, G, P);

    } else if (prob == "bondi") {
        Real mdot = pin->GetOrAddReal("bondi", "mdot", 1.0);
        Real rs = pin->GetOrAddReal("bondi", "rs", 8.0);
        // Add these to package properties, since they continue to be needed on boundaries
        if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("mdot")))
            pmb->packages.Get("GRMHD")->AddParam<Real>("mdot", mdot);
        if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("rs")))
            pmb->packages.Get("GRMHD")->AddParam<Real>("rs", rs);

        InitializeBondi(pmb, G, P, eos, mdot, rs);

    } else if (prob == "torus") {
        Real rin = pin->GetOrAddReal("torus", "rin", 6.0);
        Real rmax = pin->GetOrAddReal("torus", "rmax", 12.0);
        FLAG("Initializing torus");
        InitializeFMTorus(pmb, G, P, eos, rin, rmax);

    } else if (prob == "iharm_restart") {
        auto fname = pin->GetString("iharm_restart", "fname"); // Require this, don't guess
        bool use_tf = pin->GetOrAddBoolean("iharm_restart", "use_tf", false);
        double tf = ReadIharmRestart(pmb, G, P, B_P, fname);
        if (use_tf) {
            pin->SetReal("parthenon/time", "tlim", tf);
        }
    }

    Real u_jitter = pin->GetOrAddReal("perturbation", "u_jitter", 0.0);
    int rng_seed = pin->GetOrAddInteger("perturbation", "rng_seed", 31337);
    if (u_jitter > 0.0) {
        FLAG("Applying U perturbation");
        PerturbU(pmb, P, u_jitter, rng_seed + pmb->gid);
    }

    // Apply any floors
    FLAG("First Floors");
    ApplyFloors(rc.get());

    // We finish by filling the conserved variables U,
    // which we'll treat as the independent/fundamental state.
    // P is filled again from this later on
    IndexDomain domain = IndexDomain::entire;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);
    // Initialize U
    FLAG("First P->U");
    pmb->par_for("first_U", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            GRMHD::p_to_u(G, P, B_P, eos, k, j, i, U);
            if (use_b) BField::p_to_u(G, B_P, k, j, i, B_U);
        }
    );

    FLAG("Initialized Block");
}

void PostInitialize(ParameterInput *pin, Mesh *pmesh)
{
    FLAG("Post-initialization started");

    // Add the field for torus problems as a second pass
    // Preserves P==U and ends with all physical zones fully defined
    if (pin->GetString("b_field", "type") != "none") {
        // Calculating B has a stencil outside physical zones
        FLAG("Extra boundary sync for B");
        SyncAllBounds(pmesh);

        // "Legacy" is the much more common normalization:
        // It's the ratio of max values over the domain i.e. max_P / max_PB,
        // not "beta" per se
        Real beta_calc_legacy = pin->GetOrAddBoolean("b_field", "legacy", true);

        // Use the correct seed function based on field constraint solver
        const bool use_b_flux_ct = pmesh->packages.AllPackages().count("B_FluxCT");
        const bool use_b_cd = pmesh->packages.AllPackages().count("B_CD");

        FLAG("Seeding magnetic field");
        // Seed the magnetic field and find the minimum beta
        Real beta_min = 1.e100, p_max = 0., bsq_max = 0.;
        for (auto &pmb : pmesh->block_list) {
            auto& rc = pmb->meshblock_data.Get();

            // This initializes B_P & B_U
            if (use_b_flux_ct) {
                B_FluxCT::SeedBField(rc.get(), pin);
            } else if (use_b_cd) {
                B_CD::SeedBField(rc.get(), pin);
                //B_FluxCT::SeedBField(rc.get(), pin);
            }
        
            // TODO should this be added after normalization?
            // TODO option to add flux slowly during the run?
            // Real BHflux = pin->GetOrAddReal("b_field", "bhflux", 0.0);
            // if (BHflux > 0.) {
            //     if (use_b_flux_ct) {
            //         B_FluxCT::SeedBHFlux(rc.get(), pin);
            //     } else if (use_b_cd) {
            //         B_CD::SeedBHFlux(rc.get(), pin);
            //     }
            // }

            if (beta_calc_legacy) {
                Real bsq_local = GetLocalBsqMax(rc.get());
                if(bsq_local > bsq_max) bsq_max = bsq_local;
                Real p_local = GetLocalPMax(rc.get());
                if(p_local > p_max) p_max = p_local;
            } else {
                Real beta_local = GetLocalBetaMin(rc.get());
                if(beta_local < beta_min) beta_min = beta_local;
            }
        }

        // Then, unless we're asked not to, normalize to some standard beta
        if (pin->GetOrAddBoolean("b_field", "norm", true)) {
            // Default to iharm3d's field normalization, pg_max/pb_max = 100
            // This is *not* the same as local beta_min = 100
            Real desired_beta_min = pin->GetOrAddReal("b_field", "beta_min", 100.);

            if (beta_calc_legacy) {
                bsq_max = MPIMax(bsq_max);
                p_max = MPIMax(p_max);
                beta_min = p_max / (0.5 * bsq_max);
            } else {
                beta_min = MPIMin(beta_min);
            }

            if (pin->GetInteger("debug", "verbose") > 0) {
                cerr << "Beta min pre-norm: " << beta_min << endl;
            }

            // Then normalize B by sqrt(beta/beta_min)
            FLAG("Normalizing magnetic field");
            if (beta_min > 0) {
                Real norm = sqrt(beta_min/desired_beta_min);
                for (auto &pmb : pmesh->block_list) {
                    auto& rc = pmb->meshblock_data.Get();
                    NormalizeBField(rc.get(), norm);
                }
            }
        }

        if (pin->GetInteger("debug", "verbose") > 0) {
            // Do it again to check, and add divB for good measure
            beta_min = 1e100; p_max = 0.; bsq_max = 0.;
            Real divb_max = 0.;
            for (auto &pmb : pmesh->block_list) {
                auto& rc = pmb->meshblock_data.Get();

                if (beta_calc_legacy) {
                    Real bsq_local = GetLocalBsqMax(rc.get());
                    if(bsq_local > bsq_max) bsq_max = bsq_local;
                    Real p_local = GetLocalPMax(rc.get());
                    if(p_local > p_max) p_max = p_local;
                } else {
                    Real beta_local = GetLocalBetaMin(rc.get());
                    if(beta_local < beta_min) beta_min = beta_local;
                }
                if (use_b_flux_ct) {
                    Real divb_local = B_FluxCT::MaxDivB(rc.get());
                    if(divb_local > divb_max) divb_max = divb_local;
                } else if (use_b_cd) {
                    Real divb_local = B_CD::MaxDivB(rc.get());
                    if(divb_local > divb_max) divb_max = divb_local;
                }
            }
            if (beta_calc_legacy) {
                bsq_max = MPIMax(bsq_max);
                p_max = MPIMax(p_max);
                beta_min = p_max / (0.5 * bsq_max);
            } else {
                beta_min = MPIMin(beta_min);
            }
            divb_max = MPIMax(divb_max);
            cerr << "Beta min post-norm: " << beta_min << endl;
            cerr << "Max divB post-norm: " << divb_max << endl;
        }

    }
    FLAG("Added B Field");

    // Sync to fill the ghost zones
    FLAG("Boundary sync");
    SyncAllBounds(pmesh);

    FLAG("Post-initialization finished");
}

void SyncAllBounds(Mesh *pmesh)
{
    // Mark primitives as Independent/FillGhost temporarily
    // Normally we use values from last step as guesses for UtoP, but we can't do that on first sync(s)
    // pmesh->packages.Get("GRMHD").get()->FieldMetadata("c.c.bulk.prims").Set(Metadata::Independent);
    // pmesh->packages.Get("GRMHD").get()->FieldMetadata("c.c.bulk.prims").Set(Metadata::FillGhost);
    // pmesh->packages.Get("GRMHD").get()->FieldMetadata("c.c.bulk.B_prim").Set(Metadata::Independent);
    // pmesh->packages.Get("GRMHD").get()->FieldMetadata("c.c.bulk.B_prim").Set(Metadata::FillGhost);

    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();
        rc->ClearBoundary(BoundaryCommSubset::mesh_init);
        rc->StartReceiving(BoundaryCommSubset::mesh_init);
        rc->SendBoundaryBuffers();
    }

    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();
        rc->ReceiveAndSetBoundariesWithWait();
        rc->ClearBoundary(BoundaryCommSubset::mesh_init);
        //pmb->pbval->ProlongateBoundaries(0.0, 0.0);

        // Physical boundary conditions
        parthenon::ApplyBoundaryConditions(rc);
        ApplyCustomBoundaries(rc.get());

        // Fill P again, including ghost zones
        parthenon::Update::FillDerived(rc.get());
    }

    // pmesh->packages.Get("GRMHD").get()->FieldMetadata("c.c.bulk.prims").Unset(Metadata::Independent);
    // pmesh->packages.Get("GRMHD").get()->FieldMetadata("c.c.bulk.prims").Unset(Metadata::FillGhost);
    // pmesh->packages.Get("GRMHD").get()->FieldMetadata("c.c.bulk.B_prim").Unset(Metadata::Independent);
    // pmesh->packages.Get("GRMHD").get()->FieldMetadata("c.c.bulk.B_prim").Unset(Metadata::FillGhost);
}
