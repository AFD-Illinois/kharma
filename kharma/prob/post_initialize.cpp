/* 
 *  File: post_initialize.cpp
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

#include "post_initialize.hpp"

#include "b_cd.hpp"
#include "b_cleanup.hpp"
#include "b_ct.hpp"
#include "b_flux_ct.hpp"
#include "b_field_tools.hpp"
#include "blob.hpp"
#include "boundaries.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "kharma_driver.hpp"
#include "reductions.hpp"
#include "types.hpp"

/**
 * Perform a Parthenon MPI reduction.
 * Should only be used in initialization code, as the
 * reducer object & MPI comm are created on entry &
 * cleaned on exit
 * TODO use Reductions stuff?
 */
template<typename T>
inline T MPIReduce_once(T f, MPI_Op O)
{
    parthenon::AllReduce<T> reduction;
    reduction.val = f;
    reduction.StartReduce(O);
    // Wait on results
    while (reduction.CheckReduce() == parthenon::TaskStatus::incomplete);
    // TODO catch errors?
    return reduction.val;
}

// Shorter names for the reductions we use here
Real MaxBsq(MeshData<Real> *md)
{
    return Reductions::DomainReduction<Reductions::Var::bsq, Real>(md, UserHistoryOperation::max);
}
Real MaxPressure(MeshData<Real> *md)
{
    return Reductions::DomainReduction<Reductions::Var::gas_pressure, Real>(md, UserHistoryOperation::max);
}
Real MinBeta(MeshData<Real> *md)
{
    return Reductions::DomainReduction<Reductions::Var::beta, Real>(md, UserHistoryOperation::min);
}

void KHARMA::SeedAndNormalizeB(ParameterInput *pin, std::shared_ptr<MeshData<Real>> md)
{
    // Check which solver we'll be using
    auto pmesh = md->GetMeshPointer();
    const bool use_b_flux_ct = pmesh->packages.AllPackages().count("B_FluxCT")
                                || pmesh->packages.AllPackages().count("B_Cleanup");
    const bool use_b_cd = pmesh->packages.AllPackages().count("B_CD");
    const int verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    Flag("SeedBField");
    // Seed the magnetic field on each block
    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();

        // This initializes B_P & B_U
        if (use_b_flux_ct) {
            B_FluxCT::SeedBField(rc.get(), pin);
        } else if (use_b_cd) {
            B_CD::SeedBField(rc.get(), pin);
        }
    }
    EndFlag();

    // Then, if we're in a torus problem or we explicitly ask for it,
    // normalize the magnetic field according to the density
    auto prob = pin->GetString("parthenon/job", "problem_id");
    if (pin->GetOrAddBoolean("b_field", "norm", (prob == "torus"))) {
        Flag("NormBField");
        // Default to the general literature beta_min of 100.
        // As noted above, by default this uses the definition max(P)/max(P_B)!
        Real desired_beta_min = pin->GetOrAddReal("b_field", "beta_min", 100.);

        // "Legacy" is the much more common normalization:
        // It's the ratio of max values over the domain i.e. max(P) / max(P_B),
        // not necessarily a local min(beta)
        Real beta_calc_legacy = pin->GetOrAddBoolean("b_field", "legacy_norm", true);

        // Calculate current beta_min value
        Real bsq_max, p_max, beta_min;
        if (beta_calc_legacy) {
            bsq_max = MPIReduce_once(MaxBsq(md.get()), MPI_MAX);
            p_max = MPIReduce_once(MaxPressure(md.get()), MPI_MAX);
            beta_min = p_max / (0.5 * bsq_max);
        } else {
            beta_min = MPIReduce_once(MinBeta(md.get()), MPI_MIN);
        }

        if (MPIRank0() && verbose > 0) {
            if (beta_calc_legacy) {
                std::cout << "B^2 max pre-norm: " << bsq_max << std::endl;
                std::cout << "Pressure max pre-norm: " << p_max << std::endl;
            }
            std::cout << "Beta min pre-norm: " << beta_min << std::endl;
        }

        // Then normalize B by sqrt(beta/beta_min)
        if (beta_min > 0) {
            Real norm = m::sqrt(beta_min/desired_beta_min);
            for (auto &pmb : pmesh->block_list) {
                auto& rc = pmb->meshblock_data.Get();
                KHARMADriver::Scale(std::vector<std::string>{"prims.B"}, rc.get(), norm);
            }
        }

        // Measure again to check. We'll add divB too, later
        if (verbose > 0) {
            Real bsq_max, p_max, beta_min;
            if (beta_calc_legacy) {
                bsq_max = MPIReduce_once(MaxBsq(md.get()), MPI_MAX);
                p_max = MPIReduce_once(MaxPressure(md.get()), MPI_MAX);
                beta_min = p_max / (0.5 * bsq_max);
            } else {
                beta_min = MPIReduce_once(MinBeta(md.get()), MPI_MIN);
            }
            if (MPIRank0()) {
                if (beta_calc_legacy) {
                    std::cout << "B^2 max post-norm: " << bsq_max << std::endl;
                    std::cout << "Pressure max post-norm: " << p_max << std::endl;
                }
                std::cout << "Beta min post-norm: " << beta_min << std::endl;
            }
        }
        EndFlag(); //NormBField
    }

    // We've been initializing/manipulating P
    Flux::MeshPtoU(md.get(), IndexDomain::entire);
}

void KHARMA::PostInitialize(ParameterInput *pin, Mesh *pmesh, bool is_restart)
{
    // This call:
    // 1. Initializes any magnetic fields which are "seeded," i.e., defined with a magnetic field implementation
    //    rather than assuming an implementation and setting the field with problem initialization.
    // 2. Renormalizes magnetic fields based on a desired ratio of maximum magnetic/gas pressures
    // 3. Adds any extra material which might be superimposed when restarting, e.g. "hotspot" regions a.k.a. "blobs"
    // 4. Resets a couple of incidental flags, if Parthenon read them from a restart file
    // 5. If necessary, cleans up any magnetic field divergence present on the grid

    // Coming into this function, the *interior* regions should be initialized with a problem:
    // that is, at least rho, u, uvec on each physical zone.
    // If your problem requires custom boundary conditions, these should be implemented
    // with the problem and assigned to the relevant functions in the "Boundaries" package.

    auto &md = pmesh->mesh_data.Get();

    auto& pkgs = pmesh->packages.AllPackages();

    // Magnetic field operations
    if (pin->GetString("b_field", "solver") != "none") {
        // If we need to seed a field based on the problem's fluid initialization...
        if (pin->GetOrAddString("b_field", "type", "none") != "none" && !is_restart) {
            // B field init is not stencil-1, needs boundaries sync'd.
            // FreezeDirichlet ensures any Dirichlet conditions aren't overwritten by zeros
            KBoundaries::FreezeDirichlet(md);
            KHARMADriver::SyncAllBounds(md);

            // Then init B field on each block...
            KHARMA::SeedAndNormalizeB(pin, md);
        }

        // Regardless, if evolving a field we should print max(divB)
        // divB is not stencil-1 and we may not have run the above.
        // If we did, we still need another sync, so it works out
        KBoundaries::FreezeDirichlet(md);
        KHARMADriver::SyncAllBounds(md);

        if (pkgs.count("B_FluxCT")) {
            B_FluxCT::PrintGlobalMaxDivB(md.get());
        } else if (pkgs.count("B_CT")) {
            B_CT::PrintGlobalMaxDivB(md.get());
        } else if (pkgs.count("B_CD")) {
            //B_CD::PrintGlobalMaxDivB(md.get());
        }
    }

    // Add any hotspots.
    // Note any other modifications made when restarting should be made around here
    if (pin->GetOrAddBoolean("blob", "add_blob", false)) {
        for (auto &pmb : pmesh->block_list) {
            auto rc = pmb->meshblock_data.Get();
            // This inserts only in vicinity of some global r,th,phi
            InsertBlob(rc.get(), pin);
        }
    }

    // Any extra cleanup & init especially when restarting
    if (is_restart) {
        // Parthenon restores all parameters (global vars) when restarting,
        // but KHARMA needs a few (currently one) reset instead
        KHARMA::ResetGlobals(pin, pmesh);
    }

    // Clean the B field if we've introduced a divergence somewhere
    // We call this function any time the package is loaded:
    // if we decided to load it in kharma.cpp, we need to clean.
    if (pkgs.count("B_Cleanup")) {
        if (pin->GetOrAddBoolean("b_cleanup", "output_before_cleanup", false)) {
            auto tm = SimTime(0., 0., 0, 0, 0, 0, 0.);
            auto pouts = std::make_unique<Outputs>(pmesh, pin, &tm);
            pouts->MakeOutputs(pmesh, pin, &tm, SignalHandler::OutputSignal::now);
        }

        // This does its own MPI syncs
        B_Cleanup::CleanupDivergence(md);
    }

    // Finally, synchronize boundary values.
    // Freeze any Dirichlet physical boundaries as they are now, after cleanup/sync/etc.
    KBoundaries::FreezeDirichlet(md);
    // This is the first sync if there is no B field
    KHARMADriver::SyncAllBounds(md);
    // And make sure the trivial primitive values are up-to-date
    //Packages::MeshUtoPExceptMHD(md.get(), IndexDomain::entire, false);
}
