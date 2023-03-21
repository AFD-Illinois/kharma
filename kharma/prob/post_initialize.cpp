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

#include "b_field_tools.hpp"
#include "b_cleanup.hpp"
#include "blob.hpp"
#include "boundaries.hpp"
#include "debug.hpp"
#include "floors.hpp"
#include "flux.hpp"
#include "gr_coordinates.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "kharma_driver.hpp"
#include "reductions.hpp"
#include "types.hpp"

#include "seed_B_ct.hpp"
#include "seed_B_cd.hpp"

/**
 * Perform a Parthenon MPI reduction.
 * Should only be used in initialization code, as the
 * reducer object & MPI comm are created on entry &
 * cleaned on exit
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

// Define reductions we need just for PostInitialize code.
// TODO namespace...
KOKKOS_INLINE_FUNCTION Real bsq(REDUCE_FUNCTION_ARGS_MESH)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    return dot(Dtmp.bcon, Dtmp.bcov);
}
KOKKOS_INLINE_FUNCTION Real gas_pres(REDUCE_FUNCTION_ARGS_MESH)
{
    return (gam - 1) * P(m_p.UU, k, j, i);
}
KOKKOS_INLINE_FUNCTION Real gas_beta(REDUCE_FUNCTION_ARGS_MESH)
{
    FourVectors Dtmp;
    GRMHD::calc_4vecs(G, P, m_p, k, j, i, Loci::center, Dtmp);
    return ((gam - 1) * P(m_p.UU, k, j, i))/(0.5*(dot(Dtmp.bcon, Dtmp.bcov) + SMALL));
}
Real MaxBsq(MeshData<Real> *md)
{
    return Reductions::DomainReduction(md, UserHistoryOperation::max, bsq, 0.0);
}
Real MaxPressure(MeshData<Real> *md)
{
    return Reductions::DomainReduction(md, UserHistoryOperation::max, gas_pres, 0.0);
}
Real MinBeta(MeshData<Real> *md)
{
    return Reductions::DomainReduction(md, UserHistoryOperation::min, gas_beta, 0.0);
}

void KHARMA::SeedAndNormalizeB(ParameterInput *pin, std::shared_ptr<MeshData<Real>> md)
{
    // Check which solver we'll be using
    auto pmesh = md->GetMeshPointer();
    const bool use_b_flux_ct = pmesh->packages.AllPackages().count("B_FluxCT");
    const bool use_b_cd = pmesh->packages.AllPackages().count("B_CD");
    const int verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    // Add the field for torus problems as a second pass
    // Preserves P==U and ends with all physical zones fully defined
    if (pin->GetOrAddString("b_field", "type", "none") != "none") {
        // "Legacy" is the much more common normalization:
        // It's the ratio of max values over the domain i.e. max(P) / max(P_B),
        // not necessarily a local min(beta)
        Real beta_calc_legacy = pin->GetOrAddBoolean("b_field", "legacy", true);

        Flag("Seeding magnetic field");
        // Seed the magnetic field on each block
        Real beta_min = 1.e100, p_max = 0., bsq_max = 0., bsq_min = 0.;
        for (auto &pmb : pmesh->block_list) {
            auto& rc = pmb->meshblock_data.Get();

            // This initializes B_P & B_U
            // TODO callback, also what about B_Cleanup?
            if (use_b_flux_ct) {
                B_FluxCT::SeedBField(rc.get(), pin);
            } else if (use_b_cd) {
                B_CD::SeedBField(rc.get(), pin);
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
        }

        // Then, if we're in a torus problem or explicitly ask for it,
        // normalize the magnetic field according to the density
        auto prob = pin->GetString("parthenon/job", "problem_id");
        if (pin->GetOrAddBoolean("b_field", "norm", (prob == "torus"))) {
            // Default to the general literature beta_min of 100.
            // As noted above, by default this uses the definition max(P)/max(P_B)!
            Real desired_beta_min = pin->GetOrAddReal("b_field", "beta_min", 100.);

            // Calculate current beta_min value
            Real bsq_min, bsq_max, p_max, beta_min;
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
            Flag("Normalizing magnetic field");
            if (beta_min > 0) {
                Real norm = m::sqrt(beta_min/desired_beta_min);
                for (auto &pmb : pmesh->block_list) {
                    auto& rc = pmb->meshblock_data.Get();
                    KHARMADriver::Scale(std::vector<std::string>{"prims.B"}, rc.get(), norm);
                }
            }
        }

        if (verbose > 0) {
            // Measure again to check. We'll add divB too, later
            Real bsq_min, bsq_max, p_max, beta_min;
            if (beta_calc_legacy) {
                bsq_max = MPIReduce_once(MaxBsq(md.get()), MPI_MAX);
                p_max = MPIReduce_once(MaxPressure(md.get()), MPI_MAX);
                beta_min = p_max / (0.5 * bsq_max);
            } else {
                beta_min = MPIReduce_once(MinBeta(md.get()), MPI_MIN);
            }
            if (MPIRank0()) {
                if (beta_calc_legacy) {
                    std::cout << "B^2 max pre-norm: " << bsq_max << std::endl;
                    std::cout << "Pressure max pre-norm: " << p_max << std::endl;
                }
                std::cout << "Beta min pre-norm: " << beta_min << std::endl;
            }
        }
    }

    // We've been initializing/manipulating P
    Flux::MeshPtoU(md.get(), IndexDomain::entire);
    // Synchronize after
    KHARMADriver::SyncAllBounds(md);

    Flag("Added B Field");
}

void KHARMA::PostInitialize(ParameterInput *pin, Mesh *pmesh, bool is_restart)
{
    Flag("Post-initialization started");
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

    // Make sure we've built the MeshData object we'll be synchronizing/updating
    auto &md = pmesh->mesh_data.GetOrAdd("base", 0);

    auto& pkgs = pmesh->packages.AllPackages();

    // First, make sure any data from the per-block init is synchronized
    Flag("Initial boundary sync");
    KHARMADriver::SyncAllBounds(md);

    // Then, add/modify any magnetic field left until this step
    // (since B field initialization can depend on global maxima,
    // & is handled by the B field transport package, it's sometimes done here)
    if (!is_restart) {
        KHARMA::SeedAndNormalizeB(pin, md);
    }

    // Print divB
    if (pin->GetString("b_field", "solver") != "none") {
        // If a B field exists, print divB here
        if (pkgs.count("B_FluxCT")) {
            B_FluxCT::PrintGlobalMaxDivB(md.get());
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
        KHARMADriver::SyncAllBounds(md);
    }

    // Any extra cleanup & init especially when restarting
    if (is_restart) {
        // Parthenon restores all parameters (global vars) when restarting,
        // but KHARMA needs a few (currently one) reset instead
        KHARMA::ResetGlobals(pin, pmesh);
    }

    // Clean the B field if we've introduced a divergence somewhere
    // Call this any time the package is loaded, all the
    // logic about parsing whether to clean is there
    if (pkgs.count("B_Cleanup")) {
        B_Cleanup::CleanupDivergence(md);
    }

    Flag("Post-initialization finished");
}
