/* 
 *  File: seed_B.cpp
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
#include "seed_B.hpp"

#include "seed_B_impl.hpp"

#include "boundaries.hpp"
#include "coordinate_utils.hpp"
#include "fm_torus.hpp"
#include "grmhd_functions.hpp"

using namespace parthenon;

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

TaskStatus SeedBField(MeshData<Real> *md, ParameterInput *pin)
{
    Flag("SeedBField");
    std::string b_field_type = pin->GetString("b_field", "type");
    auto pmesh = md->GetMeshPointer();
    const int verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

    if (verbose) {
        std::cout << "Seeding B field with type " << b_field_type << std::endl;
    }

    TaskStatus status = TaskStatus::incomplete;
    for (int i=0; i < md->NumBlocks(); i++) {
        auto *rc = md->GetBlockData(i).get();

        // I could make this a map or something,
        // but this is the only place I decode it.
        // TODO could also save it to a package...
        // TODO accumulate TaskStatus properly?
        if (b_field_type == "constant") {
            status = SeedBFieldType<BSeedType::constant>(rc, pin);
        } else if (b_field_type == "monopole") {
            status = SeedBFieldType<BSeedType::monopole>(rc, pin);
        } else if (b_field_type == "monopole_cube") {
            status = SeedBFieldType<BSeedType::monopole_cube>(rc, pin);
        } else if (b_field_type == "sane") {
            status = SeedBFieldType<BSeedType::sane>(rc, pin);
        } else if (b_field_type == "mad") {
            status = SeedBFieldType<BSeedType::mad>(rc, pin);
        } else if (b_field_type == "mad_quadrupole") {
            status = SeedBFieldType<BSeedType::mad_quadrupole>(rc, pin);
        } else if (b_field_type == "r3s3") {
            status = SeedBFieldType<BSeedType::r3s3>(rc, pin);
        } else if (b_field_type == "steep" || b_field_type == "r5s5") {
            status = SeedBFieldType<BSeedType::r5s5>(rc, pin);
        } else if (b_field_type == "gaussian") {
            status = SeedBFieldType<BSeedType::gaussian>(rc, pin);
        } else if (b_field_type == "bz_monopole") {
            status = SeedBFieldType<BSeedType::bz_monopole>(rc, pin);
        } else if (b_field_type == "vertical") {
            status = SeedBFieldType<BSeedType::vertical>(rc, pin);
        } else {
            throw std::invalid_argument("Magnetic field seed type not supported: " + b_field_type);
        }
    }

    EndFlag();
    return status;
}

TaskStatus NormalizeBField(MeshData<Real> *md, ParameterInput *pin)
{
    Flag("NormBField");
    // Check which solver we'll be using
    auto pmesh = md->GetMeshPointer();
    const int verbose = pmesh->packages.Get("Globals")->Param<int>("verbose");

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
        bsq_max = MPIReduce_once(MaxBsq(md), MPI_MAX);
        p_max = MPIReduce_once(MaxPressure(md), MPI_MAX);
        beta_min = p_max / (0.5 * bsq_max);
    } else {
        beta_min = MPIReduce_once(MinBeta(md), MPI_MIN);
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
    } // else yell?

    // Measure again to check
    if (verbose > 0) {
        Real bsq_max, p_max, beta_min;
        if (beta_calc_legacy) {
            bsq_max = MPIReduce_once(MaxBsq(md), MPI_MAX);
            p_max = MPIReduce_once(MaxPressure(md), MPI_MAX);
            beta_min = p_max / (0.5 * bsq_max);
        } else {
            beta_min = MPIReduce_once(MinBeta(md), MPI_MIN);
        }
        if (MPIRank0()) {
            if (beta_calc_legacy) {
                std::cout << "B^2 max post-norm: " << bsq_max << std::endl;
                std::cout << "Pressure max post-norm: " << p_max << std::endl;
            }
            std::cout << "Beta min post-norm: " << beta_min << std::endl;
        }
    }

    // We've been initializing/manipulating P
    Flux::MeshPtoU(md, IndexDomain::entire);

    EndFlag(); //NormBField
    return TaskStatus::complete;
}