/* 
 *  File: multizone.cpp
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
#include "multizone.hpp"

#include "decs.hpp"
#include "kharma.hpp"

#include "b_ct.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

std::shared_ptr<KHARMAPackage> Multizone::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Multizone");
    Params &params = pkg->AllParams();

    // Multizone basic parameters
    int nzones = pin->GetOrAddInteger("multizone", "nzones", 8);
    params.Add("nzones", nzones);
    Real base = pin->GetOrAddReal("multizone", "base", 8.0);
    params.Add("base", base);
    int nvcycles = pin->GetOrAddInteger("multizone", "nvcycles", 1);
    params.Add("nvcycles", nvcycles);
    int nstep_per_zone = pin->GetOrAddInteger("multizone", "nstep_per_zone", 1);
    params.Add("nstep_per_zone", nstep_per_zone);
    bool move_rin = pin->GetOrAddBoolean("multizone", "move_rin", false);
    params.Add("move_rin", move_rin);


    //pkg->BlockUtoP = Electrons::BlockUtoP;
    //pkg->BoundaryUtoP = Electrons::BlockUtoP;

    return pkg;
}

void Multizone::DecideActiveBlocks(Mesh *pmesh, const SimTime &tm, bool *is_active) //, bool **apply_boundary_condition)
{
    Flag("DecideActiveBlocks");
    auto &params = pmesh->packages.Get("Multizone")->AllParams();

    // Input parameters
    int nzones = params.Get<int>("nzones");
    Real base = params.Get<Real>("base");
    int nvcycles = params.Get<int>("nvcycles");
    int nstep_per_zone = params.Get<int>("nstep_per_zone"); // TODO: generalize to when this is not 1

    // Derived parameters
    int i_within_vcycle = (tm.ncycle % (2 * (nzones - 1)));
    int izone = m::abs(i_within_vcycle - (nzones - 1));
    int ivcycle = (tm.ncycle - i_within_vcycle) / (2 * (nzones - 1));

    int r_out = m::pow(base, izone + 2);
    int r_in = m::pow(base, izone);
    std::cout << "i_within_vcycle" << i_within_vcycle << " izone " << izone << " ivcycle " << ivcycle << " r_out " << r_out << " r_in " << r_in << std::endl;

    // Determine Active Blocks
    const int num_blocks = pmesh->block_list.size();
    GReal x1min, x1max;
    for (int iblock=0; iblock < num_blocks; iblock++) {
        auto &pmb = pmesh->block_list[iblock];
        x1min = pmb->block_size.xmin(X1DIR);
        x1max = pmb->block_size.xmax(X1DIR);
        if (x1min < m::log(r_in)) is_active[iblock] = false; // TODO: currently only supporting move_rin = true
        else is_active[iblock] = true;
        std::cout << "iblock " << iblock << " x1min " << x1min << " x1max " << x1max << ": is active? " << is_active[iblock] << std::endl;
    }
    EndFlag();

}

TaskStatus Multizone::AverageEMFSeams(MeshData<Real> *md_emf_only, bool *apply_boundary_condition)
{
    Flag("AverageEMFSeams");

    for (int i=0; i < BOUNDARY_NFACES; i++) {
        if (apply_boundary_condition[i]) {
            auto& rc = md_emf_only->GetBlockData(0); // Only one block
            // This is the only thing in the MeshData we're passed anyway...
            auto& emfpack = rc->PackVariables(std::vector<std::string>{"B_CT.emf"});
            if (0) {
                B_CT::AverageBoundaryEMF(rc.get(),
                                        KBoundaries::BoundaryDomain(static_cast<BoundaryFace>(i)),
                                        emfpack, false);
            } else {
                B_CT::ZeroBoundaryEMF(rc.get(),
                                        KBoundaries::BoundaryDomain(static_cast<BoundaryFace>(i)),
                                        emfpack, false);
            }
        }
    }

    EndFlag();
    return TaskStatus::complete;
}

TaskStatus Multizone::PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc)
{
    Flag("PostStepDiagnostics");

    // Output any diagnostics after a step completes

    EndFlag();
    return TaskStatus::complete;
}

