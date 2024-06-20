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
    const int nzones = pin->GetOrAddInteger("multizone", "nzones", 8);
    params.Add("nzones", nzones);
    const Real base = pin->GetOrAddReal("multizone", "base", 8.0);
    params.Add("base", base);
    const int num_Vcycles = pin->GetOrAddInteger("multizone", "num_Vcycles", 1); // not being used yet. will we need to use it?
    params.Add("num_Vcycles", num_Vcycles);
    const int ncycle_per_zone = pin->GetOrAddInteger("multizone", "ncycle_per_zone", 1); // if -1, switch according to the characteristic timescale
    params.Add("ncycle_per_zone", ncycle_per_zone);
    const bool move_rin = pin->GetOrAddBoolean("multizone", "move_rin", false);
    params.Add("move_rin", move_rin);

    // mutable parameters
    params.Add("i_within_vcycle", 0, true);
    params.Add("i_vcycle", 0, true);
    params.Add("t0_zone", 0.0, true);
    params.Add("n0_zone", 0, true);

    // options when using the characteristic time to determine zone-switching
    const Real rs = pin->GetOrAddReal("bondi", "rs", 16.);
    params.Add("bondi_rs", rs);
    const Real gam = pin->GetReal("GRMHD", "gamma");
    const Real f_tchar = pin->GetOrAddReal("multizone", "f_tchar", m::pow(base, -3. / 2.) / 2.);
    params.Add("f_tchar", f_tchar);
    const bool loc_tchar = pin->GetOrAddReal("multizone", "loc_tchar", true);
    params.Add("loc_tchar", loc_tchar);

    //pkg->BlockUtoP = Electrons::BlockUtoP;
    //pkg->BoundaryUtoP = Electrons::BlockUtoP;

    return pkg;
}

void Multizone::DecideActiveBlocks(Mesh *pmesh, const SimTime &tm, bool *is_active, bool apply_boundary_condition[][BOUNDARY_NFACES])
{
    Flag("DecideActiveBlocks");
    auto &params = pmesh->packages.Get("Multizone")->AllParams();

    // Input parameters
    const int nzones = params.Get<int>("nzones");
    const Real base = params.Get<Real>("base");
    const int num_Vcycles = params.Get<int>("num_Vcycles");
    const int ncycle_per_zone = params.Get<int>("ncycle_per_zone");
    const bool move_rin = params.Get<bool>("move_rin");
    const Real bondi_rs = params.Get<Real>("bondi_rs");
    const Real gam = pmesh->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real f_tchar = params.Get<Real>("f_tchar");
    const bool loc_tchar = params.Get<bool>("loc_tchar");
    
    // Current location in V-cycles
    int i_vcycle = params.Get<int>("i_vcycle"); // current i_vcycle
    int i_within_vcycle = params.Get<int>("i_within_vcycle"); // current i_within_vcycle
    Real t0_zone = params.Get<Real>("t0_zone"); // time at zone-switching
    int n0_zone = params.Get<int>("n0_zone"); // cycle at zone-switching
    int i_zone = m::abs(i_within_vcycle - (nzones - 1)); // zone number. zone-0 is the smallest annulus.

    // Derived parameters
    bool switch_zone = false; // determines if we are switching zones here
    Real temp_rin, runtime_per_zone;
    int nzones_per_vcycle = 2 * (nzones - 1);
    if (ncycle_per_zone > 0) switch_zone = (tm.ncycle - n0_zone) >= ncycle_per_zone;
    else {
        temp_rin = m::pow(base, i_zone);
        runtime_per_zone = f_tchar * CalcRuntime(temp_rin, base, gam, bondi_rs, loc_tchar);
        switch_zone = (tm.time - t0_zone) > runtime_per_zone;
        //std::cout << "time now " << tm.time << ", t0_zone " << t0_zone << ", runtime_per_zone " << runtime_per_zone << std::endl;
    }
    if (switch_zone) {
        i_within_vcycle += 1;
        if (i_within_vcycle >= nzones_per_vcycle) { // if completed a V-cycle
            i_within_vcycle -= nzones_per_vcycle;
            i_vcycle += 1;
        }
        i_zone = m::abs(i_within_vcycle - (nzones - 1)); // update zone number
        params.Update<int>("i_within_vcycle", i_within_vcycle);
        params.Update<int>("i_vcycle", i_vcycle);
        params.Update<Real>("t0_zone", tm.time);
        params.Update<int>("n0_zone", tm.ncycle);
    }

    // Range of radii that is active
    int active_rout;
    int active_rin = m::pow(base, i_zone);
    if (move_rin) active_rout = m::pow(base, nzones + 1);
    else active_rout =  m::pow(base, i_zone + 2);
    Real active_x1min = m::log(active_rin);
    Real active_x1max = m::log(active_rout);
    std::cout << "i_within_vcycle" << i_within_vcycle << " i_zone " << i_zone << " i_vcycle " << i_vcycle << " active_rout " << active_rout << " active_rin " << active_rin << std::endl;

    const int num_blocks = pmesh->block_list.size();
    
    // Initialize apply_boundary_condition
    for (int i=0; i < num_blocks; i++)
        for (int j=0; j < BOUNDARY_NFACES; j++)
            apply_boundary_condition[i][j] = false;

    // Determine Active Blocks
    GReal x1min, x1max;
    for (int iblock=0; iblock < num_blocks; iblock++) {
        auto &pmb = pmesh->block_list[iblock];
        x1min = pmb->block_size.xmin(X1DIR);
        x1max = pmb->block_size.xmax(X1DIR);
        if ((x1min + 1.0e-8 < active_x1min) || (x1max - 1.0e-8 > active_x1max)) is_active[iblock] = false;
        else { // active zone
            is_active[iblock] = true;

            // Decide where to apply bc // TODO: is this ok?
            if (m::abs(x1min - active_x1min) / m::max(active_x1min, SMALL) < 1.e-10) apply_boundary_condition[iblock][BoundaryFace::inner_x1] = true;
            if (m::abs(x1max - active_x1max) / active_x1max < 1.e-10) apply_boundary_condition[iblock][BoundaryFace::outer_x1] = true;
        }
        std::cout << "iblock " << iblock << " x1min " << x1min << " x1max " << x1max << ": is active? " << is_active[iblock] << ", boundary applied? " << apply_boundary_condition[iblock][BoundaryFace::inner_x1] << apply_boundary_condition[iblock][BoundaryFace::outer_x1] << std::endl;
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

