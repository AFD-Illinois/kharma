/* 
 *  File: electrons.cpp
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
#include "entropy.hpp"

#include "decs.hpp"
#include "domain.hpp"
#include "kharma_driver.hpp"
#include "flux.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "gaussian.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

namespace Entropy
{

std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Entropy");
    Params &params = pkg->AllParams();

    // Floors
    // TODO whether to limit Ktot


    // Flags
    // Note variables from each package are also marked with a flag of the package name,
    // in this case "Entropy"
    auto& driver = packages->Get("Driver")->AllParams();
    std::vector<MetadataFlag> flags_elec = {Metadata::Cell, Metadata::GetUserFlag("Explicit")};
    auto flags_prim = driver.Get<std::vector<MetadataFlag>>("prim_flags");
    flags_prim.insert(flags_prim.end(), flags_elec.begin(), flags_elec.end());
    auto flags_cons = driver.Get<std::vector<MetadataFlag>>("cons_flags");
    flags_cons.insert(flags_cons.end(), flags_elec.begin(), flags_elec.end());

    // Fields
    // A third argument of a std::vector<int> is available for specifying vector/tensor size,
    // but scalar is the defualt.
    pkg->AddField("cons.Ktot", flags_cons);
    pkg->AddField("prims.Ktot", flags_prim);

    // Callbacks.  For an evolved variable, we need at least UtoP,
    // and an accompanying device-side PtoU clause in `flux_functions.hpp`
    pkg->BlockUtoP = Entropy::BlockUtoP;
    pkg->BoundaryUtoP = Entropy::BlockUtoP;

    return pkg;
}

TaskStatus InitEntropy(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    Flag("InitEntropy");
    auto pmb = rc->GetBlockPointer();

    // Just grab all the primitive vars, we need rho, u for Ktot
    PackIndexMap prims_map;
    auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    const IndexRange3 b = KDomain::GetRange(rc, IndexDomain::interior);
    pmb->par_for("Init_entropy", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            P(m_p.KTOT, k, j, i) = (gam - 1.) * P(m_p.UU, k, j, i) * m::pow(P(m_p.RHO, k, j, i), -gam);
        }
    );

    EndFlag();
    return TaskStatus::complete;
}

void BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    // Get all cons but just the prim we update
    PackIndexMap cons_map;
    auto& Pk = rc->PackVariables({Metadata::GetUserFlag("Entropy"), Metadata::GetUserFlag("Primitive")});
    auto& U = rc->PackVariables({Metadata::Conserved}, cons_map);
    VarMap m_u(cons_map, true);

    const IndexRange3 b = KDomain::GetRange(rc, domain, coarse);
    pmb->par_for("UtoP_entropy", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Pk(0, k, j, i) = U(m_u.KTOT, k, j, i) / U(m_u.RHO, k, j, i);
        }
    );
}

void BlockPtoU(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map, cons_map;
    auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive"), Metadata::Cell}, prims_map);
    auto& U = rc->PackVariables({Metadata::Conserved, Metadata::Cell}, cons_map);
    const VarMap m_p(prims_map, false), m_u(cons_map, true);

    const auto& G = pmb->coords;

    const IndexRange3 b = KDomain::GetRange(rc, domain, coarse);
    pmb->par_for("PtoU_electrons", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Entropy::p_to_u(G, P, m_p, k, j, i, U, m_u);
        }
    );
}

TaskStatus UpdateEntropy(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();

    PackIndexMap prims_map;
    auto& P = rc->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    // Floors (TODO)

    // This function (and any primitive-variable sources) needs to be run over the entire domain,
    // because the boundary zones have already been updated and so the same calculations must be applied
    // in order to keep them consistent.
    // See kharma_step.cpp for the full picture of what gets updated when.
    const IndexRange3 b = KDomain::GetRange(rc, IndexDomain::entire);
    pmb->par_for("heat_electrons", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            // Calculate the new total entropy in this cell considering heating
            P(m_p.KTOT, k, j, i) = (gam-1.) * P(m_p.UU, k, j, i) / m::pow(P(m_p.RHO, k, j, i), gam);
        }
    );
    // TODO only the new var?
    //Entropy::BlockPtoU(rc, IndexDomain::interior);

    EndFlag();
    return TaskStatus::complete;
}

void ApplyFloors(MeshBlockData<Real> *mbd, IndexDomain domain)
{
    auto pmb = mbd->GetBlockPointer();
    auto packages = pmb->packages;

    PackIndexMap prims_map, cons_map;
    auto P = mbd->PackVariables({Metadata::GetUserFlag("Primitive")}, prims_map);
    const VarMap m_p(prims_map, false);

    auto fflag = mbd->PackVariables(std::vector<std::string>{"flags.floors"});

    const auto& G = pmb->coords;

    const Real gam = packages.Get("GRMHD")->Param<Real>("gamma");
    const Floors::Prescription floors = packages.Get("Floors")->Param<Floors::Prescription>("prescription");
    const Real ktot_max = floors.ktot_max;

    const IndexRange3 b = KDomain::GetRange(mbd, domain);
    pmb->par_for("apply_electrons_floors", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            // Also apply the ceiling to the advected entropy KTOT, if we're keeping track of that
            // (either for electrons, or robust primitive inversions in future)
            if (P(m_p.KTOT, k, j, i) > ktot_max) {
                fflag(0, k, j, i) = Floors::FFlag::KTOT | (int) fflag(0, k, j, i);
                P(m_p.KTOT, k, j, i) = ktot_max;
            }
        }
    );
    Entropy::BlockPtoU(mbd, domain);
}

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *rc)
{
    // Output any diagnostics after a step completes

    return TaskStatus::complete;
}

void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    // Compute any variables or diagnostics that should be updated for output to a file,
    // but which are not otherwise evolved during the simulation.
    // Note this includes anything computed for reductions/history file outputs
}

} // namespace Entropy
