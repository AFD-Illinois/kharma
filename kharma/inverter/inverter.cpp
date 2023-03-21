/* 
 *  File: inverter.cpp
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
#include "inverter.hpp"

// This will include headers in the correct order
#include "invert_template.hpp"

#include "reductions.hpp"

/**
 * Internal inversion fn, templated on inverter type.  Calls through to templated u_to_p
 * This is called with the correct template argument from BlockUtoP
 */
template<Inverter::Type inverter>
inline void BlockPerformInversion(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "Filling Primitives");
    auto pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;

    PackIndexMap prims_map, cons_map;
    auto U = GRMHD::PackMHDCons(rc, cons_map);
    auto P = GRMHD::PackHDPrims(rc, prims_map);
    const VarMap m_u(cons_map, true), m_p(prims_map, false);

    GridScalar pflag = rc->Get("pflag").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    const Real err_tol = pmb->packages.Get("Inverter")->Param<Real>("err_tol");
    const int iter_max = pmb->packages.Get("Inverter")->Param<int>("iter_max");
    const Real stepsize = pmb->packages.Get("Inverter")->Param<Real>("stepsize");

    // Get the primitives from our conserved versions
    // Currently this runs over *all* zones, including all ghosts, even
    // uninitialized zones which are still zero.  We select for initialized
    // zones only in the loop below, to avoid failures to converge while
    // calculating primtive vars over as much of the domain as possible
    // We could (did formerly) save some time here by running over
    // only zones with initialized conserved variables, but the domain
    // of such values is not rectangular in the current handling
    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange3 b = GetPhysicalZones(pmb, bounds);

    pmb->par_for("U_to_P", b.kb.s, b.kb.e, b.jb.s, b.jb.e, b.ib.s, b.ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            if (inside(k, j, i, b.kb, b.jb, b.ib)) {
                // Run over all interior zones and any initialized ghosts
                pflag(k, j, i) = static_cast<double>(Inverter::u_to_p<inverter>(G, U, m_u, gam, k, j, i, P, m_p, Loci::center));
            }
        }
    );
    Flag(rc, "Filled");
}

std::shared_ptr<KHARMAPackage> Inverter::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Inverter");
    Params &params = pkg->AllParams();

    // TODO TODO THESE ARE NO-OPS
    Real err_tol = pin->GetOrAddReal("inverter", "err_tol", 1e-8);
    params.Add("err_tol", err_tol);
    int iter_max = pin->GetOrAddInteger("inverter", "iter_max", 8);
    params.Add("iter_max", iter_max);
    Real stepsize = pin->GetOrAddReal("inverter", "stepsize", 1e-5);
    params.Add("stepsize", stepsize);

    std::string inverter_name = pin->GetOrAddString("inverter", "type", "onedw");
    if (inverter_name == "onedw") {
        params.Add("inverter_type", Type::onedw);
    } else if (inverter_name == "none") {
        params.Add("inverter_type", Type::none);
    }

    bool fix_average_neighbors = pin->GetOrAddBoolean("inverter", "fix_average_neighbors", true);
    params.Add("fix_average_neighbors", fix_average_neighbors);
    // TODO add version attempting to recover from entropy, stuff like that

    // Flag denoting UtoP inversion failures
    // Only needed if we're actually calling UtoP, but always allocated as it's retrieved often
    // Needs boundary sync if treating primitive variables as fundamental
    bool sync_prims = packages->Get("Driver")->Param<bool>("sync_prims");
    bool implicit_grmhd = packages->Get("GRMHD")->Param<bool>("implicit");
    Metadata m;
    if (sync_prims && !implicit_grmhd) {
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    } else {
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    }
    pkg->AddField("pflag", m);

    // Don't operate if GRMHD variables are being evolved implicitly
    // This package is still loaded because fixes
    if (!implicit_grmhd) {
        pkg->BlockUtoP = Inverter::BlockUtoP;
    }

    pkg->PostStepDiagnosticsMesh = Inverter::PostStepDiagnostics;

    return pkg;
}

void Inverter::BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    // This only chooses an implementation.  See BlockPerformInversion and implementations e.g. onedw.hpp
    auto& type = rc->GetBlockPointer()->packages.Get("Inverter")->Param<Type>("inverter_type");
    switch(type) {
    case Type::onedw:
        BlockPerformInversion<Type::onedw>(rc, domain, coarse);
        break;
    case Type::none:
        break;
    }
}

TaskStatus Inverter::PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    Flag("Printing Floor diagnostics");
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Options
    const auto& pars = pmesh->packages.Get("Globals")->AllParams();
    const int flag_verbose = pars.Get<int>("flag_verbose");

    // Debugging/diagnostic info about floor and inversion flags
    if (flag_verbose >= 1) {
        Flag("Printing flags");
        int nflags = Reductions::CountFlags(md, "pflag", Inverter::status_names, IndexDomain::interior, flag_verbose, false);
        // TODO TODO yell here if there are too many flags
    }

    return TaskStatus::complete;
}
