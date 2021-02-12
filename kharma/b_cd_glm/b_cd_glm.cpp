/* 
 *  File: b_cd_glm.cpp
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

#include <parthenon/parthenon.hpp>

#include "b_cd_glm.hpp"

#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

using namespace parthenon;

// These are going to make this thing much more readable
#define B1 0
#define B2 1
#define B3 2

extern double ctop_max;

namespace B_CD_GLM
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    auto pkg = std::make_shared<StateDescriptor>("B_CD_GLM");
    Params &params = pkg->AllParams();

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // Constraint damping options
    Real c_h_factor = pin->GetOrAddReal("b_field", "c_h_factor", 0.9);
    params.Add("c_h_factor", c_h_factor);
    Real c_h_low = pin->GetOrAddReal("b_field", "c_h_low", 1e-20);
    params.Add("c_h_low", c_h_low);
    Real c_h_high = pin->GetOrAddReal("b_field", "c_h_high", 1e20);
    params.Add("c_h_high", c_h_high);

    bool parabolic_term = pin->GetOrAddBoolean("b_field", "parabolic_term", true);
    params.Add("parabolic_term", parabolic_term);
    bool use_cr = pin->GetOrAddBoolean("b_field", "use_cr", true);
    params.Add("use_cr", use_cr);
    Real c_r = pin->GetOrAddReal("b_field", "c_r", 0.18);
    params.Add("c_r", c_r);
    Real c_d = pin->GetOrAddReal("b_field", "c_d", 0.18); // TODO sensible default?
    params.Add("c_d", c_d);

    std::vector<int> s_vector({3});

    MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");

    // B field as usual
    Metadata m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                  Metadata::Restart, Metadata::Conserved, Metadata::Vector}, s_vector);
    pkg->AddField("c.c.bulk.B_con", m);
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Restart, isPrimitive, Metadata::Vector}, s_vector);
    pkg->AddField("c.c.bulk.B_prim", m);

    // Constraint damping scalar field psi.  "Primitive" and "conserved" forms only differ by sqrt(-g) like B
    m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                  Metadata::Restart, Metadata::Conserved});
    pkg->AddField("c.c.bulk.psi_cd_con", m);
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::Restart, isPrimitive});
    pkg->AddField("c.c.bulk.psi_cd_prim", m);

    m = Metadata({Metadata::Cell, Metadata::Derived});
    pkg->AddField("c.c.bulk.divB_cd", m);

    pkg->FillDerivedBlock = B_CD_GLM::UtoP;
    return pkg;
}

void UtoP(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();

    auto& B_U = rc->Get("c.c.bulk.B_con").data;
    auto& B_P = rc->Get("c.c.bulk.B_prim").data;
    auto& psi_u = rc->Get("c.c.bulk.psi_cd_con").data;
    auto& psi_p = rc->Get("c.c.bulk.psi_cd_prim").data;

    auto& G = pmb->coords;

    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("UtoP_B", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Update the primitive B-fields
            Real gdet = G.gdet(Loci::center, j, i);
            VLOOP B_P(v, k, j, i) = B_U(v, k, j, i) / gdet;
            psi_p(k, j, i) = psi_u(k, j, i);
        }
    );
}

TaskStatus AddSource(MeshBlockData<Real> *rc, const Real& dt)
{
    // TODO mesh-wide
    auto pmb = rc->GetBlockPointer();

    auto& psi_u = rc->Get("c.c.bulk.psi_cd_con").data;

    auto& G = pmb->coords;

    const bool use_b_cd_glm = pmb->packages.AllPackages().count("B_CD_GLM") > 0;
    const Real c_h = clip(pmb->packages.Get("B_CD_GLM")->Param<Real>("c_h_factor") * ctop_max, 
                            pmb->packages.Get("B_CD_GLM")->Param<Real>("c_h_low"),
                            pmb->packages.Get("B_CD_GLM")->Param<Real>("c_h_high"));
    const Real c_p = pmb->packages.Get("B_CD_GLM")->Param<bool>("use_cr") ?
                        sqrt(pmb->packages.Get("B_CD_GLM")->Param<Real>("c_r") * c_h) :
                        sqrt(- dt * c_h*c_h / log(pmb->packages.Get("B_CD_GLM")->Param<Real>("c_d")));

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("UtoP_B", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Update psi using the analytic solution for the source term
            psi_u(k, j, i) = exp(-dt * (c_h*c_h) / (c_p*c_p)) * psi_u(k, j, i);
        }
    );

    return TaskStatus::complete;
}

Real MaxDivB_psi(MeshBlockData<Real> *rc0, MeshBlockData<Real> *rc1, const Real& dt, IndexDomain domain)
{
    auto pmb = rc0->GetBlockPointer();
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto& G = pmb->coords;

    GridVars psi0 = rc0->Get("c.c.bulk.psi_cd_con").data;
    GridVars psi1 = rc1->Get("c.c.bulk.psi_cd_con").data;

    Real dxmin = min(G.dx1v(0), min(G.dx2v(0), G.dx3v(0)));
    const Real c_h = clip(pmb->packages.Get("B_CD_GLM")->Param<Real>("c_h_factor") * ctop_max, 
                            pmb->packages.Get("B_CD_GLM")->Param<Real>("c_h_low"),
                            pmb->packages.Get("B_CD_GLM")->Param<Real>("c_h_high"));
    const Real c_p = pmb->packages.Get("B_CD_GLM")->Param<bool>("use_cr") ?
                        sqrt(pmb->packages.Get("B_CD_GLM")->Param<Real>("c_r") * c_h) :
                        sqrt(- dt * c_h*c_h / log(pmb->packages.Get("B_CD_GLM")->Param<Real>("c_d")));

    Real bsq_max;
    Kokkos::Max<Real> bsq_max_reducer(bsq_max);
    pmb->par_reduce("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            double divb_local = abs((psi1(k, j, i) - psi0(k, j, i)) / c_h / c_h / dt + (psi1(k, j, i) + psi0(k, j, i)) / 2 / c_p / c_p);
            if(divb_local > local_result) local_result = divb_local;
        }
    , bsq_max_reducer);
    return bsq_max;
}

Real MaxDivB(MeshBlockData<Real> *rc, IndexDomain domain)
{
    auto pmb = rc->GetBlockPointer();
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto& G = pmb->coords;
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return 0.;
    // We only care about interior cells
    is += 1; ie -= 1;
    js += 1; je -= 1;
    if (ndim > 2) { ks += 1; ke -= 1; }

    GridVars B_U = rc->Get("c.c.bulk.B_con").data;

    Real bsq_max;
    Kokkos::Max<Real> bsq_max_reducer(bsq_max);
    pmb->par_reduce("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            double divb_local = (B_U(0, k, j, i+1) - B_U(0, k, j, i-1))/(2 * G.dx1v(i))
                                +(B_U(1, k, j+1, i) - B_U(1, k, j-1, i))/(2 * G.dx2v(j));
            if (ndim > 2) divb_local += (B_U(2, k+1, j, i) - B_U(2, k-1, j, i))/(2 * G.dx3v(k));
            divb_local = fabs(divb_local);

            if(divb_local > local_result) local_result = divb_local;
        }
    , bsq_max_reducer);
    return bsq_max;
}

TaskStatus PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime& tm)
{
    FLAG("Printing B field diagnostics");

    // Print this unless we quash everything
    //int verbose = pmesh->packages.Get("B_CD_GLM")->Param<int>("verbose");
    int verbose = pmesh->packages.Get("GRMHD")->Param<int>("verbose");
    if (verbose > -1) {
        // We probably only care on the physical grid
        IndexDomain domain = IndexDomain::interior;

        Real max_divb = 0, max_divb_psi = 0;
        int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
        for (int i=0; i < nmb; ++i) {
            auto& pmb = pmesh->block_list[i];
            auto& rc0 = pmb->meshblock_data.Get("preserve");
            auto& rc1 = pmb->meshblock_data.Get();

            Real max_divb_l = B_CD_GLM::MaxDivB(rc1.get(), domain);
            Real max_divb_psi_l;
            if (verbose >= 1)
                max_divb_psi_l = B_CD_GLM::MaxDivB_psi(rc0.get(), rc1.get(), tm.dt, domain);

            if (max_divb_psi_l > max_divb_psi) max_divb_psi = max_divb_psi_l;
            if (max_divb_l > max_divb) max_divb = max_divb_l;
        }
        max_divb = MPIMax(max_divb);
        if (verbose >= 1) max_divb_psi = MPIMax(max_divb_psi);

        auto& G = pmesh->block_list[0]->coords;
        Real dxmin = min(G.dx1v(0), min(G.dx2v(0), G.dx3v(0)));
        const Real c_h = pmesh->block_list[0]->packages.Get("GRMHD")->Param<Real>("cfl") / tm.dt * dxmin;

        if(MPIRank0()) {
            cout << "Max DivB: " << max_divb;
            if (verbose >= 1) cout << " psi d_t " << max_divb_psi << " c_h would be " << c_h << " or " << ctop_max;
            cout << endl;
        }
    }

    FLAG("Printed")
    return TaskStatus::complete;
}

void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    auto& G = pmb->coords;
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return;
    // We only care about interior cells
    is += 1; ie -= 1;
    js += 1; je -= 1;
    if (ndim > 2) { ks += 1; ke -= 1; }

    auto rc = pmb->meshblock_data.Get().get();
    GridVars B_U = rc->Get("c.c.bulk.B_con").data;
    GridVars divB = rc->Get("c.c.bulk.divB_cd").data;

    pmb->par_for("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            double divb_local = (B_U(0, k, j, i+1) - B_U(0, k, j, i-1))/(2 * G.dx1v(i))
                                + (B_U(1, k, j+1, i) - B_U(1, k, j-1, i))/(2 * G.dx2v(j));
            if (ndim > 2) divb_local += (B_U(2, k+1, j, i) - B_U(2, k-1, j, i))/(2 * G.dx3v(k));
            divB(k, j, i) = divb_local;
        }
    );
}

} // namespace B_FluxCT