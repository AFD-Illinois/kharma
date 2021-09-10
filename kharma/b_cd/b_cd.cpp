/* 
 *  File: b_cd.cpp
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

#include "b_cd.hpp"

#include "kharma.hpp"

using namespace parthenon;

// These are going to make this thing much more readable
#define B1 0
#define B2 1
#define B3 2

namespace B_CD
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    auto pkg = std::make_shared<StateDescriptor>("B_CD");
    Params &params = pkg->AllParams();

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    // Constraint damping options
    // Factor "lambda" in 
    Real damping = pin->GetOrAddReal("b_field", "damping", 0.1);
    params.Add("damping", damping);

    std::vector<int> s_vector({3});

    MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");

    // B field as usual
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                 Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes, Metadata::Vector}, s_vector);
    pkg->AddField("cons.B", m);
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
                  Metadata::Restart, isPrimitive, Metadata::Vector}, s_vector);
    pkg->AddField("prims.B", m);

    // Constraint damping scalar field psi.  Prim and cons forms correspond to B field forms,
    // i.e. differ by a factor of gdet.  This is apparently marginally more stable in some
    // circumstances.
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                  Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes});
    pkg->AddField("cons.psi_cd", m);
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
                  Metadata::Restart, isPrimitive});
    pkg->AddField("prims.psi_cd", m);

    // We only update the divB field for output
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("divB", m);

    pkg->FillDerivedBlock = B_CD::FillDerived;
    pkg->PostStepDiagnosticsMesh = B_CD::PostStepDiagnostics;

    // List (vector) of HistoryOutputVar that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    // In this package, we only care about MaxDivB
    // unless you want like, the median or something
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, MaxDivB, "MaxDivB"));
    // add callbacks for HST output identified by the `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

void UtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    auto& B_U = rc->Get("cons.B").data;
    auto& B_P = rc->Get("prims.B").data;
    auto& psi_U = rc->Get("cons.psi_cd").data;
    auto& psi_P = rc->Get("prims.psi_cd").data;

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    IndexRange ib = bounds.GetBoundsI(domain);
    IndexRange jb = bounds.GetBoundsJ(domain);
    IndexRange kb = bounds.GetBoundsK(domain);
    pmb->par_for("UtoP_B", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            // Update the primitive B-fields
            Real gdet = G.gdet(Loci::center, j, i);
            VLOOP B_P(v, k, j, i) = B_U(v, k, j, i) / gdet;
            // Update psi as well
            psi_P(k, j, i) = psi_U(k, j, i) / gdet;
        }
    );
}

TaskStatus AddSource(MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt)
{
    FLAG("Adding constraint damping source")
    // TODO mesh-wide
    auto pmb = rc->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return TaskStatus::complete;

    auto& psi_U = rc->Get("cons.psi_cd").data;
    auto& psiF1 = rc->Get("cons.psi_cd").flux[X1DIR];
    auto& psiF2 = rc->Get("cons.psi_cd").flux[X2DIR];
    auto& psiF3 = rc->Get("cons.psi_cd").flux[X3DIR];
    auto& psi_DU = dudt->Get("cons.psi_cd").data;

    auto& B_U = rc->Get("cons.B").data;
    auto& BF1 = rc->Get("cons.B").flux[X1DIR];
    auto& BF2 = rc->Get("cons.B").flux[X2DIR];
    auto& BF3 = rc->Get("cons.B").flux[X3DIR];
    auto& B_DU = dudt->Get("cons.B").data;

    const auto& G = pmb->coords;

    const Real lambda = pmb->packages.Get("B_CD")->Param<Real>("damping");

    FLAG("Allocated to add")

    // TODO add source terms to everything else here:
    // U1, U2, U3 get -(del*B) B
    // U gets -B*(grad psi)
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("AddSource_B_CD", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Add a source term to B based on psi
            GReal alpha_c = 1. / sqrt(-G.gcon(Loci::center, j, i, 0, 0));
            GReal gdet_c = G.gdet(Loci::center, j, i);

            double divB = ((BF1(B1, k, j, i+1) - BF1(B1, k, j, i)) / G.dx1v(i) +
                           (BF2(B2, k, j+1, i) - BF2(B2, k, j, i)) / G.dx2v(j));
            if (ndim > 2) divB += (BF3(B3, k+1, j, i) - BF3(B3, k, j, i)) / G.dx3v(k);

            VLOOP {
                // First term: gradient of psi
                B_DU(v, k, j, i) += alpha_c * G.gcon(Loci::center, j, i, v+1, 1) * (psiF1(k, j, i+1) - psiF1(k, j, i)) / G.dx1v(i) +
                                    alpha_c * G.gcon(Loci::center, j, i, v+1, 2) * (psiF2(k, j+1, i) - psiF2(k, j, i)) / G.dx2v(j);
                if (ndim > 2)
                    B_DU(v, k, j, i) += alpha_c * G.gcon(Loci::center, j, i, v+1, 3) * (psiF3(k+1, j, i) - psiF3(k, j, i)) / G.dx3v(k);

                // Second term: beta^i divB
                B_DU(v, k, j, i) += G.gcon(Loci::center, j, i, 0, v+1) * alpha_c * alpha_c * divB;
            }
            // Update psi using the analytic solution for the source term
            GReal dalpha1 = ( (1. / sqrt(-G.gcon(Loci::face1, j, i+1, 0, 0))) / G.gdet(Loci::face1, j, i+1)
                            - (1. / sqrt(-G.gcon(Loci::face1, j, i, 0, 0))) / G.gdet(Loci::face1, j, i)) / G.dx1v(i);
            GReal dalpha2 = ( (1. / sqrt(-G.gcon(Loci::face2, j+1, i, 0, 0))) / G.gdet(Loci::face2, j+1, i)
                            - (1. / sqrt(-G.gcon(Loci::face2, j, i, 0, 0))) / G.gdet(Loci::face2, j, i)) / G.dx2v(i);
            // There is not dalpha3, the coordinate system is symmetric along x3
            psi_DU(k, j, i) += B_U(B1, k, j, i) * dalpha1 + B_U(B2, k, j, i) * dalpha2 - alpha_c * lambda * psi_U(k, j, i);
        }
    );

    FLAG("Added")
    return TaskStatus::complete;
}

// TODO figure out what divB from psi looks like?

Real MaxDivB(MeshData<Real> *md)
{
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const auto& G = pmb->coords;
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return 0.;
    // We only care about interior cells
    is += 1; ie -= 1;
    js += 1; je -= 1;
    if (ndim > 2) { ks += 1; ke -= 1; }

    auto B = md->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});

    cerr << "Fluxes 6, 5, " << B.GetDim(6) << " " << B.GetDim(5);

    Real bsq_max;
    Kokkos::Max<Real> bsq_max_reducer(bsq_max);
    pmb->par_reduce("B_field_bsqmax", 0, B.GetDim(5), ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_MESH_3D_REDUCE {
            double divb_local = ((B(b).flux(1, B1, k, j, i+1) - B(b).flux(1, B1, k, j, i)) / G.dx1v(i)+
                                 (B(b).flux(2, B2, k, j+1, i) - B(b).flux(2, B2, k, j, i)) / G.dx2v(j));
            if (ndim > 2) divb_local += (B(b).flux(3, B3, k+1, j, i) - B(b).flux(3, B3, k, j, i)) / G.dx3v(k);

            if(divb_local > local_result) local_result = divb_local;
        }
    , bsq_max_reducer);
    return bsq_max;
}

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    FLAG("Printing B field diagnostics");
    auto pmesh = md->GetMeshPointer();

    // Print this unless we quash everything
    int verbose = pmesh->packages.Get("B_CD")->Param<int>("verbose");
    if (verbose >= 0) {
        Real max_divb = B_CD::MaxDivB(md);
        max_divb = MPIMax(max_divb);

        if(MPIRank0()) {
            cout << "Max DivB: " << max_divb << endl;
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
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return;
    // We only care about interior cells
    is += 1; //ie -= 1;
    js += 1; //je -= 1;
    if (ndim > 2) {
        ks += 1; //ke -= 1;
    }

    auto rc = pmb->meshblock_data.Get().get();
    auto& divB = rc->Get("divB").data;
    const auto& G = pmb->coords;

    GridVector F1, F2, F3;
    F1 = rc->Get("cons.B").flux[X1DIR];
    F2 = rc->Get("cons.B").flux[X2DIR];
    if (ndim > 2) F3 = rc->Get("cons.B").flux[X3DIR];

    pmb->par_for("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            double divb_local = ((F1(B1, k, j, i+1) - F1(B1, k, j, i)) / G.dx1v(i) +
                                 (F2(B2, k, j+1, i) - F2(B2, k, j, i)) / G.dx2v(j));
            if (ndim > 2) divb_local += (F3(B3, k, j+1, i) - F3(B3, k, j, i)) / G.dx3v(k);

            divB(k, j, i) = divb_local;
        }
    );
}

} // namespace B_CD
