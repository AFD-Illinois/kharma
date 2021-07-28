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

#include <parthenon/parthenon.hpp>

#include "b_cd.hpp"

#include "b_flux_ct.hpp"
#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

using namespace parthenon;

// These are going to make this thing much more readable
#define B1 0
#define B2 1
#define B3 2

extern double ctop_max;

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
    pkg->AddField("c.c.bulk.B_con", m);
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
                  Metadata::Restart, isPrimitive, Metadata::Vector}, s_vector);
    pkg->AddField("c.c.bulk.B_prim", m);

    // Constraint damping scalar field psi.  Prim and cons forms correspond to B field forms,
    // i.e. differ by a factor of gdet.  This is apparently marginally more stable in some
    // circumstances.
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                  Metadata::Restart, Metadata::Conserved, Metadata::WithFluxes});
    pkg->AddField("c.c.bulk.psi_cd_con", m);
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
                  Metadata::Restart, isPrimitive});
    pkg->AddField("c.c.bulk.psi_cd_prim", m);

    // We only update the divB field for output
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("c.c.bulk.divB", m);

    pkg->FillDerivedBlock = B_CD::UtoP;
    return pkg;
}

void UtoP(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();

    auto& B_U = rc->Get("c.c.bulk.B_con").data;
    auto& B_P = rc->Get("c.c.bulk.B_prim").data;
    auto& psi_U = rc->Get("c.c.bulk.psi_cd_con").data;
    auto& psi_P = rc->Get("c.c.bulk.psi_cd_prim").data;

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
            // Update psi as well
            psi_P(k, j, i) = psi_U(k, j, i) / gdet;
        }
    );
}

TaskStatus AddSource(MeshBlockData<Real> *rc, MeshBlockData<Real> *dudt, const Real& dt)
{
    FLAG("Adding constraint damping source")
    // TODO mesh-wide
    auto pmb = rc->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return TaskStatus::complete;

    auto& psi_U = rc->Get("c.c.bulk.psi_cd_con").data;
    auto& psiF1 = rc->Get("c.c.bulk.psi_cd_con").flux[X1DIR];
    auto& psiF2 = rc->Get("c.c.bulk.psi_cd_con").flux[X2DIR];
    auto& psiF3 = rc->Get("c.c.bulk.psi_cd_con").flux[X3DIR];
    auto& psi_DU = dudt->Get("c.c.bulk.psi_cd_con").data;

    auto& B_U = rc->Get("c.c.bulk.B_con").data;
    auto& BF1 = rc->Get("c.c.bulk.B_con").flux[X1DIR];
    auto& BF2 = rc->Get("c.c.bulk.B_con").flux[X2DIR];
    auto& BF3 = rc->Get("c.c.bulk.B_con").flux[X3DIR];
    auto& B_DU = dudt->Get("c.c.bulk.B_con").data;

    auto& G = pmb->coords;

    const Real lambda = pmb->packages.Get("B_CD")->Param<Real>("damping");

    FLAG("Allocated to add")

    // TODO add source terms to everything else here:
    // U1, U2, U3 get -(del*B) B
    // U gets -B*(grad psi)
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("UtoP_B", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // Add a source term to B based on psi
            GReal alpha = 1. / sqrt(-G.gcon(Loci::center, j, i, 0, 0));

            double divB = ((BF1(B1, k, j, i+1) - BF1(B1, k, j, i)) / G.dx1v(i) +
                           (BF2(B2, k, j+1, i) - BF2(B2, k, j, i)) / G.dx2v(j));
            if (ndim > 2) divB += (BF3(B3, k, j+1, i) - BF3(B3, k, j, i)) / G.dx3v(k);

            VLOOP {
                // First term: gradient of psi
                B_DU(v, k, j, i) += alpha * G.gcon(Loci::center, j, i, v+1, 1) * (psiF1(k, j, i+1) - psiF1(k, j, i)) / G.dx1v(i) +
                                    alpha * G.gcon(Loci::center, j, i, v+1, 2) * (psiF2(k, j+1, i) - psiF2(k, j, i)) / G.dx2v(j);
                if (ndim > 2)
                    B_DU(v, k, j, i) += alpha * G.gcon(Loci::center, j, i, v+1, 3) * (psiF3(k, j+1, i) - psiF3(k, j, i)) / G.dx3v(k);

                // Second term: beta^i divB
                B_DU(v, k, j, i) += G.gcon(Loci::center, j, i, 0, v+1) * alpha * alpha * divB;
            }
            // Update psi using the analytic solution for the source term
            GReal dalpha1 = (1. / sqrt(-G.gcon(Loci::face1, j, i+1, 0, 0)) - 1. / sqrt(-G.gcon(Loci::face1, j, i, 0, 0))) / G.dx1v(i);
            GReal dalpha2 = (1. / sqrt(-G.gcon(Loci::face2, j+1, i, 0, 0)) - 1. / sqrt(-G.gcon(Loci::face2, j, i, 0, 0))) / G.dx2v(i);
            psi_DU(k, j, i) += B_U(X1DIR, k, j, i) * dalpha1 + B_U(X2DIR, k, j, i) * dalpha2 - alpha * lambda * psi_U(k, j, i);
        }
    );

    FLAG("Added")
    return TaskStatus::complete;
}

// TODO figure out what divB from psi looks like?

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


    GridVector F1, F2, F3;
    F1 = rc->Get("c.c.bulk.B_con").flux[X1DIR];
    F2 = rc->Get("c.c.bulk.B_con").flux[X2DIR];
    if (ndim > 2) F3 = rc->Get("c.c.bulk.B_con").flux[X3DIR];

    Real bsq_max;
    Kokkos::Max<Real> bsq_max_reducer(bsq_max);
    pmb->par_reduce("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            double divb_local = ((F1(B1, k, j, i+1) - F1(B1, k, j, i)) / G.dx1v(i)+
                                 (F2(B2, k, j+1, i) - F2(B2, k, j, i)) / G.dx2v(j));
            if (ndim > 2) divb_local += (F3(B3, k, j+1, i) - F3(B3, k, j, i)) / G.dx3v(k);

            if(divb_local > local_result) local_result = divb_local;
        }
    , bsq_max_reducer);
    return bsq_max;
}

TaskStatus PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime& tm)
{
    FLAG("Printing B field diagnostics");

    // Print this unless we quash everything
    //int verbose = pmesh->packages.Get("B_CD")->Param<int>("verbose");
    int verbose = pmesh->packages.Get("GRMHD")->Param<int>("verbose");
    if (verbose > -1) {
        // We probably only care on the physical grid
        IndexDomain domain = IndexDomain::interior;

        Real max_divb = 0;
        for (auto &pmb : pmesh->block_list) {
            auto& rc0 = pmb->meshblock_data.Get("preserve");
            auto& rc1 = pmb->meshblock_data.Get();

            Real max_divb_l = 0;
            if (verbose >= 1)
                max_divb_l = B_CD::MaxDivB(rc1.get(), domain);

            if (max_divb_l > max_divb) max_divb = max_divb_l;
        }
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
    auto& divB = rc->Get("c.c.bulk.divB").data;
    auto& G = pmb->coords;

    GridVector F1, F2, F3;
    F1 = rc->Get("c.c.bulk.B_con").flux[X1DIR];
    F2 = rc->Get("c.c.bulk.B_con").flux[X2DIR];
    if (ndim > 2) F3 = rc->Get("c.c.bulk.B_con").flux[X3DIR];

    pmb->par_for("B_field_bsqmax", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            double divb_local = ((F1(B1, k, j, i+1) - F1(B1, k, j, i)) / G.dx1v(i) +
                                 (F2(B2, k, j+1, i) - F2(B2, k, j, i)) / G.dx2v(j));
            if (ndim > 2) divb_local += (F3(B3, k, j+1, i) - F3(B3, k, j, i)) / G.dx3v(k);

            divB(k, j, i) = divb_local;
        }
    );
}

} // namespace B_FluxCT
