/* 
 *  File: b_flux_ct.cpp
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

#include "b_flux_ct.hpp"

// For their DivB estimate
#include "b_cd_glm.hpp"

#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"

using namespace parthenon;

// These are going to make this thing much more readable
#define B1 0
#define B2 1
#define B3 2

namespace B_FluxCT
{
// Local separate flux-CT calculation for 2D problems
// Called automatically from FluxCT
TaskStatus FluxCT2D(MeshBlockData<Real> *rc);

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t packages)
{
    auto pkg = std::make_shared<StateDescriptor>("B_FluxCT");
    Params &params = pkg->AllParams();

    // Diagnostic data
    int verbose = pin->GetOrAddInteger("debug", "verbose", 0);
    params.Add("verbose", verbose);
    int flag_verbose = pin->GetOrAddInteger("debug", "flag_verbose", 0);
    params.Add("flag_verbose", flag_verbose);
    int extra_checks = pin->GetOrAddInteger("debug", "extra_checks", 0);
    params.Add("extra_checks", extra_checks);

    bool fix_flux = pin->GetOrAddBoolean("b_field", "fix_polar_flux", true);
    params.Add("fix_polar_flux", fix_flux);
    // WARNING this disables constrained transport, so the field will quickly pick up a divergence
    bool disable_flux_ct = pin->GetOrAddBoolean("b_field", "disable_flux_ct", false);
    params.Add("disable_flux_ct", disable_flux_ct);

    std::vector<int> s_vector({3});

    MetadataFlag isPrimitive = packages.Get("GRMHD")->Param<MetadataFlag>("PrimitiveFlag");

    // B fields.  Primitive and conserved fields are related analytically
    // TODO could we get away with storing one?
    // TODO Metadata::WithFluxes when Parthenon 0.6
    Metadata m = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                           Metadata::Restart, Metadata::Conserved, Metadata::Vector, Metadata::WithFluxes}, s_vector);
    pkg->AddField("c.c.bulk.B_con", m);
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::FillGhost,
                  Metadata::Restart, isPrimitive, Metadata::Vector}, s_vector);
    pkg->AddField("c.c.bulk.B_prim", m);

    m = Metadata({Metadata::Cell, Metadata::Derived});
    pkg->AddField("c.c.bulk.divB_ct", m);

    pkg->FillDerivedBlock = B_FluxCT::UtoP;
    return pkg;
}

void UtoP(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();

    auto& B_U = rc->Get("c.c.bulk.B_con").data;
    auto& B_P = rc->Get("c.c.bulk.B_prim").data;

    auto& G = pmb->coords;

    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("UtoP_B", 0, NVEC-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_VARS {
            // Update the primitive B-fields
            B_P(p, k, j, i) = B_U(p, k, j, i) / G.gdet(Loci::center, j, i);
        }
    );
}

TaskStatus FluxCT(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    // Just use a completely separate implemenatation for 2D, it's faster & cleaner
    if (n3 == 1) return FluxCT2D(rc);

    FLAG("Flux CT");
    GridVars F1 = rc->Get("c.c.bulk.B_con").flux[X1DIR];
    GridVars F2 = rc->Get("c.c.bulk.B_con").flux[X2DIR];
    GridVars F3 = rc->Get("c.c.bulk.B_con").flux[X3DIR];
    GridScalar emf1("emf1", n3, n2, n1);
    GridScalar emf2("emf2", n3, n2, n1);
    GridScalar emf3("emf3", n3, n2, n1);

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    pmb->par_for("flux_ct_emf", ks, ke+1, js, je+1, is, ie+1,
        KOKKOS_LAMBDA_3D {
            emf3(k, j, i) =  0.25 * (F1(B2, k, j, i) + F1(B2, k, j-1, i) - F2(B1, k, j, i) - F2(B1, k, j, i-1));
            emf2(k, j, i) = -0.25 * (F1(B3, k, j, i) + F1(B3, k-1, j, i) - F3(B1, k, j, i) - F3(B1, k, j, i-1));
            emf1(k, j, i) =  0.25 * (F2(B3, k, j, i) + F2(B3, k-1, j, i) - F3(B2, k, j, i) - F3(B2, k, j-1, i));
        }
    );

    // Rewrite EMFs as fluxes, after Toth
    pmb->par_for("flux_ct_1", ks, ke, js, je, is, ie+1,
        KOKKOS_LAMBDA_3D {
            F1(B1, k, j, i) =  0.0;
            F1(B2, k, j, i) =  0.5 * (emf3(k, j, i) + emf3(k, j+1, i));
            F1(B3, k, j, i) = -0.5 * (emf2(k, j, i) + emf2(k+1, j, i));
        }
    );
    pmb->par_for("flux_ct_2", ks, ke, js, je+1, is, ie,
        KOKKOS_LAMBDA_3D {
            F2(B1, k, j, i) = -0.5 * (emf3(k, j, i) + emf3(k, j, i+1));
            F2(B2, k, j, i) =  0.0;
            F2(B3, k, j, i) =  0.5 * (emf1(k, j, i) + emf1(k+1, j, i));
        }
    );
    pmb->par_for("flux_ct_3", ks, ke+1, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            F3(B1, k, j, i) =  0.5 * (emf2(k, j, i) + emf2(k, j, i+1));
            F3(B2, k, j, i) = -0.5 * (emf1(k, j, i) + emf1(k, j+1, i));
            F3(B3, k, j, i) =  0.0;
        }
    );
    FLAG("CT Finished");

    return TaskStatus::complete;
}

// Local separate version for 2D
TaskStatus FluxCT2D(MeshBlockData<Real> *rc)
{
    FLAG("Flux CT 2D");
    auto pmb = rc->GetBlockPointer();
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);

    GridVars F1 = rc->Get("c.c.bulk.B_con").flux[X1DIR];
    GridVars F2 = rc->Get("c.c.bulk.B_con").flux[X2DIR];
    GridScalar emf("emf", n2, n1);

    FLAG("allocated");

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    pmb->par_for("flux_ct_emf", js, je+1, is, ie+1,
        KOKKOS_LAMBDA_2D {
            emf(j, i) =  0.25 * (F1(B2, 0, j, i) + F1(B2, 0, j-1, i)
                                 - F2(B1, 0, j, i) - F2(B1, 0, j, i-1));
        }
    );

    FLAG("EMFd");

    // Rewrite EMFs as fluxes, after Toth
    pmb->par_for("flux_ct", js, je, is, ie+1,
        KOKKOS_LAMBDA_2D {
            F1(B1, 0, j, i) = 0.0;
            F1(B2, 0, j, i) = 0.5 * (emf(j, i) + emf(j+1, i));
        }
    );
    pmb->par_for("flux_ct", js, je+1, is, ie,
        KOKKOS_LAMBDA_2D {
            F2(B1, 0, j, i) = -0.5 * (emf(j, i) + emf(j, i+1));
            F2(B2, 0, j, i) = 0.0;
        }
    );
    FLAG("CT 2D Finished");

    return TaskStatus::complete;
}

TaskStatus FixPolarFlux(MeshBlockData<Real> *rc)
{
    FLAG("Fixing polar B fluxes");
    auto pmb = rc->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const int ndim = pmb->pmy_mesh->ndim;

    GridVars F1, F2, F3;
    F1 = rc->Get("c.c.bulk.B_con").flux[X1DIR];
    if (ndim > 1) F2 = rc->Get("c.c.bulk.B_con").flux[X2DIR];
    if (ndim > 2) F3 = rc->Get("c.c.bulk.B_con").flux[X3DIR];
    int je_e = (ndim > 1) ? je + 1 : je;
    int ke_e = (ndim > 2) ? ke + 1 : ke;

    // Assuming the fluxes through the pole are 0,
    // make sure the polar EMFs are 0 when performing fluxCT
    if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::reflect)
    {
        pmb->par_for("fix_flux_b_l", ks, ke_e, js, js, is, ie+1,
            KOKKOS_LAMBDA_3D {
                F1(B2, k, j-1, i) = -F1(B2, k, js, i);
                if (ndim > 2) F3(B2, k, j-1, i) = -F3(B2, k, js, i);
            }
        );
    }

    if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::reflect)
    {
        pmb->par_for("fix_flux_b_r", ks, ke_e, je_e, je_e, is, ie+1,
            KOKKOS_LAMBDA_3D {
                F1(B2, k, j, i) = -F1(B2, k, je, i);
                if (ndim > 2) F3(B2, k, j, i) = -F3(B2, k, je, i);
            }
        );
    }

    FLAG("Fixed polar B");
    return TaskStatus::complete;
}

TaskStatus TransportB(MeshBlockData<Real> *rc)
{
    auto pmb = rc->GetBlockPointer();
    if (pmb->packages.Get("B_FluxCT")->Param<bool>("fix_polar_flux")) {
        FixPolarFlux(rc);
    }
    FluxCT(rc);
    return TaskStatus::complete;
}

double MaxDivB(MeshBlockData<Real> *rc, IndexDomain domain)
{
    FLAG("Calculating divB");
    auto pmb = rc->GetBlockPointer();
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return 0.;
    // Note the stencil of this function extends 1 left of the domain
    // We stay off the inner edge, using only zones on the grid where we apply fluxes/fluxCT
    is += 1;
    js += 1;
    if (ndim > 2) ks += 1;

    const double norm = (ndim > 2) ? 0.25 : 0.5;

    auto& G = pmb->coords;
    GridVars B_U = rc->Get("c.c.bulk.B_con").data;

    double max_divb;
    Kokkos::Max<double> max_reducer(max_divb);
    pmb->par_reduce("divB_max", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D_REDUCE {
            // 2D divergence, averaging to corners
            double term1 = B_U(B1, k, j, i)   + B_U(B1, k, j-1, i)
                         - B_U(B1, k, j, i-1) - B_U(B1, k, j-1, i-1);
            double term2 = B_U(B2, k, j, i)   + B_U(B2, k, j, i-1)
                         - B_U(B2, k, j-1, i) - B_U(B2, k, j-1, i-1);
            double term3 = 0.;
            if (ndim > 2) {
                // Average to corners in 3D, add 3rd flux
                term1 +=  B_U(B1, k-1, j, i)   + B_U(B1, k-1, j-1, i)
                        - B_U(B1, k-1, j, i-1) - B_U(B1, k-1, j-1, i-1);
                term2 +=  B_U(B2, k-1, j, i)   + B_U(B2, k-1, j, i-1)
                        - B_U(B2, k-1, j-1, i) - B_U(B2, k-1, j-1, i-1);
                term3 =   B_U(B3, k, j, i)     + B_U(B3, k, j-1, i)
                        + B_U(B3, k, j, i-1)   + B_U(B3, k, j-1, i-1)
                        - B_U(B3, k-1, j, i)   - B_U(B3, k-1, j-1, i)
                        - B_U(B3, k-1, j, i-1) - B_U(B3, k-1, j-1, i-1);
            }
            double local_divb = fabs(norm*term1/G.dx1v(i) + norm*term2/G.dx2v(j) + norm*term3/G.dx3v(k));
            if (local_divb > local_result) local_result = local_divb;
        }
    , max_reducer);

    return max_divb;
}

TaskStatus PostStepDiagnostics(Mesh *pmesh, ParameterInput *pin, const SimTime& tm)
{
    FLAG("Printing B field diagnostics");

    // Print this unless we quash everything
    if (pmesh->packages.Get("B_FluxCT")->Param<int>("verbose") > -1) {
        // This scheme really only guarantees divB on the physical grid
        // so we stick to verifying it over that
        IndexDomain domain = IndexDomain::interior;

        Real max_divb = 0;
        int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
        for (int i=0; i < nmb; ++i) {
            auto& pmb = pmesh->block_list[i];
            auto& rc = pmb->meshblock_data.Get();
            Real max_divb_l = B_FluxCT::MaxDivB(rc.get(), domain);

            if (max_divb_l > max_divb) max_divb = max_divb_l;
        }
        max_divb = MPIMax(max_divb);

        if(MPIRank0()) cout << "Max DivB: " << max_divb << endl;
    }

    FLAG("Printed")
    return TaskStatus::complete;
}

void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    FLAG("Calculating divB for output");
    IndexDomain domain = IndexDomain::entire;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return;
    // Note the stencil of this function extends 1 left of the domain
    // We stay off the inner edge, using only zones on the grid where we apply fluxes/fluxCT
    is += 1;
    js += 1;
    if (ndim > 2) ks += 1;
    const double norm = (ndim > 2) ? 0.25 : 0.5;

    auto& G = pmb->coords;
    auto rc = pmb->meshblock_data.Get().get();
    GridVars B_U = rc->Get("c.c.bulk.B_con").data;
    GridVars divB = rc->Get("c.c.bulk.divB_ct").data;

    pmb->par_for("divB_output", ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_3D {
            // 2D divergence, averaging to corners
            double term1 = B_U(B1, k, j, i)   + B_U(B1, k, j-1, i)
                         - B_U(B1, k, j, i-1) - B_U(B1, k, j-1, i-1);
            double term2 = B_U(B2, k, j, i)   + B_U(B2, k, j, i-1)
                         - B_U(B2, k, j-1, i) - B_U(B2, k, j-1, i-1);
            double term3 = 0.;
            if (ndim > 2) {
                // Average to corners in 3D, add 3rd flux
                term1 +=  B_U(B1, k-1, j, i)   + B_U(B1, k-1, j-1, i)
                        - B_U(B1, k-1, j, i-1) - B_U(B1, k-1, j-1, i-1);
                term2 +=  B_U(B2, k-1, j, i)   + B_U(B2, k-1, j, i-1)
                        - B_U(B2, k-1, j-1, i) - B_U(B2, k-1, j-1, i-1);
                term3 =   B_U(B3, k, j, i)     + B_U(B3, k, j-1, i)
                        + B_U(B3, k, j, i-1)   + B_U(B3, k, j-1, i-1)
                        - B_U(B3, k-1, j, i)   - B_U(B3, k-1, j-1, i)
                        - B_U(B3, k-1, j, i-1) - B_U(B3, k-1, j-1, i-1);
            }
            divB(k, j, i) = fabs(norm*term1/G.dx1v(i) + norm*term2/G.dx2v(j) + norm*term3/G.dx3v(k));
        }
    );

    FLAG("Output");
}

} // namespace B_FluxCT
