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
#include "b_cd.hpp"

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
    MetadataFlag isMHD = packages.Get("GRMHD")->Param<MetadataFlag>("MHDFlag");

    // B fields.  "Primitive" form is field, "conserved" is flux
    // Note: when changing metadata, keep these in lockstep with grmhd.cpp
    Metadata m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                 Metadata::Restart, Metadata::Conserved, isMHD, Metadata::WithFluxes, Metadata::Vector}, s_vector);
    pkg->AddField("cons.B", m);
    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived,
                  Metadata::Restart, isPrimitive, isMHD, Metadata::Vector}, s_vector);
    pkg->AddField("prims.B", m);

    m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("divB", m);

    pkg->FillDerivedMesh = B_FluxCT::FillDerivedMesh;
    pkg->FillDerivedBlock = B_FluxCT::FillDerivedBlock;
    pkg->PostStepDiagnosticsMesh = B_FluxCT::PostStepDiagnostics;

    // List (vector) of HistoryOutputVar that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    // In this package, we only care about MaxDivB
    // unless you want like, the median or something
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, MaxDivB, "MaxDivB"));
    // add callbacks for HST output identified by the `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

void UtoP(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    auto pmb = md->GetBlockData(0)->GetBlockPointer();

    auto B_U = md->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = md->PackVariables(std::vector<std::string>{"prims.B"});

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    IndexRange ib = bounds.GetBoundsI(domain);
    IndexRange jb = bounds.GetBoundsJ(domain);
    IndexRange kb = bounds.GetBoundsK(domain);
    IndexRange vec = IndexRange({0, B_U.GetDim(4)-1});
    IndexRange block = IndexRange({0, B_U.GetDim(5)-1});
    pmb->par_for("UtoP_B", block.s, block.e, vec.s, vec.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_MESH_VEC {
            // Update the primitive B-fields
            B_P(b, mu, k, j, i) = B_U(b, mu, k, j, i) / G.gdet(Loci::center, j, i);
        }
    );
}
void UtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    IndexRange ib = bounds.GetBoundsI(domain);
    IndexRange jb = bounds.GetBoundsJ(domain);
    IndexRange kb = bounds.GetBoundsK(domain);
    IndexRange vec = IndexRange({0, B_U.GetDim(4)-1});
    pmb->par_for("UtoP_B", vec.s, vec.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_VEC {
            // Update the primitive B-fields
            B_P(mu, k, j, i) = B_U(mu, k, j, i) / G.gdet(Loci::center, j, i);
        }
    );
}

TaskStatus FluxCT(MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    int n1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
    int n2 = pmb->cellbounds.ncellsj(IndexDomain::entire);
    int n3 = pmb->cellbounds.ncellsk(IndexDomain::entire);
    int nb = md->NumBlocks();
    const int ndim = pmesh->ndim;
    // No need for CT in 1D
    if (ndim < 2) return TaskStatus::complete;

    FLAG("Flux CT");

    auto B_F = md->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});
    // TODO make these a true Edge field of B_FluxCT? Could then output, use elsewhere, skip re-declaring
    GridScalar emf1("emf1", nb, n3, n2, n1);
    GridScalar emf2("emf2", nb, n3, n2, n1);
    GridScalar emf3("emf3", nb, n3, n2, n1);

    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    // Don't go beyond the grid in 2D
    int ke_l = (ndim > 2) ? ke + 1 : ke;

    // Calculate emf around each face
    FLAG("Calc EMFs");
    pmb->par_for("flux_ct_emf", 0, B_F.GetDim(5)-1, ks, ke_l, js, je+1, is, ie+1,
        KOKKOS_LAMBDA_MESH_3D {
            emf3(b, k, j, i) =  0.25 * (B_F(b).flux(X1DIR, B2, k, j, i) + B_F(b).flux(X1DIR, B2, k, j-1, i) -
                                        B_F(b).flux(X2DIR, B1, k, j, i) - B_F(b).flux(X2DIR, B1, k, j, i-1));
            if (ndim > 2) {
                emf2(b, k, j, i) = -0.25 * (B_F(b).flux(X1DIR, B3, k, j, i) + B_F(b).flux(X1DIR, B3, k-1, j, i) -
                                            B_F(b).flux(X3DIR, B1, k, j, i) - B_F(b).flux(X3DIR, B1, k, j, i-1));
                emf1(b, k, j, i) =  0.25 * (B_F(b).flux(X2DIR, B3, k, j, i) + B_F(b).flux(X2DIR, B3, k-1, j, i) -
                                            B_F(b).flux(X3DIR, B2, k, j, i) - B_F(b).flux(X3DIR, B2, k, j-1, i));
            }
        }
    );

    // Rewrite EMFs as fluxes, after Toth (2000)
    // Note that zeroing FX(BX) is *necessary* -- this flux gets filled by GetFlux,
    // And it's necessary to keep track of it for B_CD
    FLAG("Calc Fluxes");
#if FUSE_EMF_KERNELS
    pmb->par_for("flux_ct_all", 0, B_F.GetDim(5)-1, ks, ke_l, js, je+1, is, ie+1,
        KOKKOS_LAMBDA_MESH_3D {
            B_F(b).flux(X1DIR, B1, k, j, i) =  0.0;
            B_F(b).flux(X1DIR, B2, k, j, i) =  0.5 * (emf3(b, k, j, i) + emf3(b, k, j+1, i));

            B_F(b).flux(X2DIR, B1, k, j, i) = -0.5 * (emf3(b, k, j, i) + emf3(b, k, j, i+1));
            B_F(b).flux(X2DIR, B2, k, j, i) =  0.0;

            if (ndim > 2) {
                B_F(b).flux(X1DIR, B3, k, j, i) = -0.5 * (emf2(b, k, j, i) + emf2(b, k+1, j, i));
                B_F(b).flux(X2DIR, B3, k, j, i) =  0.5 * (emf1(b, k, j, i) + emf1(b, k+1, j, i));

                B_F(b).flux(X3DIR, B1, k, j, i) =  0.5 * (emf2(b, k, j, i) + emf2(b, k, j, i+1));
                B_F(b).flux(X3DIR, B2, k, j, i) = -0.5 * (emf1(b, k, j, i) + emf1(b, k, j+1, i));
                B_F(b).flux(X3DIR, B3, k, j, i) =  0.0;
            }
        }
    );
#else
    pmb->par_for("flux_ct_1", 0, B_F.GetDim(5)-1, ks, ke, js, je, is, ie+1,
        KOKKOS_LAMBDA_MESH_3D {
            B_F(b).flux(X1DIR, B1, k, j, i) =  0.0;
            B_F(b).flux(X1DIR, B2, k, j, i) =  0.5 * (emf3(b, k, j, i) + emf3(b, k, j+1, i));
            if (ndim > 2) B_F(b).flux(X1DIR, B3, k, j, i) = -0.5 * (emf2(b, k, j, i) + emf2(b, k+1, j, i));
        }
    );
    pmb->par_for("flux_ct_2", 0, B_F.GetDim(5)-1, ks, ke, js, je+1, is, ie,
        KOKKOS_LAMBDA_MESH_3D {
            B_F(b).flux(X2DIR, B1, k, j, i) = -0.5 * (emf3(b, k, j, i) + emf3(b, k, j, i+1));
            B_F(b).flux(X2DIR, B2, k, j, i) =  0.0;
            if (ndim > 2) B_F(b).flux(X2DIR, B3, k, j, i) =  0.5 * (emf1(b, k, j, i) + emf1(b, k+1, j, i));
        }
    );
    if (ndim > 2) {
        pmb->par_for("flux_ct_3", 0, B_F.GetDim(5)-1, ks, ke_l, js, je, is, ie,
            KOKKOS_LAMBDA_MESH_3D {
                B_F(b).flux(X3DIR, B1, k, j, i) =  0.5 * (emf2(b, k, j, i) + emf2(b, k, j, i+1));
                B_F(b).flux(X3DIR, B2, k, j, i) = -0.5 * (emf1(b, k, j, i) + emf1(b, k, j+1, i));
                B_F(b).flux(X3DIR, B3, k, j, i) =  0.0;
            }
        );
    }
#endif
    FLAG("CT Finished");

    return TaskStatus::complete;
}

TaskStatus FixPolarFlux(MeshData<Real> *md)
{
    FLAG("Fixing polar B fluxes");
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb0->cellbounds.is(domain), ie = pmb0->cellbounds.ie(domain);
    int js = pmb0->cellbounds.js(domain), je = pmb0->cellbounds.je(domain);
    int ks = pmb0->cellbounds.ks(domain), ke = pmb0->cellbounds.ke(domain);
    const int ndim = pmesh->ndim;

    int je_e = (ndim > 1) ? je + 1 : je;
    int ke_e = (ndim > 2) ? ke + 1 : ke;

    // Assuming the fluxes through the pole are 0,
    // make sure the polar EMFs are 0 when performing fluxCT
    // TODO only invoke one kernel? We avoid invocation except on boundaries anyway
    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();
        auto& B_F = rc->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});

        if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user)
        {
            pmb->par_for("fix_flux_b_l", ks, ke_e, js, js, is, ie+1,
                KOKKOS_LAMBDA_3D {
                    B_F.flux(X1DIR, B2, k, j-1, i) = -B_F.flux(X1DIR, B2, k, js, i);
                    if (ndim > 1) B_F.flux(X2DIR, B2, k, j, i) = 0;
                    if (ndim > 2) B_F.flux(X3DIR, B2, k, j-1, i) = -B_F.flux(X3DIR, B2, k, js, i);
                }
            );
        }
        if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user)
        {
            pmb->par_for("fix_flux_b_r", ks, ke_e, je_e, je_e, is, ie+1,
                KOKKOS_LAMBDA_3D {
                    B_F.flux(X1DIR, B2, k, j, i) = -B_F.flux(X1DIR, B2, k, je, i);
                    if (ndim > 1) B_F.flux(X2DIR, B2, k, j, i) = 0;
                    if (ndim > 2) B_F.flux(X3DIR, B2, k, j, i) = -B_F.flux(X3DIR, B2, k, je, i);
                }
            );
        }
    }

    FLAG("Fixed polar B");
    return TaskStatus::complete;
}

TaskStatus TransportB(MeshData<Real> *md)
{
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    if (pmb->packages.Get("B_FluxCT")->Param<bool>("fix_polar_flux")) {
        FixPolarFlux(md);
    }
    FluxCT(md);
    return TaskStatus::complete;
}

double MaxDivB(MeshData<Real> *md)
{
    FLAG("Calculating divB");
    auto pmesh = md->GetMeshPointer();
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    IndexDomain domain = IndexDomain::interior;
    int is = pmb->cellbounds.is(domain), ie = pmb->cellbounds.ie(domain);
    int js = pmb->cellbounds.js(domain), je = pmb->cellbounds.je(domain);
    int ks = pmb->cellbounds.ks(domain), ke = pmb->cellbounds.ke(domain);
    const int ndim = pmesh->ndim;
    if (ndim < 2) return 0.;
    // Note this is a stencil-4 (or -8) function, which would involve zones outside the
    // domain unless we stay off the left edges
    is += 1;
    js += 1;
    if (ndim > 2) ks += 1;

    const double norm = (ndim > 2) ? 0.25 : 0.5;

    const auto& G = pmb->coords;

    // Note when packing that declaring std::vector<std::string> is *crucial*
    // Otherwise Parthenon will use the wrong constructor and pack everything badly
    auto B_U = md->PackVariables(std::vector<std::string>{"cons.B"});

    double max_divb;
    Kokkos::Max<double> max_reducer(max_divb);
    pmb->par_reduce("divB_max", 0, B_U.GetDim(5)-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA_MESH_3D_REDUCE {
            // 2D divergence, averaging to corners
            double term1 = B_U(b, B1, k, j, i)   + B_U(b, B1, k, j-1, i)
                         - B_U(b, B1, k, j, i-1) - B_U(b, B1, k, j-1, i-1);
            double term2 = B_U(b, B2, k, j, i)   + B_U(b, B2, k, j, i-1)
                         - B_U(b, B2, k, j-1, i) - B_U(b, B2, k, j-1, i-1);
            double term3 = 0.;
            if (ndim > 2) {
                // Average to corners in 3D, add 3rd flux
                term1 +=  B_U(b, B1, k-1, j, i)   + B_U(b, B1, k-1, j-1, i)
                        - B_U(b, B1, k-1, j, i-1) - B_U(b, B1, k-1, j-1, i-1);
                term2 +=  B_U(b, B2, k-1, j, i)   + B_U(b, B2, k-1, j, i-1)
                        - B_U(b, B2, k-1, j-1, i) - B_U(b, B2, k-1, j-1, i-1);
                term3 =   B_U(b, B3, k, j, i)     + B_U(b, B3, k, j-1, i)
                        + B_U(b, B3, k, j, i-1)   + B_U(b, B3, k, j-1, i-1)
                        - B_U(b, B3, k-1, j, i)   - B_U(b, B3, k-1, j-1, i)
                        - B_U(b, B3, k-1, j, i-1) - B_U(b, B3, k-1, j-1, i-1);
            }
            double local_divb = fabs(norm*term1/G.dx1v(i) + norm*term2/G.dx2v(j) + norm*term3/G.dx3v(k));
            if (local_divb > local_result) local_result = local_divb;
        }
    , max_reducer);

    return max_divb;
}

TaskStatus PostStepDiagnostics(const SimTime& tm, MeshData<Real> *md)
{
    FLAG("Printing B field diagnostics");
    if (md->NumBlocks() > 0) {
        auto pmb = md->GetBlockData(0)->GetBlockPointer();

        // Print this unless we quash everything
        if (pmb->packages.Get("B_FluxCT")->Param<int>("verbose") >= 0) {
            FLAG("Printing divB");
            Real max_divb = B_FluxCT::MaxDivB(md);
            max_divb = MPIMax(max_divb);

            if(MPIRank0()) cout << "Max DivB: " << max_divb << endl;
        }
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

    const auto& G = pmb->coords;
    auto rc = pmb->meshblock_data.Get().get();
    GridVars B_U = rc->Get("cons.B").data;
    GridVars divB = rc->Get("divB").data;

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
