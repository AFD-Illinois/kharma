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

#include "decs.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
#include "reductions.hpp"

using namespace parthenon;

namespace B_FluxCT
{

// Reductions: phi uses global machinery, but divB is too 
// Can also sum the hemispheres independently to be fancy (TODO?)
KOKKOS_INLINE_FUNCTION Real phi(REDUCE_FUNCTION_ARGS_EH)
{
    // \Phi == \int |*F^1^0| * gdet * dx2 * dx3 == \int |B1| * gdet * dx2 * dx3
    return 0.5 * m::abs(U(m_u.B1, k, j, i)); // factor of gdet already in cons.B
}

Real ReducePhi0(MeshData<Real> *md)
{
    return Reductions::EHReduction(md, UserHistoryOperation::sum, phi, 0);
}
Real ReducePhi5(MeshData<Real> *md)
{
    return Reductions::EHReduction(md, UserHistoryOperation::sum, phi, 5);
}

std::shared_ptr<KHARMAPackage> Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("B_FluxCT");
    Params &params = pkg->AllParams();

    // Diagnostic & inadvisable flags
    // This enables flux corrections to ensure divB preservation even with zero flux of B2 on the polar "face."
    // It effectively makes the pole a superconducting rod
    bool spherical = pin->GetBoolean("coordinates", "spherical"); //  TODO could do package
    bool fix_polar_flux = pin->GetOrAddBoolean("b_field", "fix_polar_flux", spherical);
    params.Add("fix_polar_flux", fix_polar_flux);
    // These options do the same to the inner and outer edges.  They are NOT as well tested, and it's
    // questionable whether you'd want to do this anyway.
    // They would require at least B1 to be reflected across the EH, probably straight-up reflecting conditions
    bool fix_eh_flux = pin->GetOrAddBoolean("b_field", "fix_eh_flux", false);
    params.Add("fix_eh_flux", fix_eh_flux);
    bool fix_exterior_flux = pin->GetOrAddBoolean("b_field", "fix_exterior_flux", false);
    params.Add("fix_exterior_flux", fix_exterior_flux);
    // This option uses a different (better but slower) fix which allows magnetic flux through the X1 boundaries,
    // at the cost of some speed and potentially some instability due to the non-local nature of the solve.
    // Much better tested than above options
    bool fix_x1_flux = pin->GetOrAddBoolean("b_field", "fix_x1_flux", false);
    params.Add("fix_x1_flux", fix_x1_flux);

    // KHARMA requires some kind of field transport if there is a magnetic field allocated
    // Use this if you actually want to disable all magnetic field flux corrections,
    // and allow a field divergence to grow unchecked, usually for debugging or comparison reasons
    bool disable_flux_ct = pin->GetOrAddBoolean("b_field", "disable_flux_ct", false);
    params.Add("disable_flux_ct", disable_flux_ct);

    // Driver type & implicit marker
    // By default, solve B explicitly
    auto& driver = packages->Get("Driver")->AllParams();
    bool implicit_b = pin->GetOrAddBoolean("b_field", "implicit", false);
    params.Add("implicit", implicit_b);

    // Update variable numbers
    if (implicit_b) {
        int n_current = driver.Get<int>("n_implicit_vars");
        driver.Update("n_implicit_vars", n_current+3);
    } else {
        int n_current = driver.Get<int>("n_explicit_vars");
        driver.Update("n_explicit_vars", n_current+3);
    }

    params.Add("divb_reducer", AllReduce<Real>());

    // FIELDS

    std::vector<int> s_vector({NVEC});

    // Mark if we're evolving implicitly
    MetadataFlag areWeImplicit = (implicit_b) ? Metadata::GetUserFlag("Implicit")
                                                : Metadata::GetUserFlag("Explicit");

    // Flags for B fields.  "Primitive" form is field, "conserved" is flux
    std::vector<MetadataFlag> flags_prim = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::GetUserFlag("Primitive"),
                                            Metadata::Restart, Metadata::GetUserFlag("MHD"), areWeImplicit, Metadata::Vector};
    std::vector<MetadataFlag> flags_cons = {Metadata::Real, Metadata::Cell, Metadata::Independent, Metadata::Conserved,
                                            Metadata::WithFluxes, Metadata::FillGhost, Metadata::GetUserFlag("MHD"), areWeImplicit, Metadata::Vector};

    auto m = Metadata(flags_prim, s_vector);
    pkg->AddField("prims.B", m);
    m = Metadata(flags_cons, s_vector);
    pkg->AddField("cons.B", m);

    // Hyerin (12/19/22)
    if (pin->GetString("parthenon/job", "problem_id") == "resize_restart_kharma") {
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::FillGhost, Metadata::Vector});
        pkg->AddField("B_Save", m);
    }

    // We exist basically to do this
    pkg->FixFlux = B_FluxCT::FixFlux;

    // Also ensure that prims get filled, *if* we're evolved explicitly
    if (!implicit_b) {
        pkg->MeshUtoP = B_FluxCT::MeshUtoP;
        pkg->BlockUtoP = B_FluxCT::BlockUtoP;
    }

    // Register the other callbacks
    pkg->PostStepDiagnosticsMesh = B_FluxCT::PostStepDiagnostics;

    // The definition of MaxDivB we care about actually changes per-transport,
    // so calculating it is handled by the transport package
    // We'd only ever need to declare or calculate divB for output (getting the max is independent)
    if (KHARMA::FieldIsOutput(pin, "divB")) {
        pkg->BlockUserWorkBeforeOutput = B_FluxCT::FillOutput;
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("divB", m);
    }

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    parthenon::HstVar_list hst_vars = {};
    hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, B_FluxCT::MaxDivB, "MaxDivB"));
    // Event horizon magnetization.  Might be the same or different for different representations?
    if (pin->GetBoolean("coordinates", "spherical")) {
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, ReducePhi0, "Phi_0"));
        hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, ReducePhi5, "Phi_EH"));
    }
    // add callbacks for HST output to the Params struct, identified by the `hist_param_key`
    pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

// TODO template and use as a model for future
void MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    Flag(md, "B UtoP Mesh");
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    const auto& B_U = md->PackVariables(std::vector<std::string>{"cons.B"});
    const auto& B_P = md->PackVariables(std::vector<std::string>{"prims.B"});

    auto bounds = coarse ? pmb0->c_cellbounds : pmb0->cellbounds;
    IndexRange ib = bounds.GetBoundsI(domain);
    IndexRange jb = bounds.GetBoundsJ(domain);
    IndexRange kb = bounds.GetBoundsK(domain);
    IndexRange vec = IndexRange{0, B_U.GetDim(4)-1};
    IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    pmb0->par_for("UtoP_B", block.s, block.e, vec.s, vec.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int& b, const int &mu, const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords(b);
            // Update the primitive B-fields
            B_P(b, mu, k, j, i) = B_U(b, mu, k, j, i) / G.gdet(Loci::center, j, i);
        }
    );
}
void BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "B UtoP Block");
    auto pmb = rc->GetBlockPointer();

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});

    const auto& G = pmb->coords;

    auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(domain);
    const IndexRange jb = bounds.GetBoundsJ(domain);
    const IndexRange kb = bounds.GetBoundsK(domain);
    const IndexRange vec = IndexRange({0, B_U.GetDim(4)-1});

    pmb->par_for("UtoP_B", vec.s, vec.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &mu, const int &k, const int &j, const int &i) {
            // Update the primitive B-fields
            B_P(mu, k, j, i) = B_U(mu, k, j, i) / G.gdet(Loci::center, j, i);
        }
    );
}

void FixFlux(MeshData<Real> *md)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    auto& params = pmb0->packages.Get("B_FluxCT")->AllParams();
    if (params.Get<bool>("fix_polar_flux")) {
        FixBoundaryFlux(md, IndexDomain::inner_x2, false);
        FixBoundaryFlux(md, IndexDomain::outer_x2, false);
    }
    if (params.Get<bool>("fix_x1_flux")) {
        FixX1Flux(md);
    }
    if (params.Get<bool>("fix_eh_flux")) {
        FixBoundaryFlux(md, IndexDomain::inner_x1, false);
    }
    if (params.Get<bool>("fix_exterior_flux")) {
        FixBoundaryFlux(md, IndexDomain::outer_x1, false);
    }
    FluxCT(md);
}

// INTERNAL

void FluxCT(MeshData<Real> *md)
{
    Flag(md, "Flux CT");
    // Pointers
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    // Exit on trivial operations
    const int ndim = pmesh->ndim;
    if (ndim < 2) return;

    // Pack variables
    const auto& B_F = md->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});

    // Get sizes
    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, B_F.GetDim(5)-1};
    // One zone halo on the *right only*, except for k in 2D
    const IndexRange il = IndexRange{ib.s, ib.e + 1};
    const IndexRange jl = IndexRange{jb.s, jb.e + 1};
    const IndexRange kl = (ndim > 2) ? IndexRange{kb.s, kb.e + 1} : kb;

    // Declare temporaries
    // TODO make these a true Edge field when that's available
    const int n1 = pmb0->cellbounds.ncellsi(IndexDomain::entire);
    const int n2 = pmb0->cellbounds.ncellsj(IndexDomain::entire);
    const int n3 = pmb0->cellbounds.ncellsk(IndexDomain::entire);
    const int nb = md->NumBlocks();
    GridScalar emf1("emf1", nb, n3, n2, n1);
    GridScalar emf2("emf2", nb, n3, n2, n1);
    GridScalar emf3("emf3", nb, n3, n2, n1);

    // Calculate emf around each face
    Flag(md, "Calc EMFs");
    pmb0->par_for("flux_ct_emf", block.s, block.e, kl.s, kl.e, jl.s, jl.e, il.s, il.e,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            emf3(b, k, j, i) =  0.25 * (B_F(b).flux(X1DIR, V2, k, j, i) + B_F(b).flux(X1DIR, V2, k, j-1, i) -
                                        B_F(b).flux(X2DIR, V1, k, j, i) - B_F(b).flux(X2DIR, V1, k, j, i-1));
            if (ndim > 2) {
                emf2(b, k, j, i) = -0.25 * (B_F(b).flux(X1DIR, V3, k, j, i) + B_F(b).flux(X1DIR, V3, k-1, j, i) -
                                            B_F(b).flux(X3DIR, V1, k, j, i) - B_F(b).flux(X3DIR, V1, k, j, i-1));
                emf1(b, k, j, i) =  0.25 * (B_F(b).flux(X2DIR, V3, k, j, i) + B_F(b).flux(X2DIR, V3, k-1, j, i) -
                                            B_F(b).flux(X3DIR, V2, k, j, i) - B_F(b).flux(X3DIR, V2, k, j-1, i));
            }
        }
    );

    // Rewrite EMFs as fluxes, after Toth (2000)
    // Note that zeroing FX(BX) is *necessary* -- this flux gets filled by GetFlux
    Flag(md, "Calc Fluxes");

    // Note these each have different domains, eg il vs ib.  The former extends one index farther if appropriate
    pmb0->par_for("flux_ct_1", block.s, block.e, kb.s, kb.e, jb.s, jb.e, il.s, il.e,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            B_F(b).flux(X1DIR, V1, k, j, i) =  0.0;
            B_F(b).flux(X1DIR, V2, k, j, i) =  0.5 * (emf3(b, k, j, i) + emf3(b, k, j+1, i));
            if (ndim > 2) B_F(b).flux(X1DIR, V3, k, j, i) = -0.5 * (emf2(b, k, j, i) + emf2(b, k+1, j, i));
            
            /*
            if (k <15 && k>13 && j>jb.s-1 && j<jb.s+2 && (i==il.s || i==il.e)) {
                printf("HYERIN: b,i,j,k = (%i %i %i %i) effective x1flux = ( %g %g %g ) \n",b, i, j, k, B_F(b).flux(X1DIR,V1,k,j,i), B_F(b).flux(X1DIR,V2,k,j,i), B_F(b).flux(X1DIR,V3,k,j,i));
                printf("HYERIN: b,i,j,k = (%i %i %i %i) effective x2flux = ( %g %g %g ) \n",b, i, j, k, B_F(b).flux(X2DIR,V1,k,j,i-1), B_F(b).flux(X2DIR,V2,k,j,i-1), B_F(b).flux(X2DIR,V3,k,j,i-1));
            }
            */
        }
    );
    pmb0->par_for("flux_ct_2", block.s, block.e, kb.s, kb.e, jl.s, jl.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
            B_F(b).flux(X2DIR, V1, k, j, i) = -0.5 * (emf3(b, k, j, i) + emf3(b, k, j, i+1));
            B_F(b).flux(X2DIR, V2, k, j, i) =  0.0;
            if (ndim > 2) B_F(b).flux(X2DIR, V3, k, j, i) =  0.5 * (emf1(b, k, j, i) + emf1(b, k+1, j, i));
        }
    );
    if (ndim > 2) {
        pmb0->par_for("flux_ct_3", block.s, block.e, kl.s, kl.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int& b, const int &k, const int &j, const int &i) {
                B_F(b).flux(X3DIR, V1, k, j, i) =  0.5 * (emf2(b, k, j, i) + emf2(b, k, j, i+1));
                B_F(b).flux(X3DIR, V2, k, j, i) = -0.5 * (emf1(b, k, j, i) + emf1(b, k, j+1, i));
                B_F(b).flux(X3DIR, V3, k, j, i) =  0.0;
            }
        );
    }
    
    Flag(md, "CT Finished");
}

void FixBoundaryFlux(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    Flag(md, "Fixing polar B fluxes");
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = pmesh->block_list[0];
    const int ndim = pmesh->ndim;
    if (ndim < 2) return;

    auto bounds = coarse ? pmb0->c_cellbounds : pmb0->cellbounds;
    const IndexRange ib = bounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jb = bounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = bounds.GetBoundsK(IndexDomain::interior);

    // Imagine a corner of the domain, with ghost and physical zones
    // as below, denoted w/'g' and 'p' respectively.
    // 
    // g | p | p
    //-----------
    // g | p | p
    //xxx--------
    // g | g | g
    // 
    // The flux through 'x' is not important for updating a physical zone,
    // as it does not border any.  However, FluxCT considers it when updating
    // nearby fluxes, two of which affect physical zones.
    // Therefore in e.g. X1 faces, we need to update fluxes on the domain:
    // [0,N1+1],[-1,N2+1],[-1,N3+1]
    // These indices arrange for that.

    // For faces
    const IndexRange ibf = IndexRange{ib.s, ib.e + 1};
    const IndexRange jbf = IndexRange{jb.s, jb.e + 1};
    // Won't need X3 faces
    //const IndexRange kbf = IndexRange{kb.s, kb.e + (ndim > 2)};
    // For sides
    const IndexRange ibs = IndexRange{ib.s - 1, ib.e + 1};
    const IndexRange jbs = IndexRange{jb.s - (ndim > 1), jb.e + (ndim > 1)};
    const IndexRange kbs = IndexRange{kb.s - (ndim > 2), kb.e + (ndim > 2)};

    // Make sure the polar EMFs are 0 when performing fluxCT
    // Compare this section with calculation of emf3 in FluxCT:
    // these changes ensure that boundary emfs emf3(i,js,k)=0, etc.
    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();
        auto& B_F = rc->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});

        if (domain == IndexDomain::inner_x2 &&
            pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
            pmb->par_for("fix_flux_b_l", kbs.s, kbs.e, jbf.s, jbf.s, ibs.s, ibs.e,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    B_F.flux(X2DIR, V1, k, j, i) = 0.;
                    B_F.flux(X2DIR, V3, k, j, i) = 0.;
                    B_F.flux(X1DIR, V2, k, j - 1, i) = -B_F.flux(X1DIR, V2, k, j, i);
                    if (ndim > 2) B_F.flux(X3DIR, V2, k, j - 1, i) = -B_F.flux(X3DIR, V2, k, j, i);
                }
            );
        }

        if (domain == IndexDomain::outer_x2 &&
            pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
            pmb->par_for("fix_flux_b_r", kbs.s, kbs.e, jbf.e, jbf.e, ibs.s, ibs.e,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    B_F.flux(X2DIR, V1, k, j, i) = 0.;
                    B_F.flux(X2DIR, V3, k, j, i) = 0.;
                    B_F.flux(X1DIR, V2, k, j, i) = -B_F.flux(X1DIR, V2, k, j - 1, i);
                    if (ndim > 2) B_F.flux(X3DIR, V2, k, j, i) = -B_F.flux(X3DIR, V2, k, j - 1, i);
                }
            );
        }

        // TODO the following is dead without an accompanying inverted-B1 or reflecting boundary
        // for magnetic fields in KBoundaries. (Unless you want to reflect everything, which, don't.)
        // Keeping special boundaries for this silly test kicking around KBoundaries was ugly, so they're
        // removed.  Could investigate further when Parthenon's better boundary support appears.

        // We can do the same with the outflow bounds. Kind of.
        // See, actually, outflow bounds will *always* generate divergence on the domain face.
        // So if we want to clean it up here, we would need to arrange for B1 to be inverted in ghost cells.
        // This is no longer pure outflow, but might be thought of as a "nicer" version of
        // reflecting conditions:
        // 1. Since B1 is inverted, B1 on the domain face will tend to 0 (it's not quite reflected, but basically)
        //    (obviously don't enable this for monopole test problems!)
        // 2. However, B2 and B3 are normal outflow conditions -- despite the fluxes here, the outflow
        //    conditions will set them equal to the last zone.
        if (domain == IndexDomain::inner_x1 &&
            pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
            pmb->par_for("fix_flux_b_in", kbs.s, kbs.e, jbs.s, jbs.e, ibf.s, ibf.s,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    B_F.flux(X1DIR, V2, k, j, i) = 0.;
                    B_F.flux(X1DIR, V3, k, j, i) = 0.;
                    B_F.flux(X2DIR, V1, k, j, i - 1) = -B_F.flux(X2DIR, V1, k, j, i);
                    if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i - 1) = -B_F.flux(X3DIR, V1, k, j, i);
                }
            );
        }

        if (domain == IndexDomain::outer_x1 &&
            pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
            pmb->par_for("fix_flux_b_out", kbs.s, kbs.e, jbs.s, jbs.e, ibf.e, ibf.e,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    B_F.flux(X1DIR, V2, k, j, i) = 0.;
                    B_F.flux(X1DIR, V3, k, j, i) = 0.;
                    B_F.flux(X2DIR, V1, k, j, i) = -B_F.flux(X2DIR, V1, k, j, i - 1);
                    if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i) = -B_F.flux(X3DIR, V1, k, j, i - 1);
                }
            );
        }

    }

    Flag(md, "Fixed polar B");
}

TaskStatus FixX1Flux(MeshData<Real> *md)
{
    Flag(md, "Fixing X1 fluxes");
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
    
    IndexDomain domain = IndexDomain::interior;
    int is = pmb0->cellbounds.is(domain), ie = pmb0->cellbounds.ie(domain);
    int js = pmb0->cellbounds.js(domain), je = pmb0->cellbounds.je(domain);
    int js_all = pmb0->cellbounds.js(IndexDomain::entire), je_all = pmb0->cellbounds.je(IndexDomain::entire); // added by Hyerin (12/28/22)
    int ks = pmb0->cellbounds.ks(domain), ke = pmb0->cellbounds.ke(domain);
    int ks_all = pmb0->cellbounds.ks(IndexDomain::entire), ke_all = pmb0->cellbounds.ke(IndexDomain::entire); // added by Hyerin (12/28/22)
    const int ndim = pmesh->ndim;

    int je_e = (ndim > 1) ? je + 1 : je;
    //int je_e = (ndim > 1) ? je_all + 1 : je_all; // test Hyerin(12/28/22)
    int ke_e = (ndim > 2) ? ke + 1 : ke;
    //int ke_e = (ndim > 2) ? ke_all + 1 : ke_all; // test Hyerin (12/28/22)
    int js_new, je_new; // Hyerin (02/21/23)
    bool in_x2, out_x2; // Hyerin
    
    Real x1min = pmb0->packages.Get("GRMHD")->Param<Real>("x1min"); //Hyerin (01/31/23)

    // (03/08/23) places to store
    //const int n1 = pmb0->cellbounds.ncellsi(IndexDomain::entire);
    //const int n2 = pmb0->cellbounds.ncellsj(IndexDomain::entire);
    //const int n3 = pmb0->cellbounds.ncellsk(IndexDomain::entire);
    //GridScalar B_F_X2_V1("B_F_X2_V1", n3, n2, n1);  // for B_F.flux(X2DIR,V1,k,j,i)
    //GridScalar B_F_X3_V1("B_F_X3_V1", n3, n2, n1);  // for B_F.flux(X3DIR,V1,k,j,i)
    //auto B_F_X2_V1_host = B_F_X2_V1.GetHostMirror();
    //auto B_F_X3_V1_host = B_F_X3_V1.GetHostMirror();
    //auto B_F_host = x2_fill_device.GetHostMirror();
    GridVector F1, F2, F3;

    // TODO(BSP) try to eliminate full-array copies. Host-parallel applications to inner/outer?
    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();
        auto& B_F = rc->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});

        // (03/08/23)
        F1 = rc->Get("cons.B").flux[X1DIR]; // B_F.flux(X1DIR,v,k,j,i)
        F2 = rc->Get("cons.B").flux[X2DIR]; // B_F.flux(X2DIR,v,k,j,i)
        F3 = rc->Get("cons.B").flux[X3DIR]; // B_F.flux(X3DIR,v,k,j,i)
        auto F1_host=F1.GetHostMirrorAndCopy();
        auto F2_host=F2.GetHostMirrorAndCopy();
        auto F3_host=F3.GetHostMirrorAndCopy();
        
        // update the j and k bounds (Hyerin 02/21/23)
        js_new = js+1; //js-1;
        je_new = je_e+1; //je_e+1;
        in_x2 = false;
        out_x2 = false;
        if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
            in_x2 = true;
            js_new = js;
        }
        if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
            out_x2 = true;
            je_new = je; //_e;
        }

        //printf("HYERIN: test F1V2 %g\n",F1_host(V2,30,30,is));
        //pmb->par_for("test", 30,30,30,30,is,is,
        //    KOKKOS_LAMBDA_3D {
        //        printf("HYERIN: test B_F(X1DIR,V2) %g, F1V2 %g \n",B_F.flux(X1DIR,V2,k,j,i),F1(V2,k,j,i));
        //    }
        //);

        //added by Hyerin (12/23/22) TODO: it has to ask if x2 boundary is inner_x2 or outer_x2 and update the jj bounds
        if ((pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) && (x1min>1) ) // only apply fix flux for inner bc when it is far from the EH
        {   
            for (int ktemp = ks_all+2; ktemp <=ke_all; ktemp++) {
              for (int jtemp = js_new; jtemp <= je_new; jtemp++) {
            //pmb->par_for("fix_flux_b_l", ktemp, ktemp, jtemp, jtemp, is, is, // Hyerin (02/20/23) for 3rd prescription, sequential
            //pmb->par_for("fix_flux_b_l", ks_all+2, ke_all, js_new, je_new, is, is, // Hyerin (02/20/23) for 3rd prescription
            //pmb->par_for("fix_flux_b_l", ks_all+1, ke_all+1, js_all+1, je_all+1, is, is, // Hyerin (12/28/22) for 1st & 2nd prescription
                // KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    /* 1st prescription to make the X1DIR flux = 0
                    B_F.flux(X2DIR, V1, k, j, i-1) = -B_F.flux(X2DIR, V1, k, j, is);
                    if (ndim > 1) VLOOP B_F.flux(X1DIR, V1+v, k, j, i) = 0;
                    if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i-1) = -B_F.flux(X3DIR, V1, k, j, is);
                    */
                    // (02/06/23) 2nd prescription that allows nonzero flux across X1 boundary but still keeps divB=0 (turns out effectively to have 0 flux)
                    //if (ndim > 1) B_F.flux(X2DIR, V1, k, j, i-1) = -B_F.flux(X2DIR, V1, k, j, is) + B_F.flux(X1DIR, V2, k, j, is) + B_F.flux(X1DIR, V2, k, j-1, is);
                    //if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i-1) = -B_F.flux(X3DIR, V1, k, j, is) + B_F.flux(X1DIR, V3, k, j, is) + B_F.flux(X1DIR, V3, k-1, j, is);
                    //
                    // (02/20/23) 3rd prescription that is similar to 2nd prescription but not local and nonzero effective flux 
                    if (ndim > 1) {
                        //B_F.flux(X2DIR, V1, k, j, i-1) = -B_F.flux(X2DIR, V1, k, j, is) + B_F.flux(X1DIR, V2, k, j, is) - B_F.flux(X1DIR, V2, k, j-2, is) + B_F.flux(X2DIR, V1, k, j-1, is) + B_F.flux(X2DIR, V1, k, j-1, is-1);
                        F2_host(V1, ktemp, jtemp, is-1) = -F2_host(V1, ktemp, jtemp, is) + F1_host(V2, ktemp, jtemp, is) - F1_host(V2, ktemp, jtemp-2, is) + F2_host(V1, ktemp, jtemp-1, is) + F2_host(V1, ktemp, jtemp-1, is-1);
                    }
                    if (ndim > 2) {
                        //B_F.flux(X3DIR, V1, k, j, i-1) = -B_F.flux(X3DIR, V1, k, j, is) + B_F.flux(X1DIR, V3, k, j, is) - B_F.flux(X1DIR, V3, k-2, j, is) + B_F.flux(X3DIR, V1, k-1, j, is) + B_F.flux(X3DIR, V1, k-1, j, is-1);
                        F3_host(V1, ktemp, jtemp, is-1) = -F3_host(V1, ktemp, jtemp, is) + F1_host(V3, ktemp, jtemp, is) - F1_host(V3, ktemp-2, jtemp, is) + F3_host(V1, ktemp-1, jtemp, is) + F3_host(V1, ktemp-1, jtemp, is-1);
                    }

                    //if (in_x2 && (j==js)) {// (corners are tricky so let's just initialize)
                    if (in_x2 && (jtemp==js)) {// (corners are tricky so let's just initialize)
                        //B_F.flux(X2DIR, V1,k,j,i-1) = -B_F.flux(X1DIR,V2,k,j,i+1) -B_F.flux(X1DIR,V2,k,j-1,i+1);
                        F2_host(V1,ktemp,jtemp,is-1) = -F1_host(V2,ktemp,jtemp,is+1) -F1_host(V2,ktemp,jtemp-1,is+1);
                        //B_F.flux(X2DIR, V1,k,j,i) = -0.5*B_F.flux(X2DIR,V1,k,j,i-1);
                        F2_host(V1,ktemp,jtemp,is) = -0.5*F2_host(V1,ktemp,jtemp,is-1);
                    }
                    //if (out_x2 && (j==je_e)) {// (corners are tricky)
                    if (out_x2 && (jtemp==je_e)) {// (corners are tricky) ( so maybe just don't touch it...? (03/12/23)
                        //B_F.flux(X2DIR, V1, k, j, i) = -B_F.flux(X2DIR, V1, k, je, is) - B_F.flux(X2DIR, V1, k, je, is-1) 
                        //                                +B_F.flux(X1DIR, V2, k, je, is) + B_F.flux(X1DIR, V2, k, je-1, is);
                        //B_F.flux(X2DIR, V1, k, j, i-1) = -2.*B_F.flux(X1DIR, V2, k, je-1, is) -B_F.flux(X1DIR, V2, k, je, is) + B_F.flux(X1DIR, V2, k, je+1, is)
                        //                                +2.*B_F.flux(X2DIR, V1, k, je, is) + 2.*B_F.flux(X2DIR, V1, k, je, is-1);
                        //B_F.flux(X1DIR,V2,k,j-1,i) = -B_F.flux(X1DIR,V2,k,je-1,i)+B_F.flux(X2DIR,V1,k,je,i)+B_F.flux(X2DIR,V1,k,je,i-1);
                        F1_host(V2,ktemp,jtemp-1,is) = -F1_host(V2,ktemp,je-1,is)+F2_host(V1,ktemp,je,is)+F2_host(V1,ktemp,je,is-1);
                        //B_F.flux(X1DIR,V2,k,j,i) = -B_F.flux(X1DIR,V2,k,je,i);
                        F1_host(V2,ktemp,jtemp,is) = -F1_host(V2,ktemp,je,is);
                    }
                    
                    
                //}
           // );
              }
            }

        }
        if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user)
        {
            for (int ktemp = ks_all+2; ktemp <=ke_all; ktemp++) {
              for (int jtemp = js_new; jtemp <= je_new; jtemp++) {
            //pmb->par_for("fix_flux_b_r", ktemp, ktemp, jtemp, jtemp, ie+1, ie+1, // Hyerin (02/20/23) for 3rd prescription, sequential
            //pmb->par_for("fix_flux_b_r", ks_all+2, ke_all, js_new, je_new, ie+1, ie+1, // Hyerin (02/20/23) for 3rd prescription
            //pmb->par_for("fix_flux_b_r", ks_all+1, ke_all+1, js_all+1, je_all+1, ie+1, ie+1, // Hyerin (12/28/22) for 1st & 2nd prescription
                // KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    /* 1st prescription to make the X1DIR flux = 0
                    B_F.flux(X2DIR, V1, k, j, i) = -B_F.flux(X2DIR, V1, k, j, ie);
                    if (ndim > 1) VLOOP B_F.flux(X1DIR, V1+v, k, j, i) = 0;
                    if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i) = -B_F.flux(X3DIR, V1, k, j, ie);
                    */
                    // (02/06/23) 2nd prescription that allows nonzero flux across X1 boundary but still keeps divB=0
                    //if (ndim > 1) B_F.flux(X2DIR, V1, k, j, i) = -B_F.flux(X2DIR, V1, k, j, ie) + B_F.flux(X1DIR, V2, k, j, i) + B_F.flux(X1DIR, V2, k, j-1, i);
                    //if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i) = -B_F.flux(X3DIR, V1, k, j, ie) + B_F.flux(X1DIR, V3, k, j, i) + B_F.flux(X1DIR, V3, k-1, j, i);
                    //
                    // (02/20/23) 3rd prescription that is similar to 2nd prescription but not local and nonzero effective flux 
                    //if (ndim > 1) B_F.flux(X2DIR, V1, k, j, i) = -B_F.flux(X2DIR, V1, k, j, ie) + B_F.flux(X1DIR, V2, k, j, ie+1)
                    //                                               - B_F.flux(X1DIR, V2, k, j-2, ie+1) + B_F.flux(X2DIR, V1, k, j-1, ie) + B_F.flux(X2DIR, V1, k, j-1, ie+1);
                    if (ndim > 1) F2_host(V1, ktemp, jtemp, ie+1) = -F2_host(V1, ktemp, jtemp, ie) + F1_host(V2, ktemp, jtemp, ie+1)
                                                                   - F1_host(V2, ktemp, jtemp-2, ie+1) + F2_host(V1, ktemp, jtemp-1, ie) + F2_host(V1, ktemp, jtemp-1, ie+1);
                    //if (ndim > 2) B_F.flux(X3DIR, V1, k, j, i) = -B_F.flux(X3DIR, V1, k, j, ie) + B_F.flux(X1DIR, V3, k, j, ie+1)
                    //                                               - B_F.flux(X1DIR, V3, k-2, j, ie+1) + B_F.flux(X3DIR, V1, k-1, j, ie) + B_F.flux(X3DIR, V1, k-1, j, ie+1);
                    if (ndim > 2) F3_host(V1, ktemp, jtemp, ie+1) = -F3_host(V1, ktemp, jtemp, ie) + F1_host(V3, ktemp, jtemp, ie+1)
                                                                   - F1_host(V3, ktemp-2, jtemp, ie+1) + F3_host(V1, ktemp-1, jtemp, ie) + F3_host(V1, ktemp-1, jtemp, ie+1);

                    //if (in_x2 && (j==js)) {// (corners are tricky so let's just initialize)
                    if (in_x2 && (jtemp==js)) {// (corners are tricky so let's just initialize)
                        //B_F.flux(X2DIR, V1,k,j,i) = -B_F.flux(X1DIR,V2,k,j,ie) -B_F.flux(X1DIR,V2,k,j-1,ie);
                        F2_host(V1,ktemp,jtemp,ie+1) = -F1_host(V2,ktemp,jtemp,ie) -F1_host(V2,ktemp,jtemp-1,ie);
                        //B_F.flux(X2DIR, V1,k,j,i-1) = -0.5*B_F.flux(X2DIR,V1,k,j,i);
                        F2_host(V1,ktemp,jtemp,ie) = -0.5*F2_host(V1,ktemp,jtemp,ie+1);
                    }
                    //if (out_x2 && (j==je_e)) {// (corners are tricky)
                    if (out_x2 && (jtemp==je_e)) {// (corners are tricky)
                        //B_F.flux(X2DIR, V1, k, j, i-1) = -B_F.flux(X2DIR, V1, k, je, ie) - B_F.flux(X2DIR, V1, k, je, ie+1) 
                        //                                +B_F.flux(X1DIR, V2, k, je, ie+1) + B_F.flux(X1DIR, V2, k, je-1, ie+1);
                        //B_F.flux(X2DIR, V1, k, j, i) = -2.*B_F.flux(X1DIR, V2, k, je-1, ie+1) -B_F.flux(X1DIR, V2, k, je, ie+1) + B_F.flux(X1DIR, V2, k, je+1, ie+1)
                        //                                +2.*B_F.flux(X2DIR, V1, k, je, ie) + 2.*B_F.flux(X2DIR, V1, k, je, ie+1);
                        //B_F.flux(X1DIR,V2,k,j-1,i) = -B_F.flux(X1DIR,V2,k,je-1,i)+B_F.flux(X2DIR,V1,k,je,i)+B_F.flux(X2DIR,V1,k,je,i-1);
                        F1_host(V2,ktemp,jtemp-1,ie+1) = -F1_host(V2,ktemp,je-1,ie+1)+F2_host(V1,ktemp,je,ie+1)+F2_host(V1,ktemp,je,ie);
                        //B_F.flux(X1DIR,V2,k,j,i) = -B_F.flux(X1DIR,V2,k,je,i);
                        F1_host(V2,ktemp,jtemp,ie+1) = -F1_host(V2,ktemp,je,ie+1);
                    }
                //}
            //);
              }
            }
        }
        // Deep copy to device
        F1.DeepCopy(F1_host);
        F2.DeepCopy(F2_host);
        F3.DeepCopy(F3_host);
        Kokkos::fence();
        
        // put it back to B_F.flux. is this even needed?
        //pmb->par_for("copy_to_B_F_l", ks_all+2, ke_all, js_new, je_new, is, is,
        //     KOKKOS_LAMBDA_3D {
        //        VLOOP B_F.flux(X1DIR,v,k,j,i) = F1(v,k,j,i);
        //        VLOOP B_F.flux(X2DIR,v,k,j,i) = F2(v,k,j,i);
        //        VLOOP B_F.flux(X3DIR,v,k,j,i) = F3(v,k,j,i);
        //     }
        //);
        //pmb->par_for("copy_to_B_F_r", ks_all+2, ke_all, js_new, je_new, ie+1, ie+1,
        //     KOKKOS_LAMBDA_3D {
        //        VLOOP B_F.flux(X1DIR,v,k,j,i) = F1(v,k,j,i);
        //        VLOOP B_F.flux(X2DIR,v,k,j,i) = F2(v,k,j,i);
        //        VLOOP B_F.flux(X3DIR,v,k,j,i) = F3(v,k,j,i);
        //     }
        //);

        
    }

    Flag(md, "Fixed X1 B");
    return TaskStatus::complete;
}

// Outflow boundary conditions without the fix_eh_flux special sauce *always* generate divB.
// Don't report it, as we expect it.
// TODO we could stay off x2 if two_sync, but I wanna drive home that's weird for a cycle
IndexRange ValidDivBX1(MeshBlock *pmb)
{
    const IndexRange ibl = pmb->meshblock_data.Get()->GetBoundsI(IndexDomain::interior);
    bool avoid_inner = (!pmb->packages.Get("B_FluxCT")->Param<bool>("fix_eh_flux") &&
        pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user);
    bool avoid_outer = (!pmb->packages.Get("B_FluxCT")->Param<bool>("fix_exterior_flux") &&
        pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user);
    return IndexRange{ibl.s + (avoid_inner), ibl.e + (!avoid_outer)};
}

double MaxDivB(MeshData<Real> *md)
{
    Flag(md, "Calculating divB Mesh");
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    // Packing out here avoids frequent per-mesh packs.  Do we need to?
    auto B_U = md->PackVariables(std::vector<std::string>{"cons.B"});

    const IndexRange jbl = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kbl = md->GetBoundsK(IndexDomain::interior);

    const IndexRange jb = IndexRange{jbl.s, jbl.e + (ndim > 1)};
    const IndexRange kb = IndexRange{kbl.s, kbl.e + (ndim > 2)};
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    // TODO Keep zone of max!  Also applies to ctop.

    // This is one kernel call per block, because each block will have different bounds.
    // Could consolidate at the cost of lots of bounds checking.
    double max_divb = 0.0;
    for (int b = block.s; b <= block.e; ++b) {
        auto pmb = md->GetBlockData(b)->GetBlockPointer().get();

        const IndexRange ib = ValidDivBX1(pmb);

        double max_divb_block;
        Kokkos::Max<double> max_reducer(max_divb_block);
        pmb->par_reduce("divB_max", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result) {
                const auto& G = B_U.GetCoords(b);
                const double local_divb = m::abs(corner_div(G, B_U, b, k, j, i, ndim > 2));
                if (local_divb > local_result) local_result = local_divb;
            }
        , max_reducer);

        if (max_divb_block > max_divb) max_divb = max_divb_block;
    }

    Flag("Calculated");
    return max_divb;
}

double GlobalMaxDivB(MeshData<Real> *md)
{
    static AllReduce<Real> max_divb;
    max_divb.val = MaxDivB(md);
    max_divb.StartReduce(MPI_MAX);
    while (max_divb.CheckReduce() == TaskStatus::incomplete);
    return max_divb.val;
}

TaskStatus PrintGlobalMaxDivB(MeshData<Real> *md)
{
    Flag(md, "Printing B field diagnostics");
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Since this is in the history file now, I don't bother printing it
    // unless we're being verbose. It's not costly to calculate though
    if (pmb0->packages.Get("Globals")->Param<int>("verbose") >= 1) {
        Flag(md, "Printing divB");
        // Calculate the maximum from/on all nodes
        const double divb_max = B_FluxCT::GlobalMaxDivB(md);
        // Print on rank zero
        if(MPIRank0()) {
            std::cout << "Max DivB: " << divb_max << std::endl;
        }
    }

    Flag(md, "Printed B field diagnostics");
    return TaskStatus::complete;
}

// TODO unify these by adding FillOutputMesh option

void CalcDivB(MeshData<Real> *md, std::string divb_field_name)
{
    Flag(md, "Calculating divB for output");
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    // Packing out here avoids frequent per-mesh packs.  Do we need to?
    auto B_U = md->PackVariables(std::vector<std::string>{"cons.B"});
    auto divB = md->PackVariables(std::vector<std::string>{divb_field_name});

    const IndexRange jbl = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kbl = md->GetBoundsK(IndexDomain::interior);

    const IndexRange jb = IndexRange{jbl.s, jbl.e + (ndim > 1)};
    const IndexRange kb = IndexRange{kbl.s, kbl.e + (ndim > 2)};
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    // See MaxDivB for details
    for (int b = block.s; b <= block.e; ++b) {
        auto pmb = md->GetBlockData(b)->GetBlockPointer().get();

        const IndexRange ib = ValidDivBX1(pmb);

        pmb->par_for("calc_divB", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                const auto& G = B_U.GetCoords(b);
                divB(b, 0, k, j, i) = corner_div(G, B_U, b, k, j, i, ndim > 2);
            }
        );
    }

    Flag("Calculated");
}
void FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    auto rc = pmb->meshblock_data.Get().get();
    Flag(rc, "Calculating divB for output");
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return;

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto divB = rc->PackVariables(std::vector<std::string>{"divB"});

    const IndexRange jbl = rc->GetBoundsJ(IndexDomain::interior);
    const IndexRange kbl = rc->GetBoundsK(IndexDomain::interior);

    const IndexRange jb = IndexRange{jbl.s, jbl.e + (ndim > 1)};
    const IndexRange kb = IndexRange{kbl.s, kbl.e + (ndim > 2)};
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    const IndexRange ib = ValidDivBX1(pmb);

    pmb->par_for("divB_output", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords();
            divB(0, k, j, i) = corner_div(G, B_U, 0, k, j, i, ndim > 2);
        }
    );

    Flag(rc, "Output divB");
}

} // namespace B_FluxCT
