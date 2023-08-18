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
#include "b_ct.hpp"

#include "decs.hpp"
#include "domain.hpp"
#include "grmhd.hpp"
#include "kharma.hpp"
// TODO eliminate sync
#include "kharma_driver.hpp"

#include <parthenon/parthenon.hpp>

using namespace parthenon;

std::shared_ptr<KHARMAPackage> B_CT::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("B_CT");
    Params &params = pkg->AllParams();

    // Diagnostic & inadvisable flags

    // KHARMA requires some kind of field transport if there is a magnetic field allocated.
    // Use this flag if you actually want to disable all magnetic field flux corrections,
    // and allow a field divergence to grow unchecked, usually for debugging or comparison reasons
    bool disable_ct = pin->GetOrAddBoolean("b_field", "disable_ct", false);
    params.Add("disable_ct", disable_ct);

    // Default to stopping execution when divB is large, which generally indicates something
    // has gone wrong.  As always, can be disabled by the brave.
    bool kill_on_large_divb = pin->GetOrAddBoolean("b_field", "kill_on_large_divb", true);
    params.Add("kill_on_large_divb", kill_on_large_divb);
    Real kill_on_divb_over = pin->GetOrAddReal("b_field", "kill_on_divb_over", 1.e-3);
    params.Add("kill_on_divb_over", kill_on_divb_over);

    // Currently bs99, sg09
    // TODO LDZ04, LDZ07, other GS?
    std::string ct_scheme = pin->GetOrAddString("b_field", "ct_scheme", "sg09");
    params.Add("ct_scheme", ct_scheme);

    // Add a reducer for divB to params
    params.Add("divb_reducer", AllReduce<Real>());

    // FIELDS

    // TODO maybe one day implicit?

    // Flags for B fields on faces.
    // We don't mark these as "Primitive" and "Conserved" else they'd be bundled
    // with all the cell vars in a bunch of places we don't want
    std::vector<MetadataFlag> flags_prim_f = {Metadata::Real, Metadata::Face, Metadata::Derived,
                                            Metadata::GetUserFlag("Explicit")};
    std::vector<MetadataFlag> flags_cons_f = {Metadata::Real, Metadata::Face, Metadata::Independent,
                                              Metadata::GetUserFlag("Explicit"), Metadata::FillGhost}; // TODO TODO Restart
    auto m = Metadata(flags_prim_f);
    pkg->AddField("prims.fB", m);
    m = Metadata(flags_cons_f);
    pkg->AddField("cons.fB", m);

    // Cell-centered versions.  Needed for BS, not for other schemes.
    // Probably will want to keep primitives for e.g. correct PtoU of MHD vars, but cons maybe can go
    std::vector<MetadataFlag> flags_prim = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::GetUserFlag("Primitive"),
                                            Metadata::GetUserFlag("MHD"), Metadata::GetUserFlag("Explicit"), Metadata::Vector};
    std::vector<MetadataFlag> flags_cons = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::Conserved, Metadata::WithFluxes,
                                            Metadata::GetUserFlag("MHD"), Metadata::GetUserFlag("Explicit"), Metadata::Vector};
    std::vector<int> s_vector({NVEC});
    m = Metadata(flags_prim, s_vector);
    pkg->AddField("prims.B", m);
    m = Metadata(flags_cons, s_vector);
    pkg->AddField("cons.B", m);

    // EMF on edges.
    // TODO only sync when needed
    std::vector<MetadataFlag> flags_emf = {Metadata::Real, Metadata::Edge, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost};
    m = Metadata(flags_emf);
    pkg->AddField("B_CT.emf", m);

    if (ct_scheme == "sg09") {
        std::vector<MetadataFlag> flags_emf_c = {Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy};
        m = Metadata(flags_emf_c, s_vector);
        pkg->AddField("B_CT.cemf", m);
    }

    // CALLBACKS

    // We implement a source term replacement, rather than addition,
    // but same difference, really
    //pkg->AddSource = B_CT::AddSource;

    // Also ensure that prims get filled, both during step and on boundaries
    //pkg->MeshUtoP = B_CT::MeshUtoP;
    pkg->BlockUtoP = B_CT::BlockUtoP;
    pkg->BoundaryUtoP = B_CT::BlockUtoP;

    // Register the other callbacks
    pkg->PostStepDiagnosticsMesh = B_CT::PostStepDiagnostics;
    // TODO TODO prolongation/restriction will be registered here too

    // The definition of MaxDivB we care about actually changes per-transport,
    // so calculating it is handled by the transport package
    // We'd only ever need to declare or calculate divB for output (getting the max is independent)
    if (KHARMA::FieldIsOutput(pin, "divB")) {
        pkg->BlockUserWorkBeforeOutput = B_CT::FillOutput;
        m = Metadata({Metadata::Real, Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
        pkg->AddField("divB", m);
    }

    // List (vector) of HistoryOutputVars that will all be enrolled as output variables
    // LATER
    // parthenon::HstVar_list hst_vars = {};
    // hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::max, B_CT::MaxDivB, "MaxDivB"));
    // // Event horizon magnetization.  Might be the same or different for different representations?
    // if (pin->GetBoolean("coordinates", "spherical")) {
    //     hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, ReducePhi0, "Phi_0"));
    //     hst_vars.emplace_back(parthenon::HistoryOutputVar(UserHistoryOperation::sum, ReducePhi5, "Phi_EH"));
    // }
    // // add callbacks for HST output to the Params struct, identified by the `hist_param_key`
    // pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

    return pkg;
}

TaskStatus B_CT::MeshUtoP(MeshData<Real> *md, IndexDomain domain, bool coarse)
{
    // TODO later
    for (int i=0; i < md->NumBlocks(); i++)
        B_CT::BlockUtoP(md->GetBlockData(i).get(), domain, coarse);
    return TaskStatus::complete;
}

void B_CT::BlockUtoP(MeshBlockData<Real> *rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;
    auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
    auto B_Pf = rc->PackVariables(std::vector<std::string>{"prims.fB"});
    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.B"});
    auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});
    const auto& G = pmb->coords;

    // Update the primitive B-fields on faces
    const IndexRange3 bf = KDomain::GetRange(rc, domain, 0, 1, coarse);
    pmb->par_for("UtoP_B", bf.ks, bf.ke, bf.js, bf.je, bf.is, bf.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            // TODO will we need face area here?
            B_Pf(F1, 0, k, j, i) = B_Uf(F1, 0, k, j, i) / G.gdet(Loci::face1, j, i);
            B_Pf(F2, 0, k, j, i) = B_Uf(F2, 0, k, j, i) / G.gdet(Loci::face2, j, i);
            B_Pf(F3, 0, k, j, i) = B_Uf(F3, 0, k, j, i) / G.gdet(Loci::face3, j, i);
        }
    );
    Kokkos::fence();
    // Average the primitive vals for zone centers (TODO right?)
    const IndexRange3 bc = KDomain::GetRange(rc, domain, coarse);
    pmb->par_for("UtoP_B_center", bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            B_P(V1, k, j, i) = (B_Pf(F1, 0, k, j, i) +  B_Pf(F1, 0, k, j, i + 1)) / 2;
            B_P(V2, k, j, i) = (ndim > 1) ? (B_Pf(F2, 0, k, j, i) +  B_Pf(F2, 0, k, j + 1, i)) / 2
                                          : B_Pf(F2, 0, k, j, i);
            B_P(V3, k, j, i) = (ndim > 2) ? (B_Pf(F3, 0, k, j, i) +  B_Pf(F3, 0, k + 1, j, i)) / 2
                                          : B_Pf(F3, 0, k, j, i);
        }
    );
    Kokkos::fence();
    pmb->par_for("UtoP_B_centerPtoU", 0, NVEC-1, bc.ks, bc.ke, bc.js, bc.je, bc.is, bc.ie,
        KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
            B_U(v, k, j, i) = B_P(v, k, j, i) * G.gdet(Loci::center, j, i);
        }
    );
    Kokkos::fence();
}

// TODO this isn't really a source... it's a replacement of the
// face-centered fields according to constrained transport rules
TaskStatus B_CT::UpdateFaces(std::shared_ptr<MeshData<Real>>& md, std::shared_ptr<MeshData<Real>>& mdudt)
{
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    // EMF temporary
    auto& emf_pack = md->PackVariables(std::vector<std::string>{"B_CT.emf"});

    // Figure out indices
    const IndexRange3 b = KDomain::GetRange(md, IndexDomain::interior, 0, 0);
    const IndexRange3 b1 = KDomain::GetRange(md, IndexDomain::interior, 0, 1);
    const IndexRange block = IndexRange{0, emf_pack.GetDim(5)-1};

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer().get();

    std::string scheme = pmesh->packages.Get("B_CT")->Param<std::string>("ct_scheme");
    if (scheme == "bs99") {
        // Calculate circulation by averaging fluxes (BS88)
        auto& B_U = md->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});
        pmb0->par_for("B_CT_emf_BS", block.s, block.e, b1.ks, b1.ke, b1.js, b1.je, b1.is, b1.ie,
            KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
                // TODO will we need gdet/cell length here?
                const auto& G = B_U.GetCoords(bl);
                if (ndim > 2) {
                    emf_pack(bl, E1, 0, k, j, i) =
                        0.25*(B_U(bl).flux(X2DIR, V3, k - 1, j, i)/G.Dxc<3>(k-1) + B_U(bl).flux(X2DIR, V3, k, j, i)/G.Dxc<3>(k)
                            - B_U(bl).flux(X3DIR, V2, k, j - 1, i)/G.Dxc<2>(j-1) - B_U(bl).flux(X3DIR, V2, k, j, i)/G.Dxc<2>(j));
                    emf_pack(bl, E2, 0, k, j, i) =
                        0.25*(B_U(bl).flux(X3DIR, V1, k, j, i - 1)/G.Dxc<1>(i-1) + B_U(bl).flux(X3DIR, V1, k, j, i)/G.Dxc<1>(i)
                            - B_U(bl).flux(X1DIR, V3, k - 1, j, i)/G.Dxc<3>(k-1) - B_U(bl).flux(X1DIR, V3, k, j, i)/G.Dxc<3>(k));
                }
                emf_pack(bl, E3, 0, k, j, i) =
                    0.25*(B_U(bl).flux(X1DIR, V2, k, j - 1, i)/G.Dxc<2>(j-1) + B_U(bl).flux(X1DIR, V2, k, j, i)/G.Dxc<2>(j)
                        - B_U(bl).flux(X2DIR, V1, k, j, i - 1)/G.Dxc<1>(i-1) - B_U(bl).flux(X2DIR, V1, k, j, i)/G.Dxc<1>(i));
            }
        );
    } else if (scheme == "sg09") {
        // Average fluxes and derivatives (SG09)
        auto& uvec = md->PackVariables(std::vector<std::string>{"prims.uvec"});
        auto& emfc = md->PackVariables(std::vector<std::string>{"B_CT.cemf"});
        auto& B_U = md->PackVariablesAndFluxes(std::vector<std::string>{"cons.B"});
        auto& B_P = md->PackVariables(std::vector<std::string>{"prims.B"});
        // emf in center == -v x B
        pmb0->par_for("B_CT_emf_GS09", block.s, block.e, b1.ks, b1.ke, b1.js, b1.je, b1.is, b1.ie,
            KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
                VLOOP emfc(bl, v, k, j, i) = 0.;
                VLOOP3 emfc(bl, x, k, j, i) -= antisym(v, w, x) * uvec(bl, v, k, j, i) * B_U(bl, w, k, j, i);
            }
        );

        // Get primitive velocity at face (on right side) (TODO do we need some average?)
        auto& uvecf = md->PackVariables(std::vector<std::string>{"Flux.vr"});

        pmb0->par_for("B_CT_emf_GS09", block.s, block.e, b1.ks, b1.ke, b1.js, b1.je, b1.is, b1.ie,
            KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
                // TODO will we need gdet/cell length here?
                const auto& G = B_U.GetCoords(bl);

                // "simple" flux + upwinding method, Stone & Gardiner '09 but also in Stone+08 etc.
                // Upwinded differences take in order (1-indexed):
                // 1. EMF component direction to calculate
                // 2. Direction of derivative
                // 3. Direction of upwinding
                // ...then zone number...
                // and finally, a boolean indicating a leftward (e.g., i-3/4) vs rightward (i-1/4) position
                if (ndim > 2) {
                    emf_pack(bl, E1, 0, k, j, i) =
                        0.25*(B_U(bl).flux(X2DIR, V3, k - 1, j, i)/G.Dxc<3>(k-1) + B_U(bl).flux(X2DIR, V3, k, j, i)/G.Dxc<3>(k)
                            - B_U(bl).flux(X3DIR, V2, k, j - 1, i)/G.Dxc<2>(j-1) - B_U(bl).flux(X3DIR, V2, k, j, i)/G.Dxc<2>(j))
                        + (1./4)*(upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 1, 3, 2, k, j, i, false)
                                - upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 1, 3, 2, k, j, i, true))
                        + (1./4)*(upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 1, 2, 3, k, j, i, false)
                                - upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 1, 2, 3, k, j, i, true));
                    emf_pack(bl, E2, 0, k, j, i) =
                        0.25*(B_U(bl).flux(X3DIR, V1, k, j, i - 1)/G.Dxc<1>(i-1) + B_U(bl).flux(X3DIR, V1, k, j, i)/G.Dxc<1>(i)
                            - B_U(bl).flux(X1DIR, V3, k - 1, j, i)/G.Dxc<3>(k-1) - B_U(bl).flux(X1DIR, V3, k, j, i)/G.Dxc<3>(k))
                        + (1./4)*(upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 2, 1, 3, k, j, i, false)
                                - upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 2, 1, 3, k, j, i, true))
                        + (1./4)*(upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 2, 3, 1, k, j, i, false)
                                - upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 2, 3, 1, k, j, i, true));
                }
                emf_pack(bl, E3, 0, k, j, i) =
                    0.25*(B_U(bl).flux(X1DIR, V2, k, j - 1, i)/G.Dxc<2>(j-1) + B_U(bl).flux(X1DIR, V2, k, j, i)/G.Dxc<2>(j)
                        - B_U(bl).flux(X2DIR, V1, k, j, i - 1)/G.Dxc<1>(i-1) - B_U(bl).flux(X2DIR, V1, k, j, i)/G.Dxc<1>(i))
                    + (1./4)*(upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 3, 2, 1, k, j, i, false)
                            - upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 3, 2, 1, k, j, i, true))
                    + (1./4)*(upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 3, 1, 2, k, j, i, false)
                            - upwind_diff(B_U(bl), emfc(bl), uvecf(bl), 3, 1, 2, k, j, i, true));
            }
        );
    } else {
        throw std::invalid_argument("Invalid CT scheme specified!  Must be one of bs99, sg09");
    }

    // This is what we're replacing
    auto& dB_Uf_dt = mdudt->PackVariables(std::vector<std::string>{"cons.fB"});
    // Circulation -> change in flux at face
    // Note we *replace* whatever this term in the source term was "supposed" to be
    pmb0->par_for("B_CT_Circ_1", block.s, block.e, b.ks, b.ke, b.js, b.je, b1.is, b1.ie,
        KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
            const auto& G = dB_Uf_dt.GetCoords(bl);
            dB_Uf_dt(bl, F1, 0, k, j, i) =  emf_pack(bl, E3, 0, k, j + 1, i) - emf_pack(bl, E3, 0, k, j, i);
            if (ndim > 2) {
                dB_Uf_dt(bl, F1, 0, k, j, i) += -emf_pack(bl, E2, 0, k + 1, j, i) + emf_pack(bl, E2, 0, k, j, i);
            }
        }
    );
    pmb0->par_for("B_CT_Circ_2", block.s, block.e, b.ks, b.ke, b1.js, b1.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
            const auto& G = dB_Uf_dt.GetCoords(bl);
            dB_Uf_dt(bl, F2, 0, k, j, i) = -emf_pack(bl, E3, 0, k, j, i + 1) + emf_pack(bl, E3, 0, k, j, i);
            if (ndim > 2) {
                dB_Uf_dt(bl, F2, 0, k, j, i) +=  emf_pack(bl, E1, 0, k + 1, j, i) - emf_pack(bl, E1, 0, k, j, i);
            }
        }
    );
    if (ndim > 2) {
        pmb0->par_for("B_CT_Circ_3", block.s, block.e, b1.ks, b1.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA (const int &bl, const int &k, const int &j, const int &i) {
                const auto& G = dB_Uf_dt.GetCoords(bl);
                dB_Uf_dt(bl, F3, 0, k, j, i) +=  emf_pack(bl, E2, 0, k, j, i + 1) - emf_pack(bl, E2, 0, k, j, i)
                                            - emf_pack(bl, E1, 0, k, j + 1, i) + emf_pack(bl, E1, 0, k, j, i);
            }
        );
    }
    return TaskStatus::complete;
}



double B_CT::MaxDivB(MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    auto B_U = md->PackVariables(std::vector<std::string>{"cons.fB"});

    // Figure out indices
    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer().get();

    double max_divb;
    Kokkos::Max<double> max_reducer(max_divb);
    pmb0->par_reduce("divB_max", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i, double &local_result) {
            const auto& G = B_U.GetCoords(b);
            double local_divb = face_div(G, B_U(b), ndim, k, j, i);
            if (local_divb > local_result) local_result = local_divb;
        }
    , max_reducer);

    return max_divb;
}
double B_CT::BlockMaxDivB(MeshBlockData<Real> *rc)
{
    const int ndim = KDomain::GetNDim(rc);

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.fB"});

    // Figure out indices
    const IndexRange3 b = KDomain::GetRange(rc, IndexDomain::interior);

    auto pmb = rc->GetBlockPointer();

    double max_divb;
    Kokkos::Max<double> max_reducer(max_divb);
    pmb->par_reduce("divB_max", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i, double &local_result) {
            const auto& G = B_U.GetCoords();
            double local_divb = face_div(G, B_U, ndim, k, j, i);
            if (local_divb > local_result) local_result = local_divb;
        }
    , max_reducer);

    return max_divb;
}

double B_CT::GlobalMaxDivB(MeshData<Real> *md, bool all_reduce)
{
    if (all_reduce) {
        Reductions::StartToAll<Real>(md, 2, MaxDivB(md), MPI_MAX);
        return Reductions::CheckOnAll<Real>(md, 2);
    } else {
        Reductions::Start<Real>(md, 2, MaxDivB(md), MPI_MAX);
        return Reductions::Check<Real>(md, 2);
    }
}

TaskStatus B_CT::PrintGlobalMaxDivB(MeshData<Real> *md, bool kill_on_large_divb)
{
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    // Since this is in the history file now, I don't bother printing it
    // unless we're being verbose. It's not costly to calculate though
    if (pmb0->packages.Get("Globals")->Param<int>("verbose") >= 1) {
        // Calculate the maximum from/on all nodes
        const double divb_max = B_CT::GlobalMaxDivB(md);
        // Print on rank zero
        if (MPIRank0()) {
            std::cout << "Max DivB: " << divb_max << std::endl;
        }
        if (kill_on_large_divb) {
            if (divb_max > pmb0->packages.Get("B_CT")->Param<Real>("kill_on_divb_over"))
                throw std::runtime_error("DivB exceeds maximum! Quitting...");
        }
    }

    return TaskStatus::complete;
}

// TODO unify these by adding FillOutputMesh option

void B_CT::CalcDivB(MeshData<Real> *md, std::string divb_field_name)
{
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;

    // Packing out here avoids frequent per-mesh packs.  Do we need to?
    auto B_U = md->PackVariables(std::vector<std::string>{"cons.fB"});
    auto divB = md->PackVariables(std::vector<std::string>{divb_field_name});

    const IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    auto pmb0 = md->GetBlockData(0)->GetBlockPointer().get();

    // See MaxDivB for details
    pmb0->par_for("calc_divB", block.s, block.e, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &b, const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords(b);
            divB(b, 0, k, j, i) = face_div(G, B_U(b), ndim, k, j, i);
        }
    );
}

void B_CT::FillOutput(MeshBlock *pmb, ParameterInput *pin)
{
    auto rc = pmb->meshblock_data.Get().get();
    const int ndim = pmb->pmy_mesh->ndim;
    if (ndim < 2) return;

    auto B_U = rc->PackVariables(std::vector<std::string>{"cons.fB"});
    auto divB = rc->PackVariables(std::vector<std::string>{"divB"});

    const IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
    const IndexRange block = IndexRange{0, B_U.GetDim(5)-1};

    pmb->par_for("divB_output", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            const auto& G = B_U.GetCoords();
            divB(0, k, j, i) = face_div(G, B_U, ndim, k, j, i);
        }
    );
}
