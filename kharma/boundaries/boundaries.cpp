/*
 *  File: boundaries.cpp
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
#include "boundaries.hpp"

#include "decs.hpp"
#include "domain.hpp"
#include "kharma.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "types.hpp"

#include "b_ct.hpp"
#include "b_flux_ct.hpp"
#include "flux.hpp"

// Parthenon's boundaries
#include <bvals/boundary_conditions.hpp>

std::shared_ptr<KHARMAPackage> KBoundaries::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t> &packages)
{
    auto pkg = std::make_shared<KHARMAPackage>("Boundaries");
    Params &params = pkg->AllParams();

    // OPTIONS FOR SPECIFIC BOUNDARIES
    bool spherical = pin->GetBoolean("coordinates", "spherical");
    // Global check inflow sets inner/outer X1 by default
    bool check_inflow_global = pin->GetOrAddBoolean("boundaries", "check_inflow", spherical);
    // TODO TODO Support old option names check_inflow_inner, check_inflow_outer

    // Ensure fluxes through the zero-size face at the pole are zero
    bool zero_polar_flux = pin->GetOrAddBoolean("boundaries", "zero_polar_flux", spherical);
    params.Add("zero_polar_flux", zero_polar_flux);

    // Apply physical boundaries to conserved GRMHD variables rho u^r, T^mu_nu
    // Probably inadvisable?
    bool domain_bounds_on_conserved = pin->GetOrAddBoolean("boundaries", "domain_bounds_on_conserved", false);
    params.Add("domain_bounds_on_conserved", domain_bounds_on_conserved);

    // Fix the X1/X2 corner by replacing the reflecting condition with the inflow
    // Never use this if not in spherical coordinates
    // Activates by default only with reflecting X2/outflow X1 and interior boundary inside EH
    // TODO(BSP) may also be specific to Funky MKS coords with zero_point==startx1
    bool fix_corner = false;
    if (spherical) {
        bool correct_bounds =
            (pin->GetString("boundaries", "inner_x2") == "reflecting" &&
             pin->GetString("boundaries", "outer_x2") == "reflecting" &&
             pin->GetString("boundaries", "inner_x1") == "outflow");
        bool inside_eh = pin->GetBoolean("coordinates", "domain_intersects_eh");
        fix_corner = pin->GetOrAddBoolean("boundaries", "fix_corner", correct_bounds && inside_eh);
        // Allow overriding with specific name
        fix_corner = pin->GetOrAddBoolean("boundaries", "fix_corner_inner", fix_corner);
    }
    params.Add("fix_corner_inner", fix_corner);
    params.Add("fix_corner_outer", pin->GetOrAddBoolean("boundaries", "fix_corner_outer", false));

    // We can't use GetVariablesByFlag yet, so ask the packages
    // These flags get anything that needs a physical boundary during the run
    using FC = Metadata::FlagCollection;
    FC ghost_vars = FC({Metadata::FillGhost, Metadata::Conserved})
                + FC({Metadata::FillGhost, Metadata::GetUserFlag("Primitive")})
                - FC({Metadata::GetUserFlag("StartupOnly")});
    int nvar = KHARMA::PackDimension(packages.get(), ghost_vars);
    // Face-centered fields: some duplicate stuff, leaving it separate for now
    FC ghost_vars_f = FC({Metadata::FillGhost, Metadata::Face})
                - FC({Metadata::GetUserFlag("StartupOnly")});
    int nvar_f = m::max(KHARMA::PackDimension(packages.get(), ghost_vars_f), 1);

    // TODO encapsulate this
    Metadata m_x1, m_x2, m_x3, m_x1_f, m_x2_f, m_x3_f;
    {
        // We also don't know the mesh size, since it's not constructed.  We infer.
        const int ng = pin->GetInteger("parthenon/mesh", "nghost");
        const int nx1 = pin->GetInteger("parthenon/meshblock", "nx1");
        const int n1 = nx1 + 2 * ng;
        const int nx2 = pin->GetInteger("parthenon/meshblock", "nx2");
        const int n2 = (nx2 == 1) ? nx2 : nx2 + 2 * ng;
        const int nx3 = pin->GetInteger("parthenon/meshblock", "nx3");
        const int n3 = (nx3 == 1) ? nx3 : nx3 + 2 * ng;

        // These are declared *backward* from how they will be indexed
        std::vector<int> s_x1({ng, n2, n3, nvar});
        std::vector<int> s_x2({n1, ng, n3, nvar});
        std::vector<int> s_x3({n1, n2, ng, nvar});
        // Dirichlet conditions must be restored when restarting!
        m_x1 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x1);
        m_x2 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x2);
        m_x3 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x3);

        if (nvar_f > 0) {
            // Face mesh sizes
            const int ng_f = pin->GetInteger("parthenon/mesh", "nghost") + 1;
            const int nx1 = pin->GetInteger("parthenon/meshblock", "nx1");
            const int n1_f = (nx2 == 1) ? nx1 : nx1 + 2 * ng + 1;
            const int nx2 = pin->GetInteger("parthenon/meshblock", "nx2");
            const int n2_f = (nx2 == 1) ? nx2 : nx2 + 2 * ng + 1;
            const int nx3 = pin->GetInteger("parthenon/meshblock", "nx3");
            const int n3_f = (nx3 == 1) ? nx3 : nx3 + 2 * ng + 1;

            std::cerr << "Face fluid vars: " << nvar_f << std::endl;

            // These are declared *backward* from how they will be indexed
            std::vector<int> s_x1_f({ng_f, n2_f, n3_f, nvar_f});
            std::vector<int> s_x2_f({n1_f, ng_f, n3_f, nvar_f});
            std::vector<int> s_x3_f({n1_f, n2_f, ng_f, nvar_f});
            // Note these are *NOT* face variables, they cannot be indexed by TopologicalElement.
            // Instead, we make them a normal vector and use int(TE) % 3
            m_x1_f = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x1_f);
            m_x2_f = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x2_f);
            m_x3_f = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy, Metadata::Restart}, s_x3_f);
        }
    }

    // Set options for each boundary
    for (int i = 0; i < BOUNDARY_NFACES; i++) {
        const auto bface = (BoundaryFace) i;
        const auto bdomain = BoundaryDomain(bface);
        const auto bname = BoundaryName(bface);
        const auto bdir = BoundaryDirection(bface);
        const auto binner = BoundaryIsInner(bface);

        // OPTIONS FOR ANY BOUNDARY

        // Prevent inflow at boundaries.
        // This is two separate checks, but default to enabling/disabling together for X1 and not elsewhere
        bool check_inflow = pin->GetOrAddBoolean("boundaries", "check_inflow_" + bname, check_inflow_global && bdir == X1DIR);
        params.Add("check_inflow_" + bname, check_inflow);

        // Ensure fluxes through the zero-size face at the pole are zero
        bool zero_flux = pin->GetOrAddBoolean("boundaries", "zero_flux_" + bname, zero_polar_flux && bdir == X2DIR);
        params.Add("zero_flux_" + bname, zero_flux);

        // Allow specifically dP to outflow in otherwise Dirichlet conditions
        // Only used for viscous_bondi problem
        bool outflow_EMHD = pin->GetOrAddBoolean("boundaries", "outflow_EMHD_" + bname, false);
        params.Add("outflow_EMHD_" + bname, outflow_EMHD);

        // Invert X2 face values to reflect across polar boundary
        bool invert_F2 = pin->GetOrAddBoolean("boundaries", "reflect_face_vector_" + bname, (bdir == X2DIR && spherical));
        params.Add("reflect_face_vector_"+bname, invert_F2);

        // BOUNDARY TYPES
        // Get the boundary type we specified in kharma
        auto btype = pin->GetString("boundaries", bname);
        params.Add(bname, btype);

        // Zero EMFs to prevent B field escaping the domain in polar/dirichlet bounds
        bool zero_EMF = pin->GetOrAddBoolean("boundaries", "zero_EMF_" + bname, (bdir == X2DIR && spherical)
                                                                             || (btype == "dirichlet"));
        params.Add("zero_EMF_"+bname, zero_EMF);

        // String manip to get the Parthenon boundary name, e.g., "ox1_bc"
        auto bname_parthenon = bname.substr(0, 1) + "x" + bname.substr(7, 8) + "_bc";
        // Parthenon implements periodic conditions
        // For the rest, they should call our default wrapper, which we register in main()
        if (btype == "periodic") {
            pin->SetString("parthenon/mesh", bname_parthenon, "periodic");
        } else {
            pin->SetString("parthenon/mesh", bname_parthenon, "user");

            // Register the actual boundaries with the package, which our wrapper will use
            // when called via Parthenon's "user" conditions
            if (btype == "dirichlet") {
                // Dirichlet boundaries: allocate
                pkg->AddField("Boundaries." + bname, (bdir == X1DIR) ? m_x1 : ((bdir == X2DIR) ? m_x2 : m_x3));
                if (nvar_f > 0) {
                    pkg->AddField("Boundaries.f." + bname, (bdir == X1DIR) ? m_x1_f : ((bdir == X2DIR) ? m_x2_f : m_x3_f));
                }
                switch (bface) {
                case BoundaryFace::inner_x1:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::inner_x1>;
                    break;
                case BoundaryFace::outer_x1:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::outer_x1>;
                    break;
                case BoundaryFace::inner_x2:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::inner_x2>;
                    break;
                case BoundaryFace::outer_x2:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::outer_x2>;
                    break;
                case BoundaryFace::inner_x3:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::inner_x3>;
                    break;
                case BoundaryFace::outer_x3:
                    pkg->KBoundaries[bface] = KBoundaries::Dirichlet<BoundaryFace::outer_x3>;
                    break;
                default:
                    break;
                }
            } else if (btype == "reflecting") {
                switch (bface) {
                case BoundaryFace::inner_x1:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectInnerX1;
                    break;
                case BoundaryFace::outer_x1:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectOuterX1;
                    break;
                case BoundaryFace::inner_x2:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectInnerX2;
                    break;
                case BoundaryFace::outer_x2:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectOuterX2;
                    break;
                case BoundaryFace::inner_x3:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectInnerX3;
                    break;
                case BoundaryFace::outer_x3:
                    pkg->KBoundaries[bface] = BoundaryFunction::ReflectOuterX3;
                    break;
                default:
                    break;
                }
            } else if (btype == "outflow") {
                switch (bface) {
                case BoundaryFace::inner_x1:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowInnerX1;
                    break;
                case BoundaryFace::outer_x1:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowOuterX1;
                    break;
                case BoundaryFace::inner_x2:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowInnerX2;
                    break;
                case BoundaryFace::outer_x2:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowOuterX2;
                    break;
                case BoundaryFace::inner_x3:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowInnerX3;
                    break;
                case BoundaryFace::outer_x3:
                    pkg->KBoundaries[bface] = BoundaryFunction::OutflowOuterX3;
                    break;
                default:
                    break;
                }
            } else {
                throw std::runtime_error("Unknown boundary type: "+btype);
            }
        }
    }

    // Callbacks
    // Fix flux
    pkg->FixFlux = KBoundaries::FixFlux;
    return pkg;
}

void KBoundaries::ApplyBoundary(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse)
{
    Flag("ApplyBoundary"); // this is not a callback, flag for ourselves
    // KHARMA has to do some extra tasks in addition to just applying the usual
    // boundary conditions.  Therefore, we "wrap" Parthenon's (or our own)
    // boundary functions with this one.

    auto pmb = rc->GetBlockPointer();
    auto pkg = pmb->packages.Get<KHARMAPackage>("Boundaries");
    auto& params = pkg->AllParams();

    // TODO canonize this as a function. Prints all variables in the current MBD/MD object,
    // which can now be smaller than everything.
    // std::cout << rc->GetVariableVector().size() << std::endl;
    // for (auto &var : rc->GetVariableVector()) {
    //     std::cout << var->label() << " ";
    // }
    // std::cout << std::endl;

    const auto bface = BoundaryFaceOf(domain);
    const auto bname = BoundaryName(bface);
    const auto btype_name = params.Get<std::string>(bname);
    const auto bdir = BoundaryDirection(bface);
    const bool binner = BoundaryIsInner(bface);

    Flag("Apply "+bname+" boundary: "+btype_name);
    pkg->KBoundaries[bface](rc, coarse);
    EndFlag();

    // Nothing below is designed, nor necessary, for coarse buffers
    if (coarse) {
        EndFlag();
        return;
    }

    // If we're syncing EMFs and in spherical, explicitly zero polar faces
    // Since we manipulate the j coord, we'd overstep coarse bufs
    auto& emfpack = rc->PackVariables(std::vector<std::string>{"B_CT.emf"});
    if (params.Get<bool>("zero_EMF_" + bname) && emfpack.GetDim(4) > 0) {
        Flag("BoundaryEdge_"+bname);
        std::vector<TE> te_list;
        if (bdir == 1) {
            te_list = {TE::E2, TE::E3};
        } else if (bdir == 2) {
            te_list = {TE::E1, TE::E3};
        } else {
            te_list = {TE::E1, TE::E2};
        }

        for (TE el : te_list) {
            // Augment the domain -- EMF must be zero *on* domain faces, not just beyond
            auto b = KDomain::GetRange(rc, domain, el, (binner) ? 0 : -1, (binner) ? 1 : 0, coarse);
            pmb->par_for(
                "zero_EMF_" + bname, b.ks, b.ke, b.js, b.je, b.is, b.ie,
                KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                    emfpack(el, 0, k, j, i) = 0;
                }
            );
        }
        EndFlag();
    }

    // Zero/invert XN faces at a reflecting XN boundary (nearly always X2)
    auto fpack = rc->PackVariables({Metadata::Face, Metadata::FillGhost});
    if (params.Get<bool>("reflect_face_vector_" + bname) && fpack.GetDim(4) > 0) {
        Flag("BoundaryFace_"+bname);
        TE el = (bdir == 1) ? F1 : (bdir == 2) ? F2 : F3;
        // This is the domain of the boundary/ghost zones
        // Augment the domain since we're always modifying e.g. F2 in X2 boundary
        auto b = KDomain::GetRange(rc, domain, el, (binner) ? 0 : -1, (binner) ? 1 : 0, coarse);
        // Zero the last physical face, otherwise invert.
        auto i_f = (binner) ? b.ie : b.is;
        auto j_f = (binner) ? b.je : b.js;
        auto k_f = (binner) ? b.ke : b.ks;
        // Values are *reflected* by Parthenon, but vector must be *inverted* by us
        // (since Parthenon doesn't know B2 on the F2 face is a vector X2 component & thus switches sign)
        pmb->par_for(
            "reflect_face_vector_" + bname, b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                fpack(el, 0, k, j, i) = ((bdir == 1 && i == i_f) ||
                                         (bdir == 2 && j == j_f) ||
                                         (bdir == 3 && j == k_f)) ? 0. : -fpack(el, 0, k, j, i);
            }
        );
        EndFlag();
    }

    // This will now be called in 2 places we might not expect,
    // where we still may want to control the physical bounds:
    // 1. Syncing only the EMF during runs with CT
    // 2. Syncing boundaries while solving for B field
    // this generally guards against anytime we can't do the below
    PackIndexMap prims_map;
    if (GRMHD::PackMHDPrims(rc.get(), prims_map).GetDim(4) == 0) {
        EndFlag();
        return;
    }

    // Prevent inflow of material by changing fluid speeds,
    // anywhere we've specified.
    if (params.Get<bool>("check_inflow_" + bname)) {
        Flag("CheckInflow_"+bname);
        CheckInflow(rc, domain, coarse);
        EndFlag();
    }

    // Allow specifically dP to outflow in otherwise Dirichlet conditions
    // Only used for viscous_bondi problem
    // TODO make this more general?
    if (params.Get<bool>("outflow_EMHD_" + bname)) {
        Flag("OutflowEMHD_"+bname);
        auto EMHDg = rc->PackVariables({Metadata::GetUserFlag("EMHDVar"), Metadata::FillGhost});
        const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
        const auto &range = (bdir == 1) ? bounds.GetBoundsI(IndexDomain::interior)
                                : (bdir == 2 ? bounds.GetBoundsJ(IndexDomain::interior)
                                    : bounds.GetBoundsK(IndexDomain::interior));
        const int ref = binner ? range.s : range.e;
        pmb->par_for_bndry(
            "outflow_EMHD", IndexRange{0,EMHDg.GetDim(4)-1}, domain, CC, coarse,
            KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
                EMHDg(v, k, j, i) = EMHDg(v, (bdir == 3) ? ref : k, (bdir == 2) ? ref : j, (bdir == 1) ? ref : i);
            }
        );
        EndFlag();
    }

    /*
    * KHARMA is very particular about corner boundaries.
    * In particular, we apply the outflow boundary over ALL X2 & X3.
    * Then we apply the polar bound only where outflow is not applied,
    * and periodic bounds only where neither other bound applies.
    * The latter is accomplished regardless of Parthenon's definitions,
    * since these functions are run after Parthenon's MPI boundary syncs &
    * replace whatever they've done.
    * However, the former must be added after the X2 boundary call,
    * replacing the reflecting conditions in the X1/X2 corner (or in 3D, edge)
    * with outflow conditions based on the updated ghost cells.
    */
    if (bdir == X2DIR) {
        // If we're on the interior edge, re-apply that edge for our block by calling
        // exactly the same function that Parthenon does.  This ensures we're applying
        // the same thing, just emulating calling it after X2.
        if (params.Get<bool>("fix_corner_inner")) {
            if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
                Flag("FixCorner");
                ApplyBoundary(rc, IndexDomain::inner_x1, coarse);
                EndFlag();
            }
        }
        if (params.Get<bool>("fix_corner_outer")) {
            if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
                Flag("FixCorner");
                ApplyBoundary(rc, IndexDomain::outer_x1, coarse);
                EndFlag();
            }
        }
    }

    bool sync_prims = rc->GetBlockPointer()->packages.Get("Driver")->Param<bool>("sync_prims");
    // There are two modes of operation here:
    if (sync_prims) {
        // 1. Exchange/prolongate/restrict PRIMITIVE variables: (ImEx driver)
        //    Primitive variables and conserved B field are marked FillGhost
        //    Explicitly run UtoP on B field, then PtoU on everything
        // TODO there should be a set of B field wrappers that dispatch this
        auto pkgs = pmb->packages.AllPackages();
        if (pkgs.count("B_FluxCT")) {
            B_FluxCT::BlockUtoP(rc.get(), domain, coarse);
        } else if (pkgs.count("B_CT")) {
            B_CT::BlockUtoP(rc.get(), domain, coarse);
        }
        Flux::BlockPtoU(rc.get(), domain, coarse);
    } else {
        // 2. Exchange/prolongate/restrict CONSERVED variables: (KHARMA driver, maybe ImEx+AMR)
        //    Conserved variables are marked FillGhost, plus FLUID PRIMITIVES.
        if (!params.Get<bool>("domain_bounds_on_conserved")) {
            // To apply primitive boundaries to GRMHD, we run PtoU on that ONLY,
            // and UtoP on EVERYTHING ELSE
            Packages::BoundaryPtoUElseUtoP(rc.get(), domain, coarse);
        } else {
            // If we want to apply boundaries to conserved vars, just run UtoP on EVERYTHING
            Packages::BoundaryUtoP(rc.get(), domain, coarse);
        }
    }

    EndFlag();
}

void KBoundaries::CheckInflow(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse)
{
    auto pmb = rc->GetBlockPointer();
    const auto &G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    PackIndexMap prims_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map, coarse);
    const VarMap m_p(prims_map, false);

    // Inflow check
    // Iterate over all boundary domain zones w/p=0
    pmb->par_for_bndry(
        "check_inflow", IndexRange{0, 0}, domain, CC, coarse,
        KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
            KBoundaries::check_inflow(G, P, domain, m_p.U1, k, j, i);
        }
    );
}

TaskStatus KBoundaries::FixFlux(MeshData<Real> *md)
{
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto& params = pmb0->packages.Get("Boundaries")->AllParams();

    // Fluxes are defined at faces, so there is one more valid flux than
    // valid cell in the face direction.  That is, e.g. F1 is valid on
    // an (N1+1)xN2xN3 grid, F2 on N1x(N2+1)xN3, etc.
    // These functions do *not* need an extra row outside the domain,
    // like B_FluxCT::FixBoundaryFlux does.
    const int ndim = pmesh->ndim;
    // Entire range
    const IndexRange ibe = pmb0->cellbounds.GetBoundsI(IndexDomain::entire);
    const IndexRange jbe = pmb0->cellbounds.GetBoundsJ(IndexDomain::entire);
    const IndexRange kbe = pmb0->cellbounds.GetBoundsK(IndexDomain::entire);
    // Ranges for sides
    const IndexRange ibs = pmb0->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jbs = pmb0->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kbs = pmb0->cellbounds.GetBoundsK(IndexDomain::interior);
    // Ranges for faces
    const IndexRange ibf = IndexRange{ibs.s, ibs.e + 1};
    const IndexRange jbf = IndexRange{jbs.s, jbs.e + (ndim > 1)};
    const IndexRange kbf = IndexRange{kbs.s, kbs.e + (ndim > 2)};

    for (auto &pmb : pmesh->block_list) {
        auto &rc = pmb->meshblock_data.Get();

        for (int i = 0; i < BOUNDARY_NFACES; i++) {
            BoundaryFace bface = (BoundaryFace)i;
            auto bname = BoundaryName(bface);
            auto bdir = BoundaryDirection(bface);
            auto binner = BoundaryIsInner(bface);

            if (bdir > ndim) continue;

            // Set ranges for entire width.  Probably not needed for fluxes but won't hurt
            IndexRange ib = ibe, jb = jbe, kb = kbe;
            // Range for inner_x1 bounds is first face only, etc.
            if (bdir == 1) {
                ib.s = ib.e = (binner) ? ibf.s : ibf.e;
            } else if (bdir == 2) {
                jb.s = jb.e = (binner) ? jbf.s : jbf.e;
            } else {
                kb.s = kb.e = (binner) ? kbf.s : kbf.e;
            }

            PackIndexMap cons_map;
            auto &F = rc->PackVariablesAndFluxes({Metadata::WithFluxes}, cons_map);

            // If we should check inflow on this face...
            if (params.Get<bool>("check_inflow_" + bname)) {
                const int m_rho = cons_map["cons.rho"].first;
                // ...and if this face of the block corresponds to a global boundary...
                if (pmb->boundary_flag[bface] == BoundaryFlag::user) {
                    if (binner) {
                        pmb->par_for(
                            "zero_inflow_flux_" + bname, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                                F.flux(bdir, m_rho, k, j, i) = m::min(F.flux(bdir, m_rho, k, j, i), 0.);
                            }
                        );
                    } else {
                        pmb->par_for(
                            "zero_inflow_flux_" + bname, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                                F.flux(bdir, m_rho, k, j, i) = m::max(F.flux(bdir, m_rho, k, j, i), 0.);
                            }
                        );
                    }
                }
            }

            // If we should zero flux through this face...
            if (params.Get<bool>("zero_flux_" + bname)) {
                // ...and if this face of the block corresponds to a global boundary...
                if (pmb->boundary_flag[bface] == BoundaryFlag::user) {
                    pmb->par_for(
                        "zero_flux_" + bname, 0, F.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                        KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
                            F.flux(bdir, p, k, j, i) = 0.;
                        }
                    );
                }
            }
        }
    }

    return TaskStatus::complete;
}
