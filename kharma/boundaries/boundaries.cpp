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
#include "kharma.hpp"
#include "flux.hpp"
#include "flux_functions.hpp"
#include "grmhd_functions.hpp"
#include "pack.hpp"
#include "types.hpp"

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
    }
    params.Add("fix_corner", fix_corner);

    Metadata m_x1, m_x2, m_x3;
    {
        // We can't use GetVariablesByFlag yet, so ask the packages
        int nvar = KHARMA::PackDimension(packages.get(), Metadata::FillGhost);

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
        // Dirichlet conditions must be restored when restarting!  Needs Metadata::Restart when this works!
        m_x1 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy}, s_x1);
        m_x2 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy}, s_x2);
        m_x3 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy}, s_x3);
    }

    // Set options for each boundary
    for (int i = 0; i < BOUNDARY_NFACES; i++)
    {
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
        bool check_inflow_flux = pin->GetOrAddBoolean("boundaries", "check_inflow_flux_" + bname, check_inflow);
        params.Add("check_inflow_flux_" + bname, check_inflow_flux);

        // Ensure fluxes through the zero-size face at the pole are zero
        bool zero_flux = pin->GetOrAddBoolean("boundaries", "zero_flux_" + bname, zero_polar_flux && bdir == X2DIR);
        params.Add("zero_flux_" + bname, zero_flux);

        // BOUNDARY TYPES
        // Get the boundary type we specified in kharma
        auto btype = pin->GetString("boundaries", bname);
        params.Add(bname, btype);

        // String manip to get the Parthenon boundary name, e.g., "ox1_bc"
        auto bname_parthenon = bname.substr(0, 1) + "x" + bname.substr(7, 8) + "_bc";
        // Parthenon implements periodic conditions
        // For the rest, they should call our default wrapper, which we register in main()
        if (btype == "periodic") {
            pin->SetString("parthenon/mesh", bname_parthenon, "periodic");
        } else {
            pin->SetString("parthenon/mesh", bname_parthenon, "user");
        }

        // TODO TODO any way to save this verbosity with constexpr/macros/something?
        if (btype == "dirichlet") {
            // Dirichlet boundaries: allocate
            pkg->AddField("bounds." + bname, (bdir == X1DIR) ? m_x1 : ((bdir == X2DIR) ? m_x2 : m_x3));
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

    const auto bface = BoundaryFaceOf(domain);
    const auto bname = BoundaryName(bface);
    const auto btype_name = params.Get<std::string>(bname);
    const auto bdir = BoundaryDirection(bface);

    Flag("Apply "+bname+" boundary: "+btype_name);
    pkg->KBoundaries[bface](rc, coarse);
    EndFlag();

    // Prevent inflow of material by changing fluid speeds,
    // anywhere we've specified.
    if (params.Get<bool>("check_inflow_" + bname)) {
        Flag("CheckInflow");
        CheckInflow(rc, domain, coarse);
        EndFlag();
    }

    // If specified, fix corner values when applying X2 boundaries (see function)
    if (params.Get<bool>("fix_corner") && bdir == X2DIR) {
        Flag("FixCorner");
        FixCorner(rc, domain, coarse);
        EndFlag();
    }

    // Respect the fluid primitives on boundaries (does not include B)
    // Also currently the EMHD extra variables q, dP
    Packages::BoundaryPtoU(rc.get(), domain, coarse);
    // For everything else, respect conserved variables
    Packages::BoundaryUtoP(rc.get(), domain, coarse);

    EndFlag();
}

void KBoundaries::CheckInflow(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse)
{
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    const auto &G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    PackIndexMap prims_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map, coarse);
    const VarMap m_p(prims_map, false);

    // Inflow check
    // Iterate over zones w/p=0
    pmb->par_for_bndry(
        "Outflow_check_inflow", IndexRange{0, 0}, domain, CC, coarse,
        KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
            KBoundaries::check_inflow(G, P, domain, m_p.U1, k, j, i);
        }
    );
}

void KBoundaries::FixCorner(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse)
{
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    if (pmb->pmy_mesh->ndim < 2)
        return;

    // If we're on the interior edge, re-apply that edge for our block by calling
    // exactly the same function that Parthenon does.  This ensures we're applying
    // the same thing, just emulating calling it after X2.
    if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user)
    {
        ApplyBoundary(rc, IndexDomain::inner_x1, coarse);
    }
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
    // Ranges for sides
    const IndexRange ibs = pmb0->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jbs = pmb0->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kbs = pmb0->cellbounds.GetBoundsK(IndexDomain::interior);
    // Ranges for faces
    const IndexRange ibf = IndexRange{ibs.s, ibs.e + 1};
    const IndexRange jbf = IndexRange{jbs.s, jbs.e + (ndim > 1)};
    const IndexRange kbf = IndexRange{kbs.s, kbs.e + (ndim > 2)};

    for (auto &pmb : pmesh->block_list)
    {
        auto &rc = pmb->meshblock_data.Get();

        for (int i = 0; i < BOUNDARY_NFACES; i++)
        {
            BoundaryFace bface = (BoundaryFace)i;
            auto bname = BoundaryName(bface);
            auto bdir = BoundaryDirection(bface);
            auto binner = BoundaryIsInner(bface);

            // Set ranges based
            IndexRange ib = ibs, jb = jbs, kb = kbs;
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
            if (params.Get<bool>("check_inflow_flux_" + bname)) {
                const int m_rho = cons_map["cons.rho"].first;
                // ...and if this face of the block corresponds to a global boundary...
                if (pmb->boundary_flag[bface] == BoundaryFlag::user) {
                    pmb->par_for(
                        "zero_inflow_flux_" + bname, kb.s, kb.e, jb.s, jb.e, ib.s, ib.s,
                        KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
                            F.flux(X1DIR, m_rho, k, j, i) = m::min(F.flux(X1DIR, m_rho, k, j, i), 0.);
                        });
                }
            }

            // If we should zero flux through this face...
            if (params.Get<bool>("zero_flux_" + bname)) {
                // ...and if this face of the block corresponds to a global boundary...
                if (pmb->boundary_flag[bface] == BoundaryFlag::user) {
                    pmb->par_for(
                        "zero_flux_" + bname, 0, F.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.s, ib.s, ib.e,
                        KOKKOS_LAMBDA(const int &p, const int &k, const int &j, const int &i) {
                            F.flux(X2DIR, p, k, j, i) = 0.;
                        });
                }
            }
        }
    }

    return TaskStatus::complete;
}
