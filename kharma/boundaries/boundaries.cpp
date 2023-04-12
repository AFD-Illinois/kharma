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

std::shared_ptr<KHARMAPackage> KBoundaries::Initialize(ParameterInput *pin, std::shared_ptr<Packages_t>& packages)
{
    Flag("Initializing Boundaries");

    auto pkg = std::make_shared<KHARMAPackage>("Boundaries");
    Params &params = pkg->AllParams();

    // Prevent inflow at boundaries.
    // This is two separate checks, but default to enabling/disabling together
    bool spherical = pin->GetBoolean("coordinates", "spherical");
    bool check_inflow = pin->GetOrAddBoolean("boundaries", "check_inflow", spherical);
    bool check_inflow_inner = pin->GetOrAddBoolean("boundaries", "check_inflow_inner", check_inflow);
    params.Add("check_inflow_inner", check_inflow_inner);
    bool check_inflow_flux_inner = pin->GetOrAddBoolean("boundaries", "check_inflow_flux_inner", check_inflow_inner);
    params.Add("check_inflow_flux_inner", check_inflow_flux_inner);
    bool check_inflow_outer = pin->GetOrAddBoolean("boundaries", "check_inflow_outer", check_inflow);
    params.Add("check_inflow_outer", check_inflow_outer);
    bool check_inflow_flux_outer = pin->GetOrAddBoolean("boundaries", "check_inflow_flux_outer", check_inflow_outer);
    params.Add("check_inflow_flux_outer", check_inflow_flux_outer);

    // Ensure fluxes through the zero-size face at the pole are zero
    bool fix_flux_pole = pin->GetOrAddBoolean("boundaries", "fix_flux_pole", spherical);
    params.Add("fix_flux_pole", fix_flux_pole);

    // Fix the X1/X2 corner by replacing the reflecting condition with the inflow
    // Only needed if x1min is inside BH event horizon, otherwise a nuisance for divB on corners
    if (spherical) {
        const Real a = pin->GetReal("coordinates", "a");
        bool inside_eh = pin->GetBoolean("coordinates", "r_in") < (1 + sqrt(1 - a*a));
        bool fix_corner = pin->GetOrAddBoolean("boundaries", "fix_corner", inside_eh);
        params.Add("fix_corner", fix_corner);
    }

    // Allocate space for Dirichlet boundaries if they'll be used
    // We have to trust the user here since the problem will set the function pointers later
    // TODO specify which boundaries individually for cleanliness?
    bool use_dirichlet = pin->GetOrAddBoolean("boundaries", "prob_uses_dirichlet", false);
    params.Add("use_dirichlet", use_dirichlet);
    if (use_dirichlet) {
        auto& driver = packages->Get("Driver")->AllParams();

        // We can't use GetVariablesByFlag yet, so walk through and count manually
        int nvar = 0;
        for (auto pkg : packages->AllPackages()) {
            //std::cerr << pkg.first << ": ";
            for (auto field : pkg.second->AllFields()) {
                //std::cerr << field.first.label() << " ";
                // Specifically ignore the B_Cleanup variables, we don't handle their boundary conditions
                if (field.second.IsSet(Metadata::FillGhost) && !field.second.IsSet(Metadata::GetUserFlag("B_Cleanup"))) {
                    if (field.second.Shape().size() < 1) {
                        nvar += 1;
                    } else {
                        nvar += field.second.Shape()[0];
                    }
                }
            }
            //std::cerr << std::endl;
        }

        // We also don't know the mesh size, since it's not constructed.  We infer.
        const int ng = pin->GetInteger("parthenon/mesh", "nghost");
        const int nx1 = pin->GetInteger("parthenon/meshblock", "nx1");
        const int n1 = nx1 + 2*ng;
        const int nx2 = pin->GetInteger("parthenon/meshblock", "nx2");
        const int n2 = (nx2 == 1) ? nx2 : nx2 + 2*ng;
        const int nx3 = pin->GetInteger("parthenon/meshblock", "nx3");
        const int n3 = (nx3 == 1) ? nx3 : nx3 + 2*ng;

        if (pin->GetInteger("debug", "verbose") > 0) {
            std::cout << "Allocating Dirichlet boundaries for " << nvar << " variables." << std::endl;
            if (pin->GetInteger("debug", "verbose") > 1) {
                std::cout << "Initializing Dirichlet bounds with dimensions nvar,n1,n2,n3: " << nvar << " " << n1 << " " << n2 << " " << n3 << " and ng: " << ng << std::endl;
            }
        }

        // These are declared *backward* from how they will be indexed
        std::vector<int> s_x1({ng, n2, n3, nvar});
        std::vector<int> s_x2({n1, ng, n3, nvar});
        std::vector<int> s_x3({n1, n2, ng, nvar});
        // Dirichlet conditions must be restored when restarting!  Needs Metadata::Restart when this works!
        Metadata m_x1 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy}, s_x1);
        Metadata m_x2 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy}, s_x2);
        Metadata m_x3 = Metadata({Metadata::Real, Metadata::Derived, Metadata::OneCopy}, s_x3);
        pkg->AddField("bound.inner_x1", m_x1);
        pkg->AddField("bound.outer_x1", m_x1);
        pkg->AddField("bound.inner_x2", m_x2);
        pkg->AddField("bound.outer_x2", m_x2);
        pkg->AddField("bound.inner_x3", m_x3);
        pkg->AddField("bound.outer_x3", m_x3);
    }

    // Callbacks
    // Fix flux
    pkg->FixFlux = KBoundaries::FixFlux;

    // KHARMA boundary functions take a domain and are trusted to handle it
    pkg->KHARMAInnerX1Boundary = KBoundaries::DefaultBoundary;
    pkg->KHARMAOuterX1Boundary = KBoundaries::DefaultBoundary;
    pkg->KHARMAInnerX2Boundary = KBoundaries::DefaultBoundary;
    pkg->KHARMAOuterX2Boundary = KBoundaries::DefaultBoundary;

    Flag("Initialized");
    return pkg;
}

void KBoundaries::ApplyBoundary(std::shared_ptr<MeshBlockData<Real>> &rc, IndexDomain domain, bool coarse)
{
    Flag("Applying a KHARMA boundary");
    // KHARMA has to do some extra tasks in addition to just applying the usual
    // boundary conditions.  Therefore, we "wrap" Parthenon's (or our own)
    // boundary functions with this one.
    // TODO call for all packages?

    auto pmb = rc->GetBlockPointer();
    auto pkg = static_cast<KHARMAPackage*>(pmb->packages.Get("Boundaries").get());

    // Disambiguate in order to call our pointers
    int dir = BoundarySide(domain);
    if (dir == 1) {
        if (BoundaryIsInner(domain)) {
            pkg->KHARMAInnerX1Boundary(rc, domain, coarse);
        } else {
            pkg->KHARMAOuterX1Boundary(rc, domain, coarse);
        }
    } else if (dir == 2) {
        if (BoundaryIsInner(domain)) {
            pkg->KHARMAInnerX2Boundary(rc, domain, coarse);
        } else {
            pkg->KHARMAOuterX2Boundary(rc, domain, coarse);
        }
    }

    // Respect the fluid primitives on boundaries (*not* B)
    Flux::BlockPtoUMHD(rc.get(), domain, coarse);
    // For everything else, respect conserved variables
    Packages::BlockUtoPExceptMHD(rc.get(), domain, coarse);

    Flag("Applied boundary");
}

void KBoundaries::CheckInflow(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "Checking inflow");
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    const auto& G = pmb->coords;
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    bool check_inner = pmb->packages.Get("Boundaries")->Param<bool>("check_inflow_inner");
    bool check_outer = pmb->packages.Get("Boundaries")->Param<bool>("check_inflow_outer");
    const bool check_inflow = ((check_inner && domain == IndexDomain::inner_x1)
                            || (check_outer && domain == IndexDomain::outer_x1));
    if (!check_inflow) return;

    PackIndexMap prims_map;
    auto P = GRMHD::PackMHDPrims(rc.get(), prims_map, coarse);
    const VarMap m_p(prims_map, false);

    // Inflow check
    // Iterate over zones w/p=0
    pmb->par_for_bndry("Outflow_check_inflow", IndexRange{0,0}, domain, coarse,
        KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
            KBoundaries::check_inflow(G, P, domain, m_p.U1, k, j, i);
        }
    );

    Flag(rc, "Checked");
}

void KBoundaries::FixCorner(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "Fixing X1/X2 corner block");
    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    if (pmb->pmy_mesh->ndim < 2 ||
        !pmb->packages.Get("Boundaries")->Param<bool>("fix_corner"))
        return;

    // If we're on the interior edge, re-apply that edge for our block by calling
    // exactly the same function that Parthenon does.  This ensures we're applying
    // the same thing, just emulating calling it after X2.
    if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
        ApplyBoundary(rc, IndexDomain::inner_x1, coarse);
    }

    Flag(rc, "Fixed");
}

// void KBoundaries::CorrectBField(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
// {
//     Flag(rc, "Correcting the B field w/metric");
//     std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
//     const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

//     auto B_P = rc->PackVariables(std::vector<std::string>{"prims.B"});
//     // Return if no field to correct
//     if (B_P.GetDim(4) == 0) return;

//     const auto& G = pmb->coords;

//     const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
//     const int dir = BoundarySide(domain);
//     const auto &range = (dir == 1) ? bounds.GetBoundsI(IndexDomain::interior)
//                             : (dir == 2 ? bounds.GetBoundsJ(IndexDomain::interior)
//                                 : bounds.GetBoundsK(IndexDomain::interior));
//     const int ref = BoundaryIsInner(domain) ? range.s : range.e;

//     pmb->par_for_bndry("Correct_B_P", IndexRange{0,NVEC-1}, domain, coarse,
//         KOKKOS_LAMBDA (const int &v, const int &k, const int &j, const int &i) {
//             B_P(v, k, j, i) *= G.gdet(Loci::center, (dir == 2) ? ref : j, (dir == 1) ? ref : i)
//                             / G.gdet(Loci::center, j, i);
//         }
//     );

//     Flag(rc, "Corrected");
// }

void KBoundaries::DefaultBoundary(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
{
    // Default function for applying any (non-periodic) boundary condition:
    // outflow in X1 with inflow check, Reflect in X2 with corner fix
    auto pmb = rc->GetBlockPointer();
    const int dir = BoundarySide(domain);
    if (dir == 1) {
        if (BoundaryIsInner(domain)) {
            parthenon::BoundaryFunction::OutflowInnerX1(rc, coarse);
        } else {
            parthenon::BoundaryFunction::OutflowOuterX1(rc, coarse);
        }
        CheckInflow(rc, domain, coarse);
    } else if (dir == 2) {
        if (BoundaryIsInner(domain)) {
            parthenon::BoundaryFunction::ReflectInnerX2(rc, coarse);
        } else {
            parthenon::BoundaryFunction::ReflectOuterX2(rc, coarse);
        }
        FixCorner(rc, domain, coarse);
    }
}

void KBoundaries::SetDomainDirichlet(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse) {
    Flag("Setting Dirichlet bound");

    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    using FC = Metadata::FlagCollection;
    auto q = rc->PackVariables(FC({Metadata::FillGhost}) - FC({Metadata::GetUserFlag("B_Cleanup")}), coarse);
    auto bound = rc->Get("bound."+BoundaryName(domain)).data;

    if (q.GetDim(4) != bound.GetDim(4)) {
        std::cerr << "Boundary cache mismatch! " << bound.GetDim(4) << " vs " << q.GetDim(4) << std::endl;
    }

    const IndexRange vars = IndexRange{0, q.GetDim(4) - 1};
    const bool right = !BoundaryIsInner(domain);

    // Subtract off the starting index if we're on the right
    const auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const int dir = BoundarySide(domain);
    const int ie = (dir == 1) ? bounds.ie(IndexDomain::interior)+1 : 0;
    const int je = (dir == 2) ? bounds.je(IndexDomain::interior)+1 : 0;
    const int ke = (dir == 3) ? bounds.ke(IndexDomain::interior)+1 : 0;

    const auto& G = pmb->coords;

    pmb->par_for_bndry("dirichlet_boundary", vars, domain, coarse,
        KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
            if (right) {
                bound(p, k-ke, j-je, i-ie) = q(p, k, j, i);
            } else {
                bound(p, k, j, i) = q(p, k, j, i);
            }
        }
    );

    Flag("Set");
}

void KBoundaries::Dirichlet(std::shared_ptr<MeshBlockData<Real>>& rc, IndexDomain domain, bool coarse)
{
    Flag(rc, "Applying Dirichlet bound");

    std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    using FC = Metadata::FlagCollection;
    auto q = rc->PackVariables(FC({Metadata::FillGhost}) - FC({Metadata::GetUserFlag("B_Cleanup")}), coarse);
    auto bound = rc->Get("bound."+BoundaryName(domain)).data;

    if (q.GetDim(4) != bound.GetDim(4)) {
        std::cerr << "Boundary cache mismatch! " << bound.GetDim(4) << " vs " << q.GetDim(4) << std::endl;
    }

    const IndexRange vars = IndexRange{0, q.GetDim(4) - 1};
    const bool right = !BoundaryIsInner(domain);

    // Subtract off the starting index if we're on the right
    const auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
    const int dir = BoundarySide(domain);
    const int ie = (dir == 1) ? bounds.ie(IndexDomain::interior)+1 : 0;
    const int je = (dir == 2) ? bounds.je(IndexDomain::interior)+1 : 0;
    const int ke = (dir == 3) ? bounds.ke(IndexDomain::interior)+1 : 0;

    const auto& G = pmb->coords;

    pmb->par_for_bndry("dirichlet_boundary", vars, domain, coarse,
        KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
            if (right) {
                q(p, k, j, i) = bound(p, k-ke, j-je, i-ie);
            } else {
                q(p, k, j, i) = bound(p, k, j, i);
            }
        }
    );

    Flag(rc, "Applied");
}

TaskStatus KBoundaries::FixFlux(MeshData<Real> *md)
{
    Flag("Fixing fluxes");
    auto pmesh = md->GetMeshPointer();
    auto pmb0 = md->GetBlockData(0)->GetBlockPointer();

    auto& params = pmb0->packages.Get("Boundaries")->AllParams();
    bool check_inflow_inner = params.Get<bool>("check_inflow_flux_inner");
    bool check_inflow_outer = params.Get<bool>("check_inflow_flux_outer");
    bool fix_flux_pole = params.Get<bool>("fix_flux_pole");

    IndexDomain domain = IndexDomain::interior;
    const int is = pmb0->cellbounds.is(domain), ie = pmb0->cellbounds.ie(domain);
    const int js = pmb0->cellbounds.js(domain), je = pmb0->cellbounds.je(domain);
    const int ks = pmb0->cellbounds.ks(domain), ke = pmb0->cellbounds.ke(domain);
    const int ndim = pmesh->ndim;

    // Fluxes are defined at faces, so there is one more valid flux than
    // valid cell in the face direction.  That is, e.g. F1 is valid on
    // an (N1+1)xN2xN3 grid, F2 on N1x(N2+1)xN3, etc.
    // These functions do *not* need an extra row outside the domain,
    // like B_FluxCT::FixBoundaryFlux does.
    const int ie_l = ie + 1;
    const int je_l = (ndim > 1) ? je + 1 : je;
    //const int ke_l = (ndim > 2) ? ke + 1 : ke;

    for (auto &pmb : pmesh->block_list) {
        auto& rc = pmb->meshblock_data.Get();

        PackIndexMap cons_map;
        auto& F = rc->PackVariablesAndFluxes({Metadata::WithFluxes}, cons_map);
        const int m_rho = cons_map["cons.rho"].first;

        if (check_inflow_inner) {
            if (pmb->boundary_flag[BoundaryFace::inner_x1] == BoundaryFlag::user) {
                pmb->par_for("fix_flux_in_l", ks, ke, js, je, is, is,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        F.flux(X1DIR, m_rho, k, j, i) = m::min(F.flux(X1DIR, m_rho, k, j, i), 0.);
                    }
                );
            }
        }
        if (check_inflow_outer) {
            if (pmb->boundary_flag[BoundaryFace::outer_x1] == BoundaryFlag::user) {
                pmb->par_for("fix_flux_in_r", ks, ke, js, je, ie_l, ie_l,
                    KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                        F.flux(X1DIR, m_rho, k, j, i) = m::max(F.flux(X1DIR, m_rho, k, j, i), 0.);
                    }
                );
            }
        }

        // This is a lot of zero fluxes!
        if (fix_flux_pole) {
            if (pmb->boundary_flag[BoundaryFace::inner_x2] == BoundaryFlag::user) {
                // This loop covers every flux we need
                pmb->par_for("fix_flux_pole_l", 0, F.GetDim(4) - 1, ks, ke, js, js, is, ie,
                    KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
                        F.flux(X2DIR, p, k, j, i) = 0.;
                    }
                );
            }

            if (pmb->boundary_flag[BoundaryFace::outer_x2] == BoundaryFlag::user) {
                pmb->par_for("fix_flux_pole_r", 0, F.GetDim(4) - 1, ks, ke, je_l, je_l, is, ie,
                    KOKKOS_LAMBDA (const int &p, const int &k, const int &j, const int &i) {
                        F.flux(X2DIR, p, k, j, i) = 0.;
                    }
                );
            }
        }
    }

    Flag("Fixed fluxes");
    return TaskStatus::complete;
}