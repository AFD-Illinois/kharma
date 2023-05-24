#pragma once

#include "decs.hpp"
#include "types.hpp"

#include "b_ct.hpp"
#include "domain.hpp"

using namespace parthenon;

/**
 * relativistic version of the Orszag-Tang vortex 
 * Orszag & Tang 1979, JFM 90, 129-143. 
 * original OT problem was incompressible 
 * this is based on compressible version given
 * in Toth 2000, JCP 161, 605.
 * 
 * in the limit tscale -> 0 the problem is identical
 * to the nonrelativistic problem; as tscale increases
 * the problem becomes increasingly relativistic
 * 
 * Originally stolen directly from iharm2d_v3,
 * now somewhat modified
 */
TaskStatus InitializeOrszagTang(std::shared_ptr<MeshBlockData<Real>>& rc, ParameterInput *pin)
{
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real tscale = pin->GetOrAddReal("orszag_tang", "tscale", 0.05);
    // Default phase puts the current sheet in the middle of the domain
    const Real phase = pin->GetOrAddReal("orszag_tang", "phase", M_PI);

    // TODO coord_embed for snake coords?

    IndexDomain domain = IndexDomain::entire;
    IndexRange3 b = KDomain::GetRange(rc, domain);
    pmb->par_for("ot_init", b.ks, b.ke, b.js, b.je, b.is, b.ie,
        KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
            Real X[GR_DIM];
            G.coord(k, j, i, Loci::center, X);
            rho(k, j, i) = 25./9.;
            u(k, j, i) = 5./(3.*(gam - 1.)) * tscale * tscale;
            uvec(0, k, j, i) = -m::sin(X[2] + phase) * tscale;
            uvec(1, k, j, i) = m::sin(X[1] + phase) * tscale;
            uvec(2, k, j, i) = 0.;
        }
    );

    if (pmb->packages.AllPackages().count("B_CT")) {
        auto B_Uf = rc->PackVariables(std::vector<std::string>{"cons.fB"});
        // Halo one zone right for faces
        // We don't need any more than that, since curls never take d1dx1
        IndexRange3 bA = KDomain::GetRange(rc, IndexDomain::entire, 0, 0);
        IndexSize3 s = KDomain::GetBlockSize(rc);
        GridVector A("A", NVEC, s.n3, s.n2, s.n1);
        pmb->par_for("ot_A", bA.ks, bA.ke, bA.js, bA.je, bA.is, bA.ie,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                Real Xembed[GR_DIM];
                G.coord(k, j, i, Loci::corner, Xembed);
                A(V3, k, j, i)  = (-0.5*std::cos(2*Xembed[1] + phase)
                                   + std::cos(Xembed[2] + phase)) * tscale;
            }
        );
        // This fills a couple zones outside the exact interior with bad data
        IndexRange3 bB = KDomain::GetRange(rc, domain, 0, -1);
        pmb->par_for("ot_B", bB.ks, bB.ke, bB.js, bB.je, bB.is, bB.ie,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                B_CT::curl_2D(G, A, B_Uf, k, j, i);
            }
        );
        B_CT::BlockUtoP(rc.get(), IndexDomain::entire, false);
        double max_divb = B_CT::BlockMaxDivB(rc.get());
        std::cout << "Block max DivB: " << max_divb << std::endl;

    } else if (pmb->packages.AllPackages().count("B_FluxCT") ||
               pmb->packages.AllPackages().count("B_CD")) {
        GridVector B_P = rc->Get("prims.B").data;
        pmb->par_for("ot_B", b.ks, b.ke, b.js, b.je, b.is, b.ie,
            KOKKOS_LAMBDA (const int &k, const int &j, const int &i) {
                Real X[GR_DIM];
                G.coord(k, j, i, Loci::center, X);
                B_P(V1, k, j, i) = -m::sin(X[2] + phase) * tscale;
                B_P(V2, k, j, i) = m::sin(2.*(X[1] + phase)) * tscale;
                B_P(V3, k, j, i) = 0.;
            }
        );
        B_FluxCT::BlockPtoU(rc.get(), IndexDomain::entire, false);
    }

    return TaskStatus::complete;
}
