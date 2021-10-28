#pragma once

#include "decs.hpp"


using namespace std;
using namespace parthenon;

/**
 * Generic initializer for shock tubes
 * Particular problems in pars/shocks/
 * 
 * Stolen directly from iharm3D
 */
void InitializeShockTube(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    FLAG("Initializing Shock Tube problem");
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;
    GridVector B_P = rc->Get("prims.B").data;

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real rhoL = pin->GetOrAddReal("shock", "rhoL", 0.05);
    const Real rhoR = pin->GetOrAddReal("shock", "rhoR", 0.05);
    const Real PL = pin->GetOrAddReal("shock", "PL", 0.05);
    const Real PR = pin->GetOrAddReal("shock", "PR", 0.05);
    const Real u1L = pin->GetOrAddReal("shock", "u1L", 0.05);
    const Real u1R = pin->GetOrAddReal("shock", "u1R", 0.05);
    const Real u2L = pin->GetOrAddReal("shock", "u2L", 0.05);
    const Real u2R = pin->GetOrAddReal("shock", "u2R", 0.05);
    const Real u3L = pin->GetOrAddReal("shock", "u3L", 0.05);
    const Real u3R = pin->GetOrAddReal("shock", "u3R", 0.05);
    const Real B1L = pin->GetOrAddReal("shock", "B1L", 0.05);
    const Real B1R = pin->GetOrAddReal("shock", "B1R", 0.05);
    const Real B2L = pin->GetOrAddReal("shock", "B2L", 0.05);
    const Real B2R = pin->GetOrAddReal("shock", "B2R", 0.05);
    const Real B3L = pin->GetOrAddReal("shock", "B3L", 0.05);
    const Real B3R = pin->GetOrAddReal("shock", "B3R", 0.05);


    IndexDomain domain = IndexDomain::entire;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    pmb->par_for("ot_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord(k, j, i, Loci::center, X);

            rho(k, j, i) = (X[1] < 0.) ? rhoL : rhoR;
            u(k, j, i)   = ((X[1] < 0.) ? PL : PR)/(gam - 1.);
            uvec(0, k, j, i) = (X[1] < 0.) ? u1L : u1R;
            uvec(1, k, j, i) = (X[1] < 0.) ? u2L : u2R;
            uvec(2, k, j, i) = (X[1] < 0.) ? u3L : u3R;
            B_P(0, k, j, i)  = (X[1] < 0.) ? B1L : B1R;
            B_P(1, k, j, i)  = (X[1] < 0.) ? B2L : B2R;
            B_P(2, k, j, i)  = (X[1] < 0.) ? B3L : B3R;
        }
    );

    if(pmb->packages.AllPackages().count("Electrons")) {
        // Get e- starting parameters

        // Set e- starting state
    }
}
