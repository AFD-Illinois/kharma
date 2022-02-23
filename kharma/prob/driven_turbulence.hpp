#pragma once

#include "decs.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace std;
using namespace parthenon;

/**
 * Generic initializer for shock tubes
 * Particular problems in pars/shocks/
 * 
 * Stolen directly from iharm3D
 */
TaskStatus InitializeDrivenTurbulence(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag(rc, "Initializing Shock Tube problem");
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const auto& G = pmb->coords;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");

    const Real rho0 = pin->GetOrAddReal("driven_turbulence", "rho", 0.0);
    const Real cs0 = pin->GetOrAddReal("driven_turbulence", "cs0", 0.0);
    const Real edot_frac = pin->GetOrAddReal("driven_turbulence", "edot_frac", 0.5);
    const Real edot = edot_frac * rho0 * pow(cs0, 3);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("drive_cs0")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("drive_cs0", cs0);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("drive_edot")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("drive_edot", edot);

    const Real u0 = sqrt(cs0 * cs0 * rho0 / (gam - 1));

    IndexDomain domain = IndexDomain::entire;
    IndexRange ib = pmb->cellbounds.GetBoundsI(domain);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(domain);
    IndexRange kb = pmb->cellbounds.GetBoundsK(domain);

    const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
    const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
    const Real center = (x1min + x1max) / 2.;

    pmb->par_for("ot_init", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA_3D {
            Real X[GR_DIM];
            G.coord(k, j, i, Loci::center, X);

            const bool lhs = X[1] < center;
            rho(k, j, i) = rho0;
            u(k, j, i) = u0;
            VLOOP uvec(v, k, j, i) = 0;
        }
    );

    if(pmb->packages.AllPackages().count("Electrons")) {
        // Get e- starting parameters

        // Set e- starting state
    }

    return TaskStatus::complete;
}
