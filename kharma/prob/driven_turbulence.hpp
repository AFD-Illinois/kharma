#pragma once

#include "decs.hpp"
#include "types.hpp"

#include <parthenon/parthenon.hpp>

using namespace std;
using namespace parthenon;

TaskStatus InitializeDrivenTurbulence(MeshBlockData<Real> *rc, ParameterInput *pin)
{
    Flag("InitializeDrivenTurbulence");
    auto pmb = rc->GetBlockPointer();
    GridScalar rho = rc->Get("prims.rho").data;
    GridScalar u = rc->Get("prims.u").data;
    GridVector uvec = rc->Get("prims.uvec").data;

    const Real gam = pmb->packages.Get("GRMHD")->Param<Real>("gamma");
    const Real rho0 = pin->GetOrAddReal("driven_turbulence", "rho", 1.0);
    const Real cs0 = pin->GetOrAddReal("driven_turbulence", "cs0", 8.6e-4);
    const Real dt_kick = pin->GetOrAddReal("driven_turbulence", "dt_kick", 1);
    const Real edot_frac = pin->GetOrAddReal("driven_turbulence", "edot_frac", 0.5);
    const Real x1min = pin->GetOrAddReal("parthenon/mesh", "x1min", 0);
    const Real x1max = pin->GetOrAddReal("parthenon/mesh", "x1max",  1);
    const Real x2min = pin->GetOrAddReal("parthenon/mesh", "x2min", 0);
    const Real x2max = pin->GetOrAddReal("parthenon/mesh", "x2max",  1);
    const Real x3min = pin->GetOrAddReal("parthenon/mesh", "x3min", -1);
    const Real x3max = pin->GetOrAddReal("parthenon/mesh", "x3max",  1);

    const Real edot = edot_frac * rho0 * pow(cs0, 3); const Real counter = 0.;
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("drive_edot")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("drive_edot", edot);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("counter")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("counter", counter, true);
    const Real lx1 = x1max-x1min;   const Real lx2 = x2max-x2min;
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("lx1")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("lx1", lx1);
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("lx2")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("lx2", lx2);
    //adding for later use in create_grf
    if(! (pmb->packages.Get("GRMHD")->AllParams().hasKey("dt_kick")))
        pmb->packages.Get("GRMHD")->AddParam<Real>("dt_kick", dt_kick);

    const Real u0 = cs0 * cs0 * rho0 / (gam - 1) / gam; //from flux_functions.hpp
    IndexRange myib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange myjb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange mykb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    pmb->par_for("driven_turb_rho_u_init", mykb.s, mykb.e, myjb.s, myjb.e, myib.s, myib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            rho(k, j, i) = rho0;
            u(k, j, i) = u0;
        }
    );

    EndFlag();
    return TaskStatus::complete;
}
