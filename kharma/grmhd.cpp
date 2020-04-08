// Functions defining the evolution of GRMHD fluid

#include <memory>

#include "grmhd.hpp"

using namespace parthenon;

namespace GRMHD
{

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin)
{
    auto fluid_state = std::make_shared<StateDescriptor>("GRMHD");
    Params &params = fluid_state->AllParams();

    // HARM is non-negotiably RK2 a.k.a. predictor-corrector for now
    params.Add("order", 2);

    // Only 2 fluid-related parameters:
    // 1. Fluid gamma for EOS (TODO separate EOS class to make this broader)
    // 2. Proportion of courant condition for timesteps
    double gamma = pin->GetOrAddReal("Hydro", "gamma", 5. / 3);
    double cfl = pin->GetOrAddReal("Hydro", "cfl", 0.9);
    params.Add("gamma", gamma);
    params.Add("cfl", cfl);

    // These are the conserved versions, NOT primitives -- U, not P.
    Metadata m;
    std::vector<int> v_size({3});
    std::vector<int> f_size({5}); // TODO that's what this is right?
    m = Metadata({m.Cell, m.Intensive, m.Conserved, m.Vector, m.Graphics, m.OneCopy}, f_size);
    fluid_state->AddField("c.c.bulk.cons", m, DerivedOwnership::shared);
    m = Metadata({m.Cell, m.Intensive, m.Conserved, m.Vector, m.Graphics, m.OneCopy}, v_size);
    fluid_state->AddField("c.c.bulk.cons_B", m, DerivedOwnership::shared);

    m = Metadata({m.Cell, m.Intensive, m.Derived, m.Vector, m.OneCopy}, f_size);
    fluid_state->AddField("c.c.bulk.prims", m, DerivedOwnership::shared);
    m = Metadata({m.Cell, m.Intensive, m.Derived, m.Vector, m.OneCopy}, v_size);
    fluid_state->AddField("c.c.bulk.prims_B", m, DerivedOwnership::shared);

    // Fluxes
    m = Metadata({m.Cell, m.Intensive, m.Derived, m.Vector, m.OneCopy}, f_size);
    fluid_state->AddField("c.c.bulk.F1", m, DerivedOwnership::shared);
    fluid_state->AddField("c.c.bulk.F2", m, DerivedOwnership::shared);
    fluid_state->AddField("c.c.bulk.F3", m, DerivedOwnership::shared);

    // Sound speed
    m = Metadata({m.Cell, m.Intensive, m.Derived, m.Vector, m.OneCopy}, v_size);
    fluid_state->AddField("c.c.bulk.ctop", m, DerivedOwnership::shared);

    fluid_state->FillDerived = nullptr; // TODO
    fluid_state->CheckRefinement = nullptr;
    return fluid_state;
}

TaskStatus AdvanceFluid(MeshBlock *pmb) {
    return TaskStatus::success;
}

} // namespace GRMHD