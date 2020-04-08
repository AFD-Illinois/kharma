/*
 * HARM driver-specific things -- i.e. call the GRMHD physics module in
 * the correct RK2 LLF steps we know and love
 */

#include <iostream>

#include "parthenon_manager.hpp"

#include "grmhd.hpp"
#include "harm.hpp"

// Parthenon requires we override certain things
namespace parthenon {

    Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput>& pin) {
        Packages_t packages;

        // Turn off GRMHD only if set to false in input file
        bool do_hydro = pin->GetOrAddBoolean("Physics", "GRMHD", true);

        // enable other packages as needed
        bool do_scalars = pin->GetOrAddBoolean("Physics", "scalars", false);

        if (do_hydro) {
            packages["GRMHD"] = GRMHD::Initialize(pin.get());
        }

        // TODO scalars. Or will Parthenon?
        // if (do_scalars) {
        //     packages["scalars"] = BetterScalars::Initialize(pin.get());
        // }

        // TODO electrons, like scalars but...

        return std::move(packages);
    }

} // namespace parthenon

// Implement HARMDriver class methods
TaskList HARMDriver::MakeTaskList(MeshBlock *pmb, int stage)
{
    // Make the task list for a "stage," i.e. one fluid evolution + boundary sync
    // Note that HARM only supports first-order ("rk1") and second-order ("rk2", recommended) operation
    using namespace GRMHD;
    TaskList tl;

    TaskID none(0);
    TaskID advance_fluid = tl.AddTask<BlockTask>(GRMHD::AdvanceFluid, none, pmb);
    //TaskID set_bounds = tl.AddTask<BlockTask>(GRMHD::SetBounds, none, pmb);
    //TaskID fixup = tl.AddTask<BlockTask>(GRMHD::Fixup, none, pmb);
    //TaskID set_bounds_flags = tl.AddTask<BlockTask>(GRMHD::SetBounds, none, pmb);
    //TaskID fixup_utop = tl.AddTask<BlockTask>(GRMHD::FixIntegrationFailures, none, pmb);
    // TODO timestep here?  What?

    return std::move(tl);
}