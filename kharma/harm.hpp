// HARM Driver: implement the HARM scheme for GRMHD as described in Gammie et al 2003, 2004
#pragma once

#include <memory>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/multistage.hpp"
#include "interface/StateDescriptor.hpp"
#include "task_list/tasks.hpp"

using namespace parthenon;

/**
 * A Driver object orchestrates everything that has to be done to a mesh to constitute a step.
 * For HARM, this means the predictor-corrector steps of fluid evolution
 */
class HARMDriver : public MultiStageBlockTaskDriver {
    public:
        HARMDriver(ParameterInput *pin, Mesh *pm, Outputs *pout) : MultiStageBlockTaskDriver(pin, pm, pout) {}
        TaskList MakeTaskList(MeshBlock *pmb, int stage);
};