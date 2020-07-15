// HARM Driver: implement the HARM scheme for GRMHD as described in Gammie et al 2003, 2004
#pragma once

#include <memory>

#include "parthenon/parthenon.hpp"

using namespace parthenon;

/**
 * A Driver object orchestrates everything that has to be done to a mesh to constitute a step.
 * For HARM, this means the predictor-corrector steps of fluid evolution
 */
class HARMDriver : public MultiStageBlockTaskDriver {
    public:
        HARMDriver(ParameterInput *pin, Mesh *pm) : MultiStageBlockTaskDriver(pin, pm) {}
        TaskList MakeTaskList(MeshBlock *pmb, int stage);
};