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
        /**
         * Default constructor
         */
        HARMDriver(ParameterInput *pin, ApplicationInput *papp, Mesh *pm) : MultiStageBlockTaskDriver(pin, papp, pm) {}

        /**
         * All the tasks which constitute advancing the fluid in a mesh block by a stage.
         * This includes calculation of necessary derived variables, reconstruction, calculation of fluxes,
         * Application of fluxes and a source term to update zones, and finally calculation of the next
         * timestep.
         * 
         * The function is heavily documented since order changes can introduce subtle bugs
         */
        TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
};