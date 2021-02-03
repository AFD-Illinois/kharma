// HARM Driver: implement the HARM scheme for GRMHD as described in Gammie et al 2003, 2004
#pragma once

#include <memory>

#include <parthenon/parthenon.hpp>

using namespace parthenon;

/**
 * A Driver object orchestrates everything that has to be done to a mesh to constitute a step.
 * For HARM, this means the predictor-corrector steps of fluid evolution
 */
class HARMDriver : public MultiStageDriver {
    public:
        /**
         * Default constructor
         */
        HARMDriver(ParameterInput *pin, ApplicationInput *papp, Mesh *pm) : MultiStageDriver(pin, papp, pm) {}

        /**
         * All the tasks which constitute advancing the fluid in a mesh by one stage.
         * This includes calculation of the primitives and reconstruction of their face values,
         * calculation of conserved values and fluxes thereof at faces,
         * application of fluxes and a source term in order to update zone values,
         * and finally calculation of the next timestep based on the CFL condition.
         * 
         * The function is heavily documented since order changes can introduce subtle bugs,
         * usually w.r.t. fluid "state" being spread across the primitive and conserved quantities
         */
        TaskCollection MakeTaskCollection(BlockList_t &blocks, int stage);
};
