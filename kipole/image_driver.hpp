// HARM Driver: implement the HARM scheme for GRMHD as described in Gammie et al 2003, 2004
#pragma once

#include <parthenon/parthenon.hpp>

using namespace parthenon;

/**
 * A Driver object orchestrates everything that has to be done to a mesh to constitute a step.
 * For HARM, this means the predictor-corrector steps of fluid evolution
 */
class ImageDriver : public Driver {
    public:
        /**
         * Default constructor
         */
        ImageDriver(ParameterInput *pin, ApplicationInput *papp, Mesh *pm) : Driver(pin, papp, pm) {}

        /**
         * @brief Execute the driver; create a single image from a loaded mesh
         * 
         * @return DriverStatus code for success or failure
         */
        DriverStatus Execute();
};
