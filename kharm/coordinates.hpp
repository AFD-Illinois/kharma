/*
 * coordinates.hpp:  Coordinate systems as objects
 */
#pragma once

#include "decs.hpp"

/**
 * Abstract class defining any coordinate system
 * 
 */
class CoordinateSystem {
    public:
        template <typename T> T to_embed(T native);
        template <typename T> T to_native(T embed);
    protected:
        template <typename T> T dxdX(T native);
};

/**
 * Class defining properties mapping a native coordinate system to minkowski space
 */
class Minkowski: public CoordinateSystem {

};