/**
 * In a perfect world, I would hand my loop specs to Kokkos, and it would make them fast.
 * Instead, we have this.
 */
#pragma once

#include "decs.hpp"

#include <string>

#define MDRANGE 1
#define PEELP 0
#define PEELI 0
#define PEELJ 0
#define PEELK 0
#define NO_KOKKOS 0

// TODO use fancy tricks to intercept the Kokkos::parallel_for (or just parallel_for) fn call?

template<typename RangeType, typename LambdaType>
void kharm_for(std::string name, RangeType range, LambdaType fn)
{
    cerr << "Custom parfor" << endl;
    Kokkos::parallel_for(name, range, fn);
}