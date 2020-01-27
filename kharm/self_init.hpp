

#include "decs.hpp"

GridPrimsHost mhdmodes(int n1, int n2, int n3, int nprim) {
    // TODO check nprim >= 8
    // TODO init
    return GridPrimsHost("prims_initial", n1, n2, n3, nprim);
}