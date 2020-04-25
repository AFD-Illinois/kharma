
#include "mesh/mesh.hpp"

#include "decs.hpp"

using namespace parthenon;

void count_print_pflags(MeshBlock *pmb, const ParArrayND<int> pflag, bool include_ghosts=true);

void count_print_fflags(MeshBlock *pmb, const ParArrayND<int> fflag, bool include_ghosts=false);