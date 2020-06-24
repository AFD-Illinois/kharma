// Current calculation.  Override Parthenon's "UserWorkBeforeOutput"
// Hence there is no specific header file.

#include "decs.hpp"

#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
    // TODO implement.  Possibly lots of snags on account of is member of meshblock.
    // CPU wouldn't be the end of the world
}