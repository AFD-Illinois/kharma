// Dispatch for all problems which can be generated

#include "mesh/mesh.hpp"

#include "mhdmodes.hpp"
//#include "bondi.hpp"

using namespace parthenon;

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
    auto prob = pin->GetOrAddString("Hydro", "problem", "mhdmodes");
    if (prob == "mhdmodes") {
        int nmode = pin->GetOrAddInteger("Hydro", "nmode", 1);
        mhdmodes(pmb, nmode);
    }
}