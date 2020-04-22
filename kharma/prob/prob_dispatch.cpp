// Dispatch for all problems which can be generated

#include "mesh/mesh.hpp"

#include "grid.hpp"
#include "phys.hpp"

#include "mhdmodes.hpp"
//#include "bondi.hpp"

using namespace parthenon;

/**
 * Override a parthenon method to generate the section of a problem redisiding in a given mesh block.
 */
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
    auto pmb = this;
    auto rc = pmb->real_containers.Get();
    auto P = rc.Get("c.c.bulk.prims").data;
    auto U = rc.Get("c.c.bulk.cons").data;

    Grid G(pmb);
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("cfl");
    EOS* eos = new GammaLaw(gamma);

    auto prob = pin->GetOrAddString("job", "problem_id", "mhdmodes");
    if (prob == "mhdmodes") {
        int nmode = pin->GetOrAddInteger("mhdmodes", "nmode", 1);
        int dir = pin->GetOrAddInteger("mhdmodes", "dir", 0);
        Real tf = mhdmodes(pmb, G, P, nmode, dir);
        if (nmode != 0) {
            pin->SetReal("time", "tlim", tf);
        }
    } else if (prob == "bondi") {

    }
    pmb->par_for("first_flux", pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke,
        KOKKOS_LAMBDA_3D {
            FourVectors Dtmp;
            get_state(G, P, i, j, k, Loci::center, Dtmp);
            prim_to_flux(G, P, Dtmp, eos, i, j, k, Loci::center, 0, U);
        }
    );
    FLAG("Initialized problem"); // TODO this is going to get called in every meshblock...
}