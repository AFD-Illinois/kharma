/**
 * Calculate fluxes through a zone
 */

#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"

#include "decs.hpp"

#include "reconstruction.hpp"
#include "phys.hpp"

void LRToFlux(ParArrayND<Real> pl, ParArrayND<Real> pr, const int dir, Container<Real>& rc);
void FluxCT(Container<Real>& rc);

using namespace parthenon;

TaskStatus CalculateFluxes(Container<Real>& rc)
{
    MeshBlock *pmb = rc.pmy_block;
    ParArrayND<Real> pl("pl", pmb->ncells1, pmb->ncells2, pmb->ncells3, NPRIM);
    ParArrayND<Real> pr("pr", pmb->ncells1, pmb->ncells2, pmb->ncells3, NPRIM);

    // Reconstruct primitives at left and right sides of faces
    WENO5X1(rc, pl, pr);
    FLAG("Recon 1");
    // Calculate flux from values at left & right of face
    LRToFlux(pr, pl, 1, rc);
    FLAG("LR 1");

    WENO5X2(rc, pl, pr);
    FLAG("Recon 2");
    LRToFlux(pr, pl, 2, rc);
    FLAG("LR 2");

    WENO5X3(rc, pl, pr);
    FLAG("Recon 3");
    LRToFlux(pr, pl, 3, rc);
    FLAG("LR 3");

    // Remember to fix boundary fluxes -- this will interact with Parthenon's version of the same...

    // Constrained transport for B must be applied after everything, including fixing boundary fluxes
    FluxCT(rc);
    FLAG("Flux CT");

    return TaskStatus::complete;
}

/**
 * Take reconstructed primitives at either side of a face and construct the local Lax-Friedrichs flux
 * 
 * Also fills the "ctop" vector with the highest magnetosonic speed mhd_vchar -- used to estimate timestep later.
 * 
 * Note that since this L and R are defined with respect to the *face*, they are actually the
 * opposite of the "r" and "l" above!
 */
void LRToFlux(ParArrayND<Real> pl, ParArrayND<Real> pr, const int dir, Container<Real>& rc)
{
    MeshBlock *pmb = rc.pmy_block;
    int is = pmb->is, js = pmb->js, ks = pmb->ks;
    int ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    ParArrayND<Real> pll("pll", pmb->ncells1, pmb->ncells2, pmb->ncells3);
    auto& flux = rc.Get("c.c.bulk.prims").flux[dir];
    auto& ctop = rc.Get("c.c.bulk.ctop").data;

    // So far we don't need fluxes that don't match faces
    Loci loc;

    // TODO *sigh*
    Grid G(pmb);
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("cfl");
    EOS* eos = new GammaLaw(gamma);

    // Offset "left" variables by one zone to line up L- and R-fluxes at *faces*
    // TODO can this be done without a copy/temporary? Fewer ranks?
    switch (dir) {
    case 1:
        pmb->par_for("offset_left_1", pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_VARS {
                pll(i, j, k, p) = pl(i-1, j, k, p);
            }
        );
        loc = Loci::face1;
        break;
    case 2:
        pmb->par_for("offset_left_2", pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_VARS {
                pll(i, j, k, p) = pl(i, j-1, k, p);
            }
        );
        loc = Loci::face2;
        break;
    case 3:
        pmb->par_for("offset_left_3", pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_VARS {
                pll(i, j, k, p) = pl(i, j, k-1, p);
            }
        );
        loc = Loci::face3;
        break;
#if DEBUG
    // TODO Throw an error
#endif
    }
    FLAG("Left offset");

    //  LOOP FUSION BABY
    pmb->par_for("uber_flux", pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_3D {
                FourVectors Dtmp;
                Real cmaxL, cmaxR, cminL, cminR;
                Real cmin, cmax;

                Real fluxL[8], fluxR[8];
                Real Ul[8], Ur[8];

                get_state(G, pll, i, j, k, loc, Dtmp);

                prim_to_flux(G, pll, Dtmp, eos, i, j, k, loc, 0, Ul); // dir==0 -> U instead of F in direction
                prim_to_flux(G, pll, Dtmp, eos, i, j, k, loc, dir, fluxL);

                mhd_vchar(G, pll, Dtmp, eos, i, j, k, loc, dir, cmaxL, cminL);

                get_state(G, pr, i, j, k, loc, Dtmp);

                prim_to_flux(G, pr, Dtmp, eos, i, j, k, loc, 0, Ur);
                prim_to_flux(G, pr, Dtmp, eos, i, j, k, loc, dir, fluxR);

                mhd_vchar(G, pr, Dtmp, eos, i, j, k, loc, dir, cmaxR, cminR);

                cmax = fabs(max(max(0., cmaxL), cmaxR)); // TODO suspicious use of abs()
                cmin = fabs(max(max(0., -cminL), -cminR));
                ctop(i, j, k, dir) = max(cmax, cmin);

                for (int p=0; p < NPRIM; p++)
                    flux(i, j, k, p) = 0.5 * (fluxL[p] + fluxR[p] - ctop(i, j, k, dir) * (Ur[p] - Ul[p]));
            }
    );
    FLAG("Uber fluxcalc");
}

void FluxCT(Container<Real>& rc)
{
    MeshBlock *pmb = rc.pmy_block;
    int is = pmb->is, js = pmb->js, ks = pmb->ks;
    int ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    int n1 = pmb->ncells1, n2 = pmb->ncells2, n3 = pmb->ncells3;
    ParArrayND<Real> emf1("emf1", n1, n2, n3);
    ParArrayND<Real> emf2("emf2", n1, n2, n3);
    ParArrayND<Real> emf3("emf3", n1, n2, n3);

    ParArrayND<Real> P = rc.Get("c.c.bulk.prims").data;
    ParArrayND<Real> F1 = rc.Get("c.c.bulk.prims").flux[0];
    ParArrayND<Real> F2 = rc.Get("c.c.bulk.prims").flux[1];
    ParArrayND<Real> F3 = rc.Get("c.c.bulk.prims").flux[2];

    pmb->par_for("flux_ct_emf", ks-2, ke+2, js-2, je+2, is-2, ie+2,
        KOKKOS_LAMBDA_3D {
            emf3(i, j, k) = 0.25 * (F1(i, j, k, prims::B2) + F1(i, j-1, k, prims::B2) - F2(i, j, k, prims::B1) - F2(i-1, j, k, prims::B1));
            emf2(i, j, k) = -0.25 * (F1(i, j, k, prims::B3) + F1(i, j, k-1, prims::B3) - F3(i, j, k, prims::B1) - F3(i-1, j, k, prims::B1));
            emf1(i, j, k) = 0.25 * (F2(i, j, k, prims::B3) + F2(i, j, k-1, prims::B3) - F3(i, j, k, prims::B2) - F3(i, j-1, k, prims::B2));
        });

    // Rewrite EMFs as fluxes, after Toth
    pmb->par_for("flux_ct_F1", ks-1, ke+1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA_3D {
            F1(i, j, k, prims::B1) = 0.;
            F1(i, j, k, prims::B2) =  0.5 * (emf3(i, j, k) + emf3(i, j+1, k));
            F1(i, j, k, prims::B3) = -0.5 * (emf2(i, j, k) + emf2(i, j, k+1));
        });
    pmb->par_for("flux_ct_F2", ks-1, ke+1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA_3D {
            F2(i, j, k, prims::B1) = -0.5 * (emf3(i, j, k) + emf3(i+1, j, k));
            F2(i, j, k, prims::B2) = 0.;
            F2(i, j, k, prims::B3) =  0.5 * (emf1(i, j, k) + emf1(i, j, k+1));
        });
    pmb->par_for("flux_ct_F3", ks-1, ke+1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA_3D {
            F3(i, j, k, prims::B1) =  0.5 * (emf2(i, j, k) + emf2(i+1, j, k));
            F3(i, j, k, prims::B2) = -0.5 * (emf1(i, j, k) + emf1(i, j+1, k));
            F3(i, j, k, prims::B3) = 0.;
        });
}
