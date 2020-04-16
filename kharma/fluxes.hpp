/**
 * Calculate fluxes through a zone
 */

#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"

#include "decs.hpp"

#include "reconstruction.hpp"
#include "phys.hpp"
#include "debug.hpp"

void lr_to_flux(const Grid &G, const EOS* eos, const CellVariable<Real> Pr, const CellVariable<Real> Pl,
                const int dir, const Loci loc, GridVars flux, ParArray3D<Real> ctop);

using namespace parthenon;

TaskStatus CalculateFluxes(Container<Real>& rc)
{
    MeshBlock *pmb = rc.pmy_block;
    ParArrayND<Real> pl("pl", pmb->ncells1, pmb->ncells2, pmb->ncells3, NPRIM);
    ParArrayND<Real> pr("pr", pmb->ncells1, pmb->ncells2, pmb->ncells3, NPRIM);

    // Reconstruct primitives at left and right sides of faces
    WENO5X1(pmb, pl, pr);
    FLAG("Recon 1");
    // Calculate flux from values at left & right of face, give to MeshBlock
    LRToFlux(pr, pl, 1, pmb);
    FLAG("LR 1");

    WENO5X2(pmb, pl, pr);
    FLAG("Recon 2");
    LRToFlux(pr, pl, 2, pmb);
    FLAG("LR 2");

    WENO5X3(pmb, pl, pr);
    FLAG("Recon 3");
    LRToFlux(pr, pl, 3, pmb);
    FLAG("LR 3");

    return TaskStatus::complete;
}

// Note that these are the primitives at the left and right of the *interface*
void LRToFlux(ParArrayND<Real> pl, ParArrayND<Real> pr, const int dir, MeshBlock *pmb)
{
    ParArrayND<Real> pll("pl", pmb->ncells1, pmb->ncells2, pmb->ncells3);

    // Offset "left" variables by one zone to line up L- and R-fluxes at *faces*
    // TODO can this be done without a copy/temporary?
    if (dir == 1) {
        pmb->par_for("offset_left_1", pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_VARS {
                pll(i, j, k, p) = pl(i-1, j, k, p);
            }
        );
    } else if (dir == 2) {
        pmb->par_for("offset_left_2", pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_VARS {
                pll(i, j, k, p) = pl(i, j-1, k, p);
            }
        );
    } else if (dir == 3) {
        pmb->par_for("offset_left_3", pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_VARS {
                pll(i, j, k, p) = pl(i, j, k-1, p);
            }
        );
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

void flux_ct(MeshBlock *pmb)
{
    pmb->par_for("flux_ct_emf", G.bulk_plus(2),
        KOKKOS_LAMBDA_3D {
            emf3(i, j, k) = 0.25 * (F1(i, j, k, prims::B2) + F1(i, j-1, k, prims::B2) - F2(i, j, k, prims::B1) - F2(i-1, j, k, prims::B1));
            emf2(i, j, k) = -0.25 * (F1(i, j, k, prims::B3) + F1(i, j, k-1, prims::B3) - F3(i, j, k, prims::B1) - F3(i-1, j, k, prims::B1));
            emf1(i, j, k) = 0.25 * (F2(i, j, k, prims::B3) + F2(i, j, k-1, prims::B3) - F3(i, j, k, prims::B2) - F3(i, j-1, k, prims::B2));
        });

        // Rewrite EMFs as fluxes, after Toth
    pmb->par_for("flux_ct_F1", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            F1(i, j, k, prims::B1) = 0.;
            F1(i, j, k, prims::B2) =  0.5 * (emf3(i, j, k) + emf3(i, j+1, k));
            F1(i, j, k, prims::B3) = -0.5 * (emf2(i, j, k) + emf2(i, j, k+1));
        });
    pmb->par_for("flux_ct_F2", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            F2(i, j, k, prims::B1) = -0.5 * (emf3(i, j, k) + emf3(i+1, j, k));
            F2(i, j, k, prims::B2) = 0.;
            F2(i, j, k, prims::B3) =  0.5 * (emf1(i, j, k) + emf1(i, j, k+1));
        });
    pmb->par_for("flux_ct_F3", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            F3(i, j, k, prims::B1) =  0.5 * (emf2(i, j, k) + emf2(i+1, j, k));
            F3(i, j, k, prims::B2) = -0.5 * (emf1(i, j, k) + emf1(i, j+1, k));
            F3(i, j, k, prims::B3) = 0.;
        });
}
