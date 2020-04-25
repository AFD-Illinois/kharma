/**
 * Calculate fluxes through a zone
 */
#pragma once

#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"

#include "decs.hpp"

#include "reconstruction.hpp"
#include "phys.hpp"

using namespace parthenon;

/**
 * Take reconstructed primitives at either side of a face and construct the local Lax-Friedrichs flux
 *
 * Also fills the "ctop" vector with the highest magnetosonic speed mhd_vchar -- used to estimate timestep later.
 *
 * Note that since this L and R are defined with respect to the *face*, they are actually the
 * opposite of the "r" and "l" in the caller, CalculateFluxes!
 */
void LRToFlux(Container<Real>& rc, GridVars pl, GridVars pr, const int dir, GridVars flux)
{
    FLAG("LR to flux");
    MeshBlock *pmb = rc.pmy_block;

    GridVars pll("pll", NPRIM, pmb->ncells3, pmb->ncells2, pmb->ncells1);

    auto& ctop = rc.GetFace("f.f.bulk.ctop").data;

    // So far we don't need fluxes that don't match faces
    Loci loc;

    // TODO *sigh*
    Grid G(pmb);
    Real gamma = pmb->packages["GRMHD"]->Param<Real>("gamma");
    EOS* eos = new GammaLaw(gamma);

    // Offset "left" variables by one zone to line up L- and R-fluxes at *faces*
    // TODO can this be done without a copy/temporary? Fewer ranks?
    switch (dir) {
    case 1:
        pmb->par_for("offset_left_1", 0, NPRIM-1, pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_VARS {
                pll(p, k, j, i) = pl(p, k, j, i-1);
            }
        );
        loc = Loci::face1;
        break;
    case 2:
        pmb->par_for("offset_left_2", 0, NPRIM-1, pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_VARS {
                pll(p, k, j, i) = pl(p, k, j-1, i);
            }
        );
        loc = Loci::face2;
        break;
    case 3:
        pmb->par_for("offset_left_3", 0, NPRIM-1, pmb->ks-1, pmb->ke+1, pmb->js-1, pmb->je+1, pmb->is-1, pmb->ie+1,
            KOKKOS_LAMBDA_VARS {
                pll(p, k, j, i) = pl(p, k-1, j, i);
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

                // Left
                get_state(G, pll, k, j, i, loc, Dtmp);

                prim_to_flux(G, pll, Dtmp, eos, k, j, i, loc, 0, Ul); // dir==0 -> U instead of F in direction
                prim_to_flux(G, pll, Dtmp, eos, k, j, i, loc, dir, fluxL);

                mhd_vchar(G, pll, Dtmp, eos, k, j, i, loc, dir, cmaxL, cminL);

                // Right
                get_state(G, pr, k, j, i, loc, Dtmp);

                prim_to_flux(G, pr, Dtmp, eos, k, j, i, loc, 0, Ur);
                prim_to_flux(G, pr, Dtmp, eos, k, j, i, loc, dir, fluxR);

                mhd_vchar(G, pr, Dtmp, eos, k, j, i, loc, dir, cmaxR, cminR);

                cmax = fabs(max(max(0.,  cmaxL),  cmaxR)); // TODO suspicious use of abs()
                cmin = fabs(max(max(0., -cminL), -cminR));
                ctop(dir, k, j, i) = max(cmax, cmin);

                PLOOP flux(p, k, j, i) = 0.5 * (fluxL[p] + fluxR[p] - ctop(dir, k, j, i) * (Ur[p] - Ul[p]));
            }
    );
    FLAG("Uber fluxcalc");
}

/**
 * Constrained transport.  Modify B-field fluxes to preserve divB==0 condition to machine precision per-step
 */
void FluxCT(Container<Real>& rc)
{
    FLAG("Flux CT");
    MeshBlock *pmb = rc.pmy_block;
    int is = pmb->is, js = pmb->js, ks = pmb->ks;
    int ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    int n1 = pmb->ncells1, n2 = pmb->ncells2, n3 = pmb->ncells3;
    GridScalar emf1("emf1", n3, n2, n1);
    GridScalar emf2("emf2", n3, n2, n1);
    GridScalar emf3("emf3", n3, n2, n1);
    GridVars F1 = rc.Get("c.c.bulk.cons").flux[0];
    GridVars F2 = rc.Get("c.c.bulk.cons").flux[1];
    GridVars F3 = rc.Get("c.c.bulk.cons").flux[2];

    pmb->par_for("flux_ct_emf", ks-2, ke+2, js-2, je+2, is-2, ie+2,
        KOKKOS_LAMBDA_3D {
            emf3(k, j, i) =  0.25 * (F1(prims::B2, k, j, i) + F1(prims::B2, k, j-1, i) - F2(prims::B1, k, j, i) - F2(prims::B1, k, j, i-1));
            emf2(k, j, i) = -0.25 * (F1(prims::B3, k, j, i) + F1(prims::B3, k-1, j, i) - F3(prims::B1, k, j, i) - F3(prims::B1, k, j, i-1));
            emf1(k, j, i) =  0.25 * (F2(prims::B3, k, j, i) + F2(prims::B3, k-1, j, i) - F3(prims::B2, k, j, i) - F3(prims::B2, k, j-1, i));
        }
    );

    // Rewrite EMFs as fluxes, after Toth
    // TODO worthwhile to split and only do +1 in relevant direction?
    pmb->par_for("flux_ct", ks-1, ke+1, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA_3D {
            F1(prims::B1, k, j, i) =  0.0;
            F1(prims::B2, k, j, i) =  0.5 * (emf3(k, j, i) + emf3(k, j+1, i));
            F1(prims::B3, k, j, i) = -0.5 * (emf2(k, j, i) + emf2(k+1, j, i));

            F2(prims::B1, k, j, i) = -0.5 * (emf3(k, j, i) + emf3(k, j, i+1));
            F2(prims::B2, k, j, i) =  0.0;
            F2(prims::B3, k, j, i) =  0.5 * (emf1(k, j, i) + emf1(k+1, j, i));

            F3(prims::B1, k, j, i) =  0.5 * (emf2(k, j, i) + emf2(k, j, i+1));
            F3(prims::B2, k, j, i) = -0.5 * (emf1(k, j, i) + emf1(k, j+1, i));
            F3(prims::B3, k, j, i) =  0.0;
        }
    );
}

double MaxDivB(Container<Real>& rc)
{
    FLAG("Calculating divB");
    MeshBlock *pmb = rc.pmy_block;
    auto is = pmb->is, js = pmb->js, ks = pmb->ks;
    auto ie = pmb->ie, je = pmb->je, ke = pmb->ke;
    auto dx1v = pmb->pcoord->dx1v;
    auto dx2v = pmb->pcoord->dx2v;
    auto dx3v = pmb->pcoord->dx3v;
    GridVars P = rc.Get("c.c.bulk.prims").data;

    Grid G(pmb);

    double max_divb;
    Kokkos::Max<double> max_reducer(max_divb);
    Kokkos::parallel_reduce("divB", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({is, js, ks}, {ie, je, ke}),
        KOKKOS_LAMBDA (const int &i, const int &j, const int &k, double &local_max_divb) {
            double local_divb = fabs(0.25*(
                            P(prims::B1, k, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B1, k, j-1, i) * G.gdet(Loci::center, j-1, i)
                            + P(prims::B1, k-1, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B1, k-1, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B1, k, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B1, k, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            - P(prims::B1, k-1, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B1, k-1, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            )/dx1v(i) +
                            0.25*(
                            P(prims::B2, k, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B2, k, j, i-1) * G.gdet(Loci::center, j, i-1)
                            + P(prims::B2, k-1, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B2, k-1, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B2, k, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B2, k, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            - P(prims::B2, k-1, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B2, k-1, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            )/dx2v(j) +
                            0.25*(
                            P(prims::B3, k, j, i) * G.gdet(Loci::center, j, i)
                            + P(prims::B3, k, j-1, i) * G.gdet(Loci::center, j-1, i)
                            + P(prims::B3, k, j, i-1) * G.gdet(Loci::center, j, i-1)
                            + P(prims::B3, k, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            - P(prims::B3, k-1, j, i) * G.gdet(Loci::center, j, i)
                            - P(prims::B3, k-1, j-1, i) * G.gdet(Loci::center, j-1, i)
                            - P(prims::B3, k-1, j, i-1) * G.gdet(Loci::center, j, i-1)
                            - P(prims::B3, k-1, j-1, i-1) * G.gdet(Loci::center, j-1, i-1)
                            )/dx3v(k));
            if (local_divb > local_max_divb) local_max_divb = local_divb;
        }
    , max_reducer);

    return max_divb;
}
