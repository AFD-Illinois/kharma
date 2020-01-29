/**
 * Calculate fluxes through a zone
 */

#include "decs.hpp"
#include "reconstruction.hpp"
#include "phys.hpp"

void lr_to_flux(const Grid &G, const GridVars Pr, const GridVars Pl,
                const int dir, const Loci loc, GridVars flux, GridVector ctop);
double ndt_min(const Grid &G, GridVector ctop);

/**
 * Returns the maximum (ironically) possible timestep, by evaluating
 * the Courant condition in the entire domain and taking the minimum
 */
Real ndt_min(const Grid &G, GridVector ctop)
{
    Real dt_min;
    Kokkos::parallel_reduce("ndt_min", G.bulk_ng(),
        KOKKOS_LAMBDA (const int i, const int j, const int k, Real local_min) {
            Real ndt_zone = 1 / (1 / (G.dx1 / ctop(i, j, k, 1)) +
                                 1 / (G.dx2 / ctop(i, j, k, 2)) +
                                 1 / (G.dx3 / ctop(i, j, k, 3)));
            if (ndt_zone < local_min) local_min = ndt_zone;
        }
    , Kokkos::Min<Real>(dt_min));

    // TODO MPI, record zone of minimum

  return dt_min;
}

double get_flux(const Grid &G, const GridVars P, GridVars F1, GridVars F2, GridVars F3)
{
    static GridVars Pl("Pl", G.gn1, G.gn2, G.gn3, G.nvar);
    static GridVars Pr("Pr", G.gn1, G.gn2, G.gn3, G.nvar);
    static GridVector ctop("ctop", G.gn1, G.gn2, G.gn3);

    // Reconstruct primitives at left and right sides of faces, then find conserved variables
    reconstruct(G, P, Pl, Pr, 1);
    lr_to_flux(G, Pl, Pr, 1, Loci::face1, F1, ctop);

    reconstruct(G, P, Pl, Pr, 2);
    lr_to_flux(G, Pl, Pr, 2, Loci::face2, F2, ctop);

    reconstruct(G, P, Pl, Pr, 3);
    lr_to_flux(G, Pl, Pr, 3, Loci::face3, F3, ctop);

    return ndt_min(G, ctop);
}

// Note that the sense of L/R flips from zone to interface during function call
void lr_to_flux(const Grid &G, const GridVars Pr, const GridVars Pl,
                const int dir, const Loci loc, GridVars flux, GridVector ctop)
{
    static GridVars fluxL("fluxL", G.gn1, G.gn2, G.gn3, G.nvar);
    static GridVars fluxR("fluxR", G.gn1, G.gn2, G.gn3, G.nvar);
    static GridScalar cmaxL("cmaxL", G.gn1, G.gn2, G.gn3), cmaxR("cmaxR", G.gn1, G.gn2, G.gn3);
    static GridScalar cminL("cminL", G.gn1, G.gn2, G.gn3), cminR("cminR", G.gn1, G.gn2, G.gn3);
    static GridScalar cmax("cmax", G.gn1, G.gn2, G.gn3), cmin("cmin", G.gn1, G.gn2, G.gn3);

    // Offset "left" variables by one zone to line up L- and R-fluxes at *faces*
    // These are un-macro'd to bundle OpenMP thread tasks rather than memory accesses
    static GridVars Pll("Pll", G.gn1, G.gn2, G.gn3, G.nvar);
    int np = G.nvar;
    if (dir == 1) {
        Kokkos::parallel_for("offset_left_1", G.bulk_plus(1),
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                for(int p=0; p<np; ++p) Pll(i, j, k, p) = Pl(i-1, j, k, p);
            }
        );
    } else if (dir == 2) {
        Kokkos::parallel_for("offset_left_2", G.bulk_plus(1),
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                for(int p=0; p<np; ++p) Pll(i, j, k, p) = Pl(i, j-1, k, p);
            }
        );
    } else if (dir == 3) {
        Kokkos::parallel_for("offset_left_3", G.bulk_plus(1),
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                for(int p=0; p<np; ++p) Pll(i, j, k, p) = Pl(i, j, k-1, p);
            }
        );
    }

    //  ALL THIS IS BULK+1 (or better?)
    Kokkos::parallel_for("vchar_l", G.bulk_plus(1),
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                get_state(G, Pll, D, i, j, k, loc);

                prim_to_flux(G, Pll, D, i, j, k, loc, 0, Ul);
                prim_to_flux(G, Pll, D, i, j, k, loc, dir, fluxL);

                mhd_vchar(G, Pll, i, j, k, loc, dir, cmaxL, cminL);
            }
    );

    Kokkos::parallel_for("vchar_r", G.bulk_plus(1),
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                get_state(G, Pr, i, j, k, loc, D);

                prim_to_flux(G, Pr, D, i, j, k, loc, 0, Ur); // dir==0 -> U instead of F in direction
                prim_to_flux(G, Pr, D, i, j, k, loc, dir, fluxR);

                mhd_vchar(G, Pr, i, j, k, loc, dir, cmaxR, cminR);
            }
    );


    Kokkos::parallel_for("ctop", G.bulk_plus(1),
        KOKKOS_LAMBDA (const int i, const int j, const int k)
        {
            // TODO this seems wrong. What should the abs value policy actually be?
            Real cmax = fabs(std::max(std::max(0., cmaxL(i, j, k)), cmaxR(i, j, k)));
            Real cmin = fabs(std::max(std::max(0., -cminL(i, j, k)), -cminR(i, j, k)));
            ctop(i, j, k, dir) = std::max(cmax, cmin);
#if DEBUG
            if (isnan(1. / ctop(i, j, k, dir)))
            {
                std::cerr << format_string("ctop is 0 or NaN at zone: %i %i %i (%i) ", i, j, k, dir) << std::endl;
                // double X[NDIM];
                // double r, th;
                // coord(i, j, k, CENT, X);
                // bl_coord(X, &r, &th);
                // printf("(r,th,phi = %f %f %f)\n", r, th, X[3]);
                throw std::runtime_error("Ctop 0 or NaN, cannot continue")
            }
#endif
        }
    );

    Kokkos::parallel_for("flux", G.bulk_plus(1),
        KOKKOS_LAMBDA (const int i, const int j, const int k)
        {
            for (int p=0; p<np; ++p)
                flux(i, j, k, p) = 0.5 * (fluxL(i, j, k, p) + fluxR(i, j, k, p) -
                                          ctop(i, j, k, dir) * (Ur(i, j, k, p) - Ul(i, j, k, p)));
        }
    );
}

void flux_ct(GridVars F1, GridVars F2, GridVars F3)
{
    // TODO I pay an extra 30% memory for adding ghosts, and 25% in places for 4-vectors where 3 will do
    // is that enough to Plow us down much?
    static int firstc = 1;
    if (firstc)
    {
        emf1 = new GridScalar(G.gn1, G.gn2, G.gn3);
        emf2 = new GridScalar(G.gn1, G.gn2, G.gn3);
        emf3 = new GridScalar(G.gn1, G.gn2, G.gn3);

        firstc = 0;
    }

    Kokkos::parallel_for("flux_ct_emf", G.bulk_plus(1),
        KOKKOS_LAMBDA (const int i, const int j, const int k)
        {
            emf3(i, j, k) = 0.25 * (F1(i, j, k, prims::B2) + F1(i, j-1, k, prims::B2) - F2(i, j, k, prims::B1) - F2(i-1, j, k, prims::B1));
            emf2(i, j, k) = -0.25 * (F1(i, j, k, prims::B3) + F1(i, j, k-1, prims::B3) - F3(i, j, k, prims::B1) - F3(i-1, j, k, prims::B1));
            emf1(i, j, k) = 0.25 * (F2(i, j, k, prims::B3) + F2(i, j, k-1, prims::B3) - F3(i, j, k, prims::B2) - F3(i, j-1, k, prims::B2));
        });

        // Rewrite EMFs as fluxes, after Toth
    Kokkos::parallel_for("flux_ct_F1", G.bulk_plus(1),
        KOKKOS_LAMBDA (const int i, const int j, const int k)
        {
            F1(i, j, k, prims::B1) = 0.;
            F1(i, j, k, prims::B2) = 0.5 * (emf3(i, j, k) + emf3[k][j + 1][i]);
            F1(i, j, k, prims::B3) = -0.5 * (emf2(i, j, k) + emf2[k + 1][j][i]);
        });
    Kokkos::parallel_for("flux_ct_F2", G.bulk_plus(1),
        KOKKOS_LAMBDA (const int i, const int j, const int k)
        {
            F2(i, j, k, prims::B1) = -0.5 * (emf3(i, j, k) + emf3[k][j][i + 1]);
            F2(i, j, k, prims::B2) = 0.;
            F2(i, j, k, prims::B3) = 0.5 * (emf1(i, j, k) + emf1[k + 1][j][i]);
        });
    Kokkos::parallel_for("flux_ct_F3", G.bulk_plus(1),
        KOKKOS_LAMBDA (const int i, const int j, const int k)
        {
            F3(i, j, k, prims::B1) = 0.5 * (emf2(i, j, k) + emf2[k][j][i + 1]);
            F3(i, j, k, prims::B2) = -0.5 * (emf1(i, j, k) + emf1[k][j + 1][i]);
            F3(i, j, k, prims::B3) = 0.;
        });
}
