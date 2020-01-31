/**
 * Calculate fluxes through a zone
 */

#include "decs.hpp"
#include "reconstruction.hpp"
#include "phys.hpp"

#include <chrono>

void lr_to_flux(const Grid &G, const EOS eos, const GridVars Pr, const GridVars Pl,
                const int dir, const Loci loc, GridVars flux, GridVector ctop);
double ndt_min(const Grid &G, GridVector ctop);

/**
 * Returns the maximum (ironically) possible timestep, by evaluating
 * the Courant condition in the entire domain and taking the minimum
 */
double ndt_min(const Grid &G, GridVector ctop)
{
    double dt_min;
    Kokkos::Min<double> min_reducer(dt_min);
    Kokkos::parallel_reduce("ndt_min", G.bulk_ng(),
        KOKKOS_LAMBDA (const int &i, const int &j, const int &k, double &local_min) {
            double ndt_zone = 1 / (1 / (G.dx1 / ctop(i, j, k, 1)) +
                                 1 / (G.dx2 / ctop(i, j, k, 2)) +
                                 1 / (G.dx3 / ctop(i, j, k, 3)));
            if (ndt_zone < local_min) local_min = ndt_zone;
        }
    , min_reducer);

    // TODO MPI, record zone of minimum

  return dt_min;
}

double get_flux(const Grid &G, const EOS eos, const GridVars P, GridVars F1, GridVars F2, GridVars F3)
{
    GridVars Pl("Pl", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars Pr("Pr", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVector ctop("ctop", G.gn1, G.gn2, G.gn3);
    FLAG("Allocate recon temporaries");

    // Reconstruct primitives at left and right sides of faces, then find conserved variables
    reconstruct(G, P, Pl, Pr, 1);
    FLAG("Recon 1");
    lr_to_flux(G, eos, Pl, Pr, 1, Loci::face1, F1, ctop);
    FLAG("LR 1");

    reconstruct(G, P, Pl, Pr, 2);
    FLAG("Recon 2");
    lr_to_flux(G, eos, Pl, Pr, 2, Loci::face2, F2, ctop);
    FLAG("LR 2");

    reconstruct(G, P, Pl, Pr, 3);
    FLAG("Recon 3");
    lr_to_flux(G, eos, Pl, Pr, 3, Loci::face3, F3, ctop);
    FLAG("LR 3");

    return ndt_min(G, ctop);
}

// Note that these are the primitives at the left and right of the *interface*
void lr_to_flux(const Grid &G, const EOS eos, const GridVars Pr, const GridVars Pl,
                const int dir, const Loci loc, GridVars flux, GridVector ctop)
{
    GridVars fluxL("fluxL", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars fluxR("fluxR", G.gn1, G.gn2, G.gn3, G.nvar);
    GridScalar cmaxL("cmaxL", G.gn1, G.gn2, G.gn3), cmaxR("cmaxR", G.gn1, G.gn2, G.gn3);
    GridScalar cminL("cminL", G.gn1, G.gn2, G.gn3), cminR("cminR", G.gn1, G.gn2, G.gn3);
    // TODO constructors
    GridDerived Dr;
    //Dl.ucon = GridVector("Dl_ucon", G.gn1, G.gn2, G.gn3);
    //Dl.ucov = GridVector("Dl_ucov", G.gn1, G.gn2, G.gn3);
    //Dl.bcon = GridVector("Dl_bcon", G.gn1, G.gn2, G.gn3);
    //Dl.bcov = GridVector("Dl_bcov", G.gn1, G.gn2, G.gn3);
    Dr.ucon = GridVector("Dr_ucon", G.gn1, G.gn2, G.gn3);
    Dr.ucov = GridVector("Dr_ucov", G.gn1, G.gn2, G.gn3);
    Dr.bcon = GridVector("Dr_bcon", G.gn1, G.gn2, G.gn3);
    Dr.bcov = GridVector("Dr_bcov", G.gn1, G.gn2, G.gn3);

    GridVars Ul("Ul", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars Ur("Ur", G.gn1, G.gn2, G.gn3, G.nvar);

    // Offset "left" variables by one zone to line up L- and R-fluxes at *faces*
    // These are un-macro'd to bundle OpenMP thread tasks rather than memory accesses
    GridVars Pll("Pll", G.gn1, G.gn2, G.gn3, G.nvar);
    if (dir == 1) {
        Kokkos::parallel_for("offset_left_1", G.bulk_plus_p(1),
            KOKKOS_LAMBDA_VARS {
                Pll(i, j, k, p) = Pl(i-1, j, k, p);
            }
        );
    } else if (dir == 2) {
        Kokkos::parallel_for("offset_left_2", G.bulk_plus_p(1),
            KOKKOS_LAMBDA_VARS {
                Pll(i, j, k, p) = Pl(i, j-1, k, p);
            }
        );
    } else if (dir == 3) {
        Kokkos::parallel_for("offset_left_3", G.bulk_plus_p(1),
            KOKKOS_LAMBDA_VARS {
                Pll(i, j, k, p) = Pl(i, j, k-1, p);
            }
        );
    }
    FLAG("Left");

    auto start_left = std::chrono::high_resolution_clock::now();

    //  ALL THIS IS BULK+1 (or better?)
    // TODO this is almost certainly too much for a single loop. Split and see
    Kokkos::parallel_for("vchar_l", G.bulk_plus(1),
            KOKKOS_LAMBDA_3D {
                Derived Dl;
                get_state(G, Pll, i, j, k, loc, Dl);

                prim_to_flux(G, Pll, Dl, eos, i, j, k, loc, 0, Ul); // dir==0 -> U instead of F in direction
                prim_to_flux(G, Pll, Dl, eos, i, j, k, loc, dir, fluxL);

                mhd_vchar(G, Pll, Dl, eos, i, j, k, loc, dir, cmaxL, cminL);
            }
    );
    auto end_left = TIME_NOW;
    FLAG("vchar_l");

    auto start_right = TIME_NOW;
    Kokkos::parallel_for("state_r", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            get_state(G, Pr, i, j, k, loc, Dr);
        }
    );

    auto start_ptof = TIME_NOW;
    Kokkos::parallel_for("ptof_r", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            prim_to_flux(G, Pr, Dr, eos, i, j, k, loc, 0, Ur);
            prim_to_flux(G, Pr, Dr, eos, i, j, k, loc, dir, fluxR);
        }
    );

    auto start_vchar = TIME_NOW;
    Kokkos::parallel_for("vchar_r", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            mhd_vchar(G, Pr, Dr, eos, i, j, k, loc, dir, cmaxR, cminR);
        }
    );
    auto end_right = TIME_NOW;
    FLAG("vchar_r");

#if DEBUG
    cerr << "Left: " << PRINT_SEC(end_left - start_left) << " Right: " << PRINT_SEC(end_right - start_right) << endl;
    cerr << "Vchar: " << PRINT_SEC(end_right - start_vchar);
    cerr << " Ptof: " << PRINT_SEC(start_vchar - start_ptof);
    cerr << " State: " << PRINT_SEC(start_ptof - start_right) << endl;
#endif

    Kokkos::parallel_for("ctop", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            // TODO this seems wrong. What should the abs value policy actually be?
            Real cmax = fabs(max(max(0., cmaxL(i, j, k)), cmaxR(i, j, k)));
            Real cmin = fabs(max(max(0., -cminL(i, j, k)), -cminR(i, j, k)));
            ctop(i, j, k, dir) = max(cmax, cmin);
        }
    );
    FLAG("ctop");

#if DEBUG
    // TODO consistent zone_loc tools
    int nan_count;
    Kokkos::parallel_reduce("any_nan", G.bulk_plus(1),
        KOKKOS_LAMBDA (const int& i, const int& j, const int& k, int& nan_count_local) {
            nan_count_local += isnan(1. / ctop(i, j, k, dir));
        }
    , nan_count);
    if(nan_count > 0) {
        throw std::runtime_error("Ctop 0 or NaN, cannot continue");
    }
    FLAG("any_nan");
#endif

    Kokkos::parallel_for("flux", G.bulk_plus_p(1),
        KOKKOS_LAMBDA_VARS {
            flux(i, j, k, p) = 0.5 * (fluxL(i, j, k, p) + fluxR(i, j, k, p) -
                                        ctop(i, j, k, dir) * (Ur(i, j, k, p) - Ul(i, j, k, p)));
        }
    );
    FLAG("Flux_calc");
}

void flux_ct(const Grid G, GridVars F1, GridVars F2, GridVars F3)
{
    // TODO I pay an extra 30% memory for adding ghosts, and 25% in places for 4-vectors where 3 will do
    // is that enough to slow us down much?
    GridScalar emf1("emf1", G.gn1, G.gn2, G.gn3);
    GridScalar emf2("emf2", G.gn1, G.gn2, G.gn3);
    GridScalar emf3("emf3", G.gn1, G.gn2, G.gn3);

    Kokkos::parallel_for("flux_ct_emf", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            emf3(i, j, k) = 0.25 * (F1(i, j, k, prims::B2) + F1(i, j-1, k, prims::B2) - F2(i, j, k, prims::B1) - F2(i-1, j, k, prims::B1));
            emf2(i, j, k) = -0.25 * (F1(i, j, k, prims::B3) + F1(i, j, k-1, prims::B3) - F3(i, j, k, prims::B1) - F3(i-1, j, k, prims::B1));
            emf1(i, j, k) = 0.25 * (F2(i, j, k, prims::B3) + F2(i, j, k-1, prims::B3) - F3(i, j, k, prims::B2) - F3(i, j-1, k, prims::B2));
        });

        // Rewrite EMFs as fluxes, after Toth
    Kokkos::parallel_for("flux_ct_F1", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            F1(i, j, k, prims::B1) = 0.;
            F1(i, j, k, prims::B2) =  0.5 * (emf3(i, j, k) + emf3(i, j+1, k));
            F1(i, j, k, prims::B3) = -0.5 * (emf2(i, j, k) + emf2(i, j, k+1));
        });
    Kokkos::parallel_for("flux_ct_F2", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            F2(i, j, k, prims::B1) = -0.5 * (emf3(i, j, k) + emf3(i+1, j, k));
            F2(i, j, k, prims::B2) = 0.;
            F2(i, j, k, prims::B3) =  0.5 * (emf1(i, j, k) + emf1(i, j, k+1));
        });
    Kokkos::parallel_for("flux_ct_F3", G.bulk_plus(1),
        KOKKOS_LAMBDA_3D {
            F3(i, j, k, prims::B1) =  0.5 * (emf2(i, j, k) + emf2(i+1, j, k));
            F3(i, j, k, prims::B2) = -0.5 * (emf1(i, j, k) + emf1(i, j+1, k));
            F3(i, j, k, prims::B3) = 0.;
        });
}