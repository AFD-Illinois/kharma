/**
 * Coordinates applying a time evolution to the fluid
 */

#include "decs.hpp"
#include "mpi.hpp"
#include "fluxes.hpp"

#define FLAG(x) cout << x << endl;

using namespace std;

// Declarations
double advance_fluid(const Grid &G, const EOS eos,
                    const GridVars Pi, const GridVars Ps, GridVars Pf,
                    const double dt, GridInt pflag);

/**
 * Take one step.  Returns the Courant limit, to be used for the next step
 */
double step(const Grid &G, const EOS eos, const GridVars vars, const double dt)
{
    // Don't re-allocate scratch space per step
    // TODO be more civilised about this
    // TODO save a version when we have to calculate current
    GridVars vars_tmp("vars_tmp", G.gn1, G.gn2, G.gn3, G.nvar);
    GridInt pflag("pflag", G.gn1, G.gn2, G.gn3);

    // Predictor step
    advance_fluid(G, eos, vars, vars, vars_tmp, 0.5 * dt, pflag);
    FLAG("Advance Fluid Tmp");

    // Fixup routines: smooth over outlier zones
    // fixup(G, vars_tmp);
    // FLAG("Fixup Tmp");
    // Need an MPI call _before_ fixup_utop to obtain correct pflags
    //set_bounds(G, Stmp); // TODO flags only. Only if necessary?
    //FLAG("First bounds Tmp");
    // fixup_utoprim(G, vars_tmp);
    // FLAG("Fixup U_to_P Tmp");
    // set_bounds(G, vars_tmp);
    // FLAG("Second bounds Tmp");

    // Corrector step
    double ndt = advance_fluid(G, eos, vars, vars_tmp, vars, dt, pflag);
    FLAG("Advance Fluid Full");

    // fixup(G, S);
    // FLAG("Fixup Full");
    //set_bounds(G, S);
    //FLAG("First bounds Full");
    // fixup_utoprim(G, S);
    // FLAG("Fixup U_to_P Full");
    // set_bounds(G, S);
    // FLAG("Second bounds Full");

    // TODO take these from parameters...
    return std::min(0.9*ndt, 1.3*dt);
}

double advance_fluid(const Grid &G, const EOS eos,
                    const GridVars Pi, const GridVars Ps, GridVars Pf,
                    const double dt, GridInt pflag)
{
    GridVars Ui("Ui", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars Uf("Uf", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars dU("dU", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars F1("F1", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars F2("F2", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars F3("F3", G.gn1, G.gn2, G.gn3, G.nvar);
    GridDerived Dtmp;
    Dtmp.ucon = GridVector("Dtmp_ucon", G.gn1, G.gn2, G.gn3);
    Dtmp.ucov = GridVector("Dtmp_ucov", G.gn1, G.gn2, G.gn3);
    Dtmp.bcon = GridVector("Dtmp_bcon", G.gn1, G.gn2, G.gn3);
    Dtmp.bcov = GridVector("Dtmp_bcov", G.gn1, G.gn2, G.gn3);

    double ndt = get_flux(G, eos, Ps, F1, F2, F3);

    // TODO to add after fixup
// #if METRIC == MKS
//     fix_flux(F1, F2, F3);
// #endif

    //Constrained transport for B
    flux_ct(G, F1, F2, F3);

    // Update Si to Sf
    // TODO get_state_vec equivalent, just pass a slice
    Kokkos::parallel_for("get_dU", G.bulk_ng(),
        KOKKOS_LAMBDA (const int i, const int j, const int k) {
            get_state(G, Ps, i, j, k, Loci::center, Dtmp);
        }
    );
    get_fluid_source(G, Ps, Dtmp, eos, dU);


    Kokkos::parallel_for("get_Ui", G.bulk_ng(),
        KOKKOS_LAMBDA (const int i, const int j, const int k) {
            get_state(G, Pi, i, j, k, Loci::center, Dtmp);
            prim_to_flux(G, Pi, Dtmp, eos, i, j, k, Loci::center, 0, Ui);
        }
    );

    Kokkos::parallel_for("finite_diff", G.bulk_ng_p(),
        KOKKOS_LAMBDA (const int i, const int j, const int k, const int p) {
                Uf(i, j, k, p) = Ui(i, j, k, p) +
                                    dt * ((F1(i, j, k, p) - F1(i+1, j, k, p)) / G.dx1 +
                                        (F2(i, j, k, p) - F2(i, j+1, k, p)) / G.dx2 +
                                        (F3(i, j, k, p) - F3(i, j, k+1, p)) / G.dx3 +
                                        dU(i, j, k, p));
        }
    );

    // If the pointers are different, initialize final state to initial state
    // This seeds the U_to_P iteration
    if (Pi != Pf) {
        Kokkos::parallel_for("memcpy", G.all_0_p(),
            KOKKOS_LAMBDA (const int i, const int j, const int k, const int p) {
                Pf(i,j,k,p) = Pi(i,j,k,p);
            }
        );
    }

    // Finally, recover the primitives at the end of the substep
    // Kokkos::parallel_for("cons_to_prim", G.bulk_ng(),
    //     KOKKOS_LAMBDA (const int i, const int j, const int k) {
    //         // TODO ever called on not the center? Maybe w/face fields?
    //         pflag(i, j, k) = cons_to_prim(G, Uf, Pf, i, j, k, Loci::center);
    //     }
    // );

    return ndt;
}
