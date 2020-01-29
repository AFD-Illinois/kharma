/**
 * Coordinates applying a time evolution to the fluid
 */

#include "decs.hpp"
#include "mpi.hpp"
#include "fluxes.hpp"

#define FLAG(x) cout << x << endl;

using namespace std;

// Declarations
double advance_fluid(const Grid &G, const GridVars vars_in, const GridVars vars_mid,
                        GridVars vars_out, const double dt);

/**
 * Take one step.  Returns the Courant limit, to be used for the next step
 */
double step(const Grid &G, const GridVars vars, const double dt)
{
    // Don't re-allocate scratch space per step
    // TODO be more civilised about this
    // TODO save a version when we have to calculate current
    static GridVars vars_tmp("vars_tmp", G.gn1, G.gn2, G.gn3, G.nvar);

    // Predictor step
    advance_fluid(G, vars, vars, vars_tmp, 0.5 * dt);
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
    double ndt = advance_fluid(G, vars, vars_tmp, vars, dt);
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

double advance_fluid(const Grid &G, const GridVars vars_in, const GridVars vars_mid, GridVars vars_out, const double dt)
{
    static GridVars dU("dU", G.gn1, G.gn2, G.gn3, G.nvar);
    static GridVars F1("F1", G.gn1, G.gn2, G.gn3, G.nvar);
    static GridVars F2("F2", G.gn1, G.gn2, G.gn3, G.nvar);
    static GridVars F3("F3", G.gn1, G.gn2, G.gn3, G.nvar);

    // If the pointers are different, initialize final state to initial state
    // TODO pretty sure this is unnecessary?
    if (vars_in != vars_out) {
        int np = G.nvar;
        Kokkos::parallel_for("memcpy", G.all_0(),
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                for (int p=0; p < np; ++p)
                    vars_out(i,j,k,p) = vars_in(i,j,k,p);
            }

        );
    }

    double ndt = get_flux(G, vars_mid, F1, F2, F3);

    // TODO next thing to add
// #if METRIC == MKS
//     fix_flux(F);
// #endif

    //Constrained transport for B
    flux_ct(F1, F2, F3);

    // Update Si to Sf
    get_state(G, vars_mid, Loci::center, 0);
    get_fluid_source(G, vars_mid, dU);

    if (vars_in != vars_out) {
        get_state(G, vars_in, Loci::center, 0);
        prim_to_flux(G, vars_in, 0, Loci::center, 0, Ui);
    }

    Kokkos::parallel_for("finite_diff", G.bulk_ng(),
        KOKKOS_LAMBDA (const int i, const int j, const int k) {
            for (int p=0; p < np; ++p)
                Uf(i, j, k, p) = Ui(i, j, k, p) +
                                    Dt * ((F1(i, j, k, p) - F1(i+1, j, k, p)) / G.dx1 +
                                        (F2(i, j, k, p) - F2(i, j+1, k, p)) / G.dx2 +
                                        (F3(i, j, k, p) - F3(i, j, k+1, p)) / G.dx3 +
                                        dU(i, j, k, p));
        }

#pragma omp parallel for collapse(3)
    ZLOOP
    {
        pflag[k][j][i] = cons_to_prim(G, Sf, i, j, k, Loci::center); // TODO is this worth having anywhere else? ...
    }

#pragma omp parallel for simd collapse(2)
    ZLOOPALL
    {
        fail_save[k][j][i] = pflag[k][j][i];
    }

    return ndt;
}
