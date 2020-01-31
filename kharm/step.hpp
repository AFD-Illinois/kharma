/**
 * Coordinates applying a time evolution to the fluid
 */

#include "utils.hpp"
#include "decs.hpp"
#include "mpi.hpp"
#include "fluxes.hpp"
#include "U_to_P.hpp"
#include "source.hpp"

#include <chrono>

using namespace std;

// Declarations
double advance_fluid(const Grid &G, const EOS eos,
                    const GridVars Pi, const GridVars Ps, GridVars Pf,
                    const double dt, GridInt pflag);

/**
 * Take one step.  Returns the Courant limit, to be used for the next step
 */
double step(const Grid &G, const EOS eos, GridVars vars, const double dt)
{
    static double t;

    FLAG("Start step")
    // Don't re-allocate scratch space per step
    // TODO be more civilised about this
    // TODO save a copy of current state when we have to calculate j
    GridVars vars_tmp("vars_tmp", G.gn1, G.gn2, G.gn3, G.nvar);
    GridInt pflag("pflag", G.gn1, G.gn2, G.gn3);
    FLAG("Allocate temporaries");

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

    t += dt;

    #define ERR_NEG_INPUT -100
    #define ERR_MAX_ITER 1
    #define ERR_UTSQ 2
    #define ERR_GAMMA 3
    #define ERR_RHO_NEGATIVE 6
    #define ERR_U_NEGATIVE 7
    #define ERR_BOTH_NEGATIVE 8

    int pflags;
    Kokkos::parallel_reduce("ndt_min", G.bulk_ng(),
        KOKKOS_LAMBDA (const int &i, const int &j, const int &k, int &local_flags) {
            local_flags += (pflag(i, j, k) == ERR_MAX_ITER);
        }
    , pflags);

    cerr << string_format("t = %.5f, %d pflags, %d floors", t, pflags, 0 ) << endl;

    // TODO take these from parameters...
    return std::min(0.9*ndt, 1.3*dt);
}

double advance_fluid(const Grid &G, const EOS eos,
                    const GridVars Pi, const GridVars Ps, GridVars Pf,
                    const double dt, GridInt pflag)
{
    // GridVars Ui("Ui", G.gn1, G.gn2, G.gn3, G.nvar);
    // GridVars dU("dU", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars Uf("Uf", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars F1("F1", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars F2("F2", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars F3("F3", G.gn1, G.gn2, G.gn3, G.nvar);
    FLAG("Allocate flux temporaries");

    auto start_get_flux = TIME_NOW;
    // Get the fluxes in each direction on the zone faces
    double ndt = get_flux(G, eos, Ps, F1, F2, F3);
    FLAG("Get flux");

    // Fix boundary fluxes
//    fix_flux(F1, F2, F3);
//    FLAG("Fix flux");

    auto start_flux_ct = TIME_NOW;
    // Constrained transport for B
    flux_ct(G, F1, F2, F3);
    FLAG("Flux CT");

    // Update Si to Sf
    // TODO get_state_vec equivalent, just pass a slice
    // Kokkos::parallel_for("get_dU", G.bulk_ng(),
    //     KOKKOS_LAMBDA_3D {
    //         Derived Dtmp;
    //         get_state(G, Ps, i, j, k, Loci::center, Dtmp);
    //         get_fluid_source(G, Ps, Dtmp, eos, i, j, k, dU);
    //     }
    // );
    // FLAG("Get source");

    // Kokkos::parallel_for("get_Ui", G.bulk_ng(),
    //     KOKKOS_LAMBDA_3D {
    //         Derived Dtmp;
    //         get_state(G, Pi, i, j, k, Loci::center, Dtmp);
    //         prim_to_flux(G, Pi, Dtmp, eos, i, j, k, Loci::center, 0, Ui);
    //     }
    // );
    // FLAG("Get Ui");

    auto start_uberkernel = TIME_NOW;
    const int np = G.nvar;
    Kokkos::parallel_for("finite_diff", G.bulk_ng(),
        KOKKOS_LAMBDA_3D {
            Derived Dtmp;
            Real dU[8], Ui[8]; // TODO but what if we use >12 vars?
            get_state(G, Ps, i, j, k, Loci::center, Dtmp);
            get_fluid_source(G, Ps, Dtmp, eos, i, j, k, dU);

            if(Pi != Ps)
                get_state(G, Pi, i, j, k, Loci::center, Dtmp);
            prim_to_flux(G, Pi, Dtmp, eos, i, j, k, Loci::center, 0, Ui); // (i, j, k, p)

            for(int p=0; p < np; ++p)
                Uf(i, j, k, p) = Ui[p] +
                                    dt * ((F1(i, j, k, p) - F1(i+1, j, k, p)) / G.dx1 +
                                        (F2(i, j, k, p) - F2(i, j+1, k, p)) / G.dx2 +
                                        (F3(i, j, k, p) - F3(i, j, k+1, p)) / G.dx3 +
                                        dU[p]);
        }
    );
    FLAG("Finite diff");

    auto start_utop = TIME_NOW;
    // Finally, recover the primitives at the end of the substep
    Kokkos::parallel_for("cons_to_prim", G.bulk_ng(),
        KOKKOS_LAMBDA_3D {
            // TODO ever called on not the center? Maybe w/face fields?
            if (Pf != Pi) for(int p=0; p < np; ++p) Pf(i,j,k,p) = Pi(i,j,k,p); // Seed with initial state
            pflag(i, j, k) = U_to_P(G, Uf, eos, i, j, k, Loci::center, Pf);
        }
    );

    auto end = TIME_NOW;

#if DEBUG
    cerr << "Get Flux: " << PRINT_SEC(start_flux_ct - start_get_flux) << endl;
    cerr << "Flux CT: " << PRINT_SEC(start_uberkernel - start_flux_ct) << endl;
    cerr << "Uberkernel: " << PRINT_SEC(start_utop - start_uberkernel) << endl;
    cerr << "U_to_P: " << PRINT_SEC(end - start_utop) << endl;
#endif

    return ndt;
}
