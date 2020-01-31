/**
 * Coordinates applying a time evolution to the fluid
 */

#include "utils.hpp"
#include "decs.hpp"
#include "mpi.hpp"
#include "fluxes.hpp"
#include "U_to_P.hpp"
#include "source.hpp"

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
    static GReal t;

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
    FLAG("Allocate flux temporaries");

    // Get the fluxes in each direction on the zone faces
    double ndt = get_flux(G, eos, Ps, F1, F2, F3);
    FLAG("Get flux");
    // Fix boundary fluxes
//    fix_flux(F1, F2, F3);
//    FLAG("Fix flux");
    // Constrained transport for B
    flux_ct(G, F1, F2, F3);
    FLAG("Flux CT");

    // Update Si to Sf
    // TODO get_state_vec equivalent, just pass a slice
    Kokkos::parallel_for("get_dU", G.bulk_ng(),
        KOKKOS_LAMBDA_3D {
            get_state(G, Ps, i, j, k, Loci::center, Dtmp);
        }
    );
    get_fluid_source(G, Ps, Dtmp, eos, dU); // TODO PAIN POINT
    FLAG("Get source");


    if (Pi != Ps) { // Only need this if Dtmp does not already hold the goods
        Kokkos::parallel_for("get_Pi_D", G.bulk_ng(),
            KOKKOS_LAMBDA_3D {
                get_state(G, Pi, i, j, k, Loci::center, Dtmp);
            }
        );
        FLAG("Get PiD");
    }
    Kokkos::parallel_for("get_Ui", G.bulk_ng(),
        KOKKOS_LAMBDA_3D {
            prim_to_flux(G, Pi, Dtmp, eos, i, j, k, Loci::center, 0, Ui);
        }
    );
    FLAG("Get Ui");

    Kokkos::parallel_for("finite_diff", G.bulk_ng_p(),
        KOKKOS_LAMBDA_VARS {
                Uf(i, j, k, p) = Ui(i, j, k, p) +
                                    dt * ((F1(i, j, k, p) - F1(i+1, j, k, p)) / G.dx1 +
                                        (F2(i, j, k, p) - F2(i, j+1, k, p)) / G.dx2 +
                                        (F3(i, j, k, p) - F3(i, j, k+1, p)) / G.dx3 +
                                        dU(i, j, k, p));
        }
    );
    FLAG("Finite diff");

    // If the pointers are different, initialize final state to initial state
    // This seeds the U_to_P iteration
    if (Pi != Pf) {
        Kokkos::parallel_for("memcpy", G.all_0_p(),
            KOKKOS_LAMBDA_VARS {
                Pf(i,j,k,p) = Pi(i,j,k,p);
            }
        );
    }

    // Finally, recover the primitives at the end of the substep
    Kokkos::parallel_for("cons_to_prim", G.bulk_ng(),
        KOKKOS_LAMBDA_3D {
            // TODO ever called on not the center? Maybe w/face fields?
            pflag(i, j, k) = U_to_P(G, Uf, eos, i, j, k, Loci::center, Pf);
        }
    );

    return ndt;
}
