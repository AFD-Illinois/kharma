/**
 * Coordinates applying a time evolution to the fluid
 */

#include "utils.hpp"
#include "decs.hpp"
#include "for_loop.hpp"

#include "mpi.hpp"
#include "fluxes.hpp"
#include "U_to_P.hpp"
#include "source.hpp"
#include "boundaries.hpp"

#include "debug.hpp"

#include <chrono>

using namespace std;

// Declarations
double advance_fluid(const Grid &G, const EOS* eos,
                    const GridVars Pi, const GridVars Ps, GridVars Pf,
                    const double dt, GridInt pflag);

/**
 * Take one step.  Returns the Courant limit, to be used for the next step
 */
void step(const Grid& G, const EOS* eos, GridVars vars, Parameters params, double& dt, double& t)
{
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
    if (params.verbose) count_print_flags(pflag);

    // Fixup routines: smooth over outlier zones
    // fixup(G, vars_tmp);
    // FLAG("Fixup Tmp");
    // Need an MPI call _before_ fixup_utop to obtain correct pflags
    //set_bounds(G, Stmp); // TODO flags only. Only if necessary?
    //FLAG("First bounds Tmp");
    // fixup_utoprim(G, vars_tmp);
    // FLAG("Fixup U_to_P Tmp");
    set_bounds(G, vars_tmp, pflag, params);
    FLAG("Full bounds Tmp");

    // Corrector step
    double ndt = advance_fluid(G, eos, vars, vars_tmp, vars, dt, pflag);
    FLAG("Advance Fluid Full");
    if (params.verbose) count_print_flags(pflag);

    // fixup(G, S);
    // FLAG("Fixup Full");
    //set_bounds(G, S);
    //FLAG("First bounds Full");
    // fixup_utoprim(G, S);
    // FLAG("Fixup U_to_P Full");
    set_bounds(G, vars, pflag, params);
    FLAG("Full bounds Full");

    t += dt;

    int pflags;
    Kokkos::parallel_reduce("ndt_min", G.bulk_ng(),
        KOKKOS_LAMBDA (const int &i, const int &j, const int &k, int &local_flags) {
            local_flags += (pflag(i, j, k) == ERR_MAX_ITER);
        }
    , pflags);

    // DEBUG
    cerr << string_format("%d pflags, %d floors", t, pflags, 0 ) << endl;

    dt = 0.9*ndt;
}

double advance_fluid(const Grid &G, const EOS* eos,
                    const GridVars Pi, const GridVars Ps, GridVars Pf,
                    const double dt, GridInt pflag)
{
#if DEBUG
    // GridVars Ui("Ui", G.gn1, G.gn2, G.gn3, G.nvar);
    // GridVars dU("dU", G.gn1, G.gn2, G.gn3, G.nvar);
    GridDerived Dtmp;
    Dtmp.ucon = GridVector("Dtmp_ucon", G.gn1, G.gn2, G.gn3);
    Dtmp.ucov = GridVector("Dtmp_ucov", G.gn1, G.gn2, G.gn3);
    Dtmp.bcon = GridVector("Dtmp_bcon", G.gn1, G.gn2, G.gn3);
    Dtmp.bcov = GridVector("Dtmp_bcov", G.gn1, G.gn2, G.gn3);
    GridVector mhd_stor("mhd_stor", G.gn1, G.gn2, G.gn3);
    GridScalar gamma_stor("gamma_stor", G.gn1, G.gn2, G.gn3);
#endif
    // TODO Cuda is buggy when this is a local temp
    GridVars Ui("Ui", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars dU("dU", G.gn1, G.gn2, G.gn3, G.nvar);

    GridVars Uf("Uf", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars F1("F1", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars F2("F2", G.gn1, G.gn2, G.gn3, G.nvar);
    GridVars F3("F3", G.gn1, G.gn2, G.gn3, G.nvar);
    FLAG("Allocate flux temporaries");

#if DEBUG
    print_a_zone(Pi, 11, 12, 13);
#endif

    auto start_get_flux = TIME_NOW;
    // Get the fluxes in each direction on the zone faces
    double ndt = get_flux(G, eos, Ps, F1, F2, F3);
    FLAG("Get flux");

#if DEBUG
    print_a_zone(F1, 11, 12, 13);
    print_a_zone(F2, 11, 12, 13);
    print_a_zone(F3, 11, 12, 13);
#endif

    // Fix boundary fluxes
//    fix_flux(F1, F2, F3);
//    FLAG("Fix flux");

    auto start_flux_ct = TIME_NOW;
    // Constrained transport for B
    flux_ct(G, F1, F2, F3);
    FLAG("Flux CT");

    auto start_uberkernel = TIME_NOW;
    const int np = G.nvar;
    parallel_for("uber_diff", G.bulk_ng(),
        KOKKOS_LAMBDA_3D {
#if DEBUG
#else
            Derived Dtmp;
            //Real dU[8], Ui[8]; // TODO but what if we use more vars?
#endif

            get_state(G, Pi, i, j, k, Loci::center, Dtmp);
            prim_to_flux(G, Pi, Dtmp, eos, i, j, k, Loci::center, 0, Ui); // (i, j, k, p)

#if DEBUG
            Real mhd[NDIM];
            mhd_calc(Pi, Dtmp, eos, i, j, k, 0, mhd);
            DLOOP1 mhd_stor(i, j, k, mu) = mhd[mu];
            gamma_stor(i, j, k) = mhd_gamma_calc(G, Pi, i, j, k, Loci::center);
#endif

            if (Ps != Pi)
                get_state(G, Ps, i, j, k, Loci::center, Dtmp);
            get_fluid_source(G, Ps, Dtmp, eos, i, j, k, dU);

#if 0
            for(int p=0; p < np; ++p)
                Uf(i, j, k, p) = Ui[p] +
                                    dt * ((F1(i, j, k, p) - F1(i+1, j, k, p)) / G.dx1 +
                                        (F2(i, j, k, p) - F2(i, j+1, k, p)) / G.dx2 +
                                        (F3(i, j, k, p) - F3(i, j, k+1, p)) / G.dx3 +
                                        dU[p]);
#else
            for(int p=0; p < np; ++p)
                Uf(i, j, k, p) = Ui(i, j, k, p) +
                                    dt * ((F1(i, j, k, p) - F1(i+1, j, k, p)) / G.dx1 +
                                        (F2(i, j, k, p) - F2(i, j+1, k, p)) / G.dx2 +
                                        (F3(i, j, k, p) - F3(i, j, k+1, p)) / G.dx3 +
                                        dU(i, j, k, p));
#endif
        }
    );
    FLAG("Uber diff");

#if DEBUG
    print_derived_at(Dtmp, 11, 12, 13);
    print_a_vec(mhd_stor, 11, 12, 13);
    print_a_scalar(gamma_stor, 11, 12, 13);
    print_a_zone(Ui, 11, 12, 13);
    print_a_zone(dU, 11, 12, 13);
    print_a_zone(Uf, 11, 12, 13);
#endif

    auto start_utop = TIME_NOW;
    // Finally, recover the primitives at the end of the substep
    parallel_for("cons_to_prim", G.bulk_ng(),
        KOKKOS_LAMBDA_3D {
            // TODO ever called on not the center? Maybe w/face fields?
            if (Pf != Pi) for(int p=0; p < np; ++p) Pf(i,j,k,p) = Pi(i,j,k,p); // Seed with initial state
            pflag(i, j, k) = U_to_P(G, Uf, eos, i, j, k, Loci::center, Pf);
        }
    );

#if DEBUG
    print_a_zone(Pf, 11, 12, 13);
#endif

    auto end = TIME_NOW;

#if DEBUG
    cerr << "Get Flux: " << PRINT_SEC(start_flux_ct - start_get_flux) << endl;
    cerr << "Flux CT: " << PRINT_SEC(start_uberkernel - start_flux_ct) << endl;
    cerr << "Uberkernel: " << PRINT_SEC(start_utop - start_uberkernel) << endl;
    cerr << "U_to_P: " << PRINT_SEC(end - start_utop) << endl;
#endif

    return ndt;
}
