/*
 * K/HARM -- Implementation of the HARM scheme for GRMHD,
 * in C++ with Kokkos performance portability library
 *
 * Ben Prather
 */

#include "utils.hpp"
#include "decs.hpp"

#include "self_init.hpp"
#include "grid.hpp"
#include "io.hpp"
#include "step.hpp"
#include "debug.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <chrono>

using namespace Kokkos;
using namespace std;

int main(int argc, char **argv)
{
    //mpi_init(argc, argv);
    Kokkos::initialize();
    {
        std::cerr << "K/HARM version " << VERSION << std::endl;
        std::cerr << "Using Kokkos environment:" << std::endl;
        DefaultExecutionSpace::print_configuration(std::cerr);
        std::cerr << std::endl;

        // TODO parse paraemeters and/or read restart here
        Parameters params;
        params.verbose = 1;
        int sz = 128;
        int ng = 3;
        int nvar = 8;

        // Allocate device-side objects
        EOS *eos;
        eos = (EOS*)Kokkos::kokkos_malloc(sizeof(GammaLaw));
        Kokkos::parallel_for("CreateEOS", 1,
            KOKKOS_LAMBDA(const int&) {
                new ((GammaLaw*)eos) GammaLaw(4./3);
            }
        );

        CoordinateSystem* coords;
        // if (true) {
        //     coords = (CoordinateSystem*)Kokkos::kokkos_malloc(sizeof(FMKS));
        //     Kokkos::parallel_for("CreateFMKS", 1,
        //         KOKKOS_LAMBDA(const int&) {
        //             new ((FMKS*)coords) FMKS();
        //         }
        //     );
        // } else {
            coords = (CoordinateSystem*)Kokkos::kokkos_malloc(sizeof(Minkowski));
            Kokkos::parallel_for("CreateMinkowski", 1,
                KOKKOS_LAMBDA(const int&) {
                    new ((Minkowski*)coords) Minkowski();
                }
            );
        // }

        // Make the grid
        Grid G(coords, {sz, sz, sz}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, ng, nvar);
        cerr << "Grid initialized" << std::endl;

        // Make an array of the primitive variables
        GridVarsHost h_vars_input = mhdmodes(G, 1);
        cerr << "Vars initialized" << std::endl;
        dump(G, h_vars_input, params, "dump_0000.h5", true);

        // Allocate device memory and host mirror memory
        GridVars vars("all_vars", G.gn1, G.gn2, G.gn3, G.nvar);
        auto m_vars = create_mirror_view(vars);
        cerr << "Memory initialized" << std::endl;

        // Copy input (no ghosts, Host order) into working array (ghosts, device order)
        // deep_copy would do this automatically if not for ghosts (TODO try that?)
        parallel_for("copy_to_ghosts", G.h_bulk_0_p(),
            KOKKOS_LAMBDA_VARS {
                m_vars(i + G.ng, j + G.ng, k + G.ng, p) = h_vars_input(i, j, k, p);
            }
        );
        cerr << "Copying to device" << endl;
        deep_copy(vars, m_vars);

        cerr << "Starting iteration" << endl;

        auto walltime_start = std::chrono::high_resolution_clock::now();
        double dt = 1.e-3;
        double t = 0;
        double tend = 0.5;
        double dump_cadence = 0.05;
        double next_dump_t = t + dump_cadence;
        int dump_cnt = 1;
        bool dump_every_step = false;
        bool dump_this_step = false;

        int out_iter = 0;
        while (t < tend)
        {
            if (dump_every_step) {
                // Skip any messing with dt, we don't care when dumps are
                dump_this_step = true;
            } else if (t+dt > next_dump_t) {
                // TODO test this: 1. Make sure dt>next_dump_t even at late time, and that behavior is expected
                dt = next_dump_t - t;
                dump_this_step = true;
            }

            step(G, eos, vars, params, dt, t);
            cerr << string_format("t = %.5f dt = %.5f", t, dt) << endl;

            if (out_iter % 10 == 0) {
                auto walltime_now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = walltime_now - walltime_start;
                cerr << "ZCPS " << G.n1*G.n2*G.n3*(out_iter+1)/elapsed.count() << endl;
            }

            if (dump_this_step) {
                deep_copy(m_vars, vars);
                dump(G, m_vars, params, string_format("dump_%04d.h5", dump_cnt), true);
                ++dump_cnt;
                next_dump_t += dump_cadence;
            }

            ++out_iter;
            dump_this_step = false;
        }

        // Clean up device-side objects
        Kokkos::parallel_for("DestroyObjects", 1,
            KOKKOS_LAMBDA(const int&) {
                coords->~CoordinateSystem();
            }
        );
        Kokkos::kokkos_free(coords);
    }
    Kokkos::finalize();
}
