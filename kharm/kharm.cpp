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

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>

using namespace Kokkos;
using namespace std;

int main(int argc, char **argv)
{
    mpi_init(argc, argv);
    Kokkos::initialize();
    {
        std::cerr << "K/HARM version " << VERSION << std::endl;
        std::cerr << "Using Kokkos environment:" << std::endl;
        DefaultExecutionSpace::print_configuration(std::cerr);
        std::cerr << std::endl;

        // TODO read an input with grid size here
        int sz = 128;
        int ng = 3;
        int nvar = 8;

        CoordinateSystem *coords = new Minkowski();
        Grid G(coords, {sz, sz, sz}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, ng, nvar);
        EOS eos = Gammalaw(13/9);
        cerr << "Grid allocated" << std::endl;
        G.init_grids();
        cerr << "Grid initialized" << std::endl;

        // Allocate and initialize host primitives
        GridVarsHost h_vars_input = mhdmodes(G, 0);
        cerr << "Vars initialized" << std::endl;
        dump(G, h_vars_input, Parameters(), "dump_0000.h5", true);

        // Allocate device memory and host mirror memory
        GridVars vars("all_vars", G.gn1, G.gn2, G.gn3, G.nvar);
        auto m_vars = create_mirror_view(vars);
        cerr << "Memory initialized" << std::endl;

        // Copy input (no ghosts, Host order) into working array (ghosts, device order)
        // deep_copy would do this automatically if not for ghosts (TODO try that?)
        parallel_for("copy_to_ghosts", G.h_bulk_0_p(),
            KOKKOS_LAMBDA (const int i, const int j, const int k, const int p) {
                        m_vars(i + G.ng, j + G.ng, k + G.ng, p) = h_vars_input(i, j, k, p);
            }
        );
        cerr << "Copying to device" << endl;
        deep_copy(vars, m_vars);

        cerr << "Starting iteration" << std::endl;

        double dt = 1.e-5;
        for (int out_iter = 0; out_iter < 10; ++out_iter)
        {
            for (int iter = 0; iter < 10; ++iter)
            {
                dt = step(G, eos, vars, dt);
            }

            deep_copy(m_vars, vars);
            dump(G, m_vars, Parameters(), string_format("dump_%04d.h5", out_iter+1), true);
        }
        delete coords;
    }
    Kokkos::finalize();
}
