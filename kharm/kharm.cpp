/*
 * K/HARM -- Implementation of the HARM scheme for GRMHD,
 * in C++ with Kokkos performance portability library
 *
 * Ben Prather
 */

#include "decs.hpp"
#include "diffuse.hpp"
#include "self_init.hpp"
#include "grid.hpp"
#include "io.hpp"

#include "step.hpp"

#include "utils.hpp"

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

        // TODO Read file here as primary input

        CoordinateSystem coords = Minkowski();
        Grid G(&coords, {128, 128, 128}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
        EOS eos = Gammalaw(13/9);
        cerr << "Grid init" << std::endl;

        GridVarsHost h_vars_input = mhdmodes(G, 0);
        cerr << "Vars init" << std::endl;
        dump(G, h_vars_input, Parameters(), "dump_0000.h5", true);

        GridVars vars("all_vars", G.gn1, G.gn2, G.gn3, G.nvar);
        auto m_vars = create_mirror_view(vars);

        // Copy input (no ghosts, Host order) into working array (ghosts, device order)
        // deep_copy would do this automatically if not for ghosts (TODO try that?)
        int ng = G.ng; int nvar = G.nvar;
        parallel_for("copy_to_ghosts", G.h_bulk_0(),
            KOKKOS_LAMBDA (const int i, const int j, const int k) {
                    for (int p = 0; p < nvar; ++p)
                        m_vars(i + ng, j + ng, k + ng, p) = h_vars_input(i, j, k, p);
            }
        );

        // copy TO DEVICE
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
    }
    Kokkos::finalize();
}
