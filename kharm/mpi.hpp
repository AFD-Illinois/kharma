/**
 * MPI Functions or dummies so as not to be sprinkling #if USE_MPI everywhere
 */

#if USE_MPI
// This is very not supported right now.
// Question whether we even use Boost MPI or stick to native bindings
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
namespace mpi = boost::mpi;
#endif

void mpi_init(int argc, char **argv) {
#if USE_MPI
    static mpi::environment env(argc, argv);
    static mpi::communicator world;
#endif
}

template<typename T>
T mpi_min(T val) {
    return val;
}