// mpi.hpp
// Init code adapted from parthenon because it's just so pretty

#include <mpi.h>

/**
 * Initialize MPI and OpenMP according to flags
 * Sets Globals::{my_rank,nranks}
 */
int mpi_init(int argc, char *argv[])
{
#ifdef MPI_PARALLEL
#ifdef OPENMP_PARALLEL
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    return 1;
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for the hybrid parallelzation. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv
              << std::endl;
    MPI_Finalize();
    return 2;
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI Initialization failed." << std::endl;
    return 3;
  }
#endif  // OPENMP_PARALLEL
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank))) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    MPI_Finalize();
    return 4;
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_size failed." << std::endl;
    MPI_Finalize();
    return 5;
  }
#else  // no MPI
  Globals::my_rank = 0;
  Globals::nranks  = 1;
#endif  // MPI_PARALLEL

  return 0;
}

/**
 * Finalize MPI if necessary
 */
void mpi_finalize()
{
#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
}