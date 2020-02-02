/**
 * MPI Functions or dummies so as not to be sprinkling #if USE_MPI everywhere
 */

#if USE_MPI
#include mpi.h

void mpi_x1() {

}

void mpi_init(int argc, char *argv[])
{
  numprocs = N3CPU*N2CPU*N1CPU;

  int cpudims[3] = {N3CPU, N2CPU, N1CPU};

  // Make MPI communication periodic if required
  int periodic[3] = {X3L_BOUND == PERIODIC && X3R_BOUND == PERIODIC,
      X2L_BOUND == PERIODIC && X2R_BOUND == PERIODIC,
      X1L_BOUND == PERIODIC && X1R_BOUND == PERIODIC};

  // Check for minimal required MPI thread support
  int threadSafety;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadSafety);
  if (threadSafety < MPI_THREAD_FUNNELED) {
    fprintf(stderr, "Thread support < MPI_THREAD_FUNNELED. Unsafe.\n");
    exit(1);
  }

  // Set up communicator for Cartesian processor topology
  // Use X3,2,1 ordering
  MPI_Cart_create(MPI_COMM_WORLD, 3, cpudims, periodic, 1, &comm);

  int coord[3];
  MPI_Comm_rank(comm, &rank);
  MPI_Cart_coords(comm, rank, 3, coord);

  // Find the ranks of neighbors, including edge/corner neighbors
  int n[3];
  for (int k = -1; k < 2; k++) {
    n[0] = coord[0] + k;
    for (int j = -1; j < 2; j++) {
      n[1] = coord[1] + j;
      for (int i = -1; i < 2; i++) {
        n[2] = coord[2] + i;
        if (((n[0] < 0 || n[0] >= N3CPU) && !periodic[0]) ||
            ((n[1] < 0 || n[1] >= N2CPU) && !periodic[1]) ||
            ((n[2] < 0 || n[2] >= N1CPU) && !periodic[2])) {
          neighbors[k+1][j+1][i+1] = MPI_PROC_NULL;
        } else {
          MPI_Cart_rank(comm, n, &neighbors[k+1][j+1][i+1]);
        }
      }
    }
  }
}

template<typename T>
T mpi_min(T val) {
    return val;
}

#else
// Alternate no-op versions of all external functions
// Priority to support non-MPI configs given 
#endif