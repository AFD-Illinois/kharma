/******************************************************************************
 *                                                                            *
 * MPI.C                                                                      *
 *                                                                            *
 * HANDLES COMMUNICATION ACROSS MPI NODES                                     *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

#include <mpi.h>

static MPI_Comm comm;
static int neighbors[3][3][3];
static MPI_Datatype face_type[3];
static MPI_Datatype pflag_face_type[3];
static int rank;
static int numprocs;

void mpi_initialization(int argc, char *argv[])
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

  // Diagnostic for processor topology
  if (DEBUG) {
    int me = mpi_myrank();
    fprintf(stderr,"Process %d topology:\n", me);
    fprintf(stderr,"%d in X:[%d]\t[%d]\t[%d]\n", me, neighbors[1][1][0], neighbors[1][1][1], neighbors[1][1][2]);
    fprintf(stderr,"%d in Y:[%d]\t[%d]\t[%d]\n", me, neighbors[1][0][1], neighbors[1][1][1], neighbors[1][2][1]);
    fprintf(stderr,"%d in Z:[%d]\t[%d]\t[%d]\n", me, neighbors[0][1][1], neighbors[1][1][1], neighbors[2][1][1]);
  }

  // Start and stop in global index space
  // These are in the usual X1,2,3 order, or things would get /very confusing/
  int sdims[3] = {N3TOT, N2TOT, N1TOT};
  for (int d = 0; d < 3; d++) {
    global_start[2-d] = coord[d] * sdims[d]/cpudims[d];
    global_stop[2-d] = (coord[d]+1) * sdims[d]/cpudims[d];
  }

  if (DEBUG) {
    printf("Process %d has X,Y,Z space [%d-%d, %d-%d, %d-%d]\n", rank,
           global_start[0], global_stop[0],
           global_start[1], global_stop[1],
           global_start[2], global_stop[2]);
  }

  // Make MPI datatypes
  MPI_Datatype scalar_type = MPI_DOUBLE;
  MPI_Datatype flag_type = MPI_INT;

  // N3 face: Update all zones
  // Need slice P[0-NVAR][i:i+NG][:][:]

  int count = NVAR;
  int countp = 1;
  int block3 = NG*(N2+2*NG)*(N1+2*NG);
  int stride3 = (N3+2*NG)*(N2+2*NG)*(N1+2*NG);

  MPI_Type_vector(count, block3, stride3, scalar_type, &face_type[0]);
  MPI_Type_commit(&face_type[0]);
  MPI_Type_vector(countp, block3, stride3, flag_type, &pflag_face_type[0]);
  MPI_Type_commit(&pflag_face_type[0]);

  // N2 face: update all N1 zones, but only current good N3
  // Slice P[0-NVAR][NG:N3+NG][i:i+NG][:]

  int sizes[4] = {NVAR,N3+2*NG,N2+2*NG,N1+2*NG};
  int subsizes2[4] = {NVAR,N3,NG,N1+2*NG};
  int starts[4] = {0,0,0,0};
  MPI_Type_create_subarray(4, sizes, subsizes2, starts,
               MPI_ORDER_C, scalar_type, &face_type[1]);
  MPI_Type_commit(&face_type[1]);

  int sizes_pflag[3] = {N3+2*NG,N2+2*NG,N1+2*NG};
  int subsizes2_pflag[3] = {N3,NG,N1+2*NG};
  int starts_pflag[3] = {0,0,0};
  MPI_Type_create_subarray(3, sizes_pflag, subsizes2_pflag, starts_pflag,
               MPI_ORDER_C, flag_type, &pflag_face_type[1]);
  MPI_Type_commit(&pflag_face_type[1]);

  // N1 face: update only current good zones (No ghosts)
  // Slice P[0-NVAR][NG:N3+NG][NG:N2+NG][i:i+NG]

  int subsizes3[4] = {NVAR,N3,N2,NG};
  MPI_Type_create_subarray(4, sizes, subsizes3, starts,
               MPI_ORDER_C, scalar_type, &face_type[2]);
  MPI_Type_commit(&face_type[2]);

  int subsizes3_pflag[3] = {N3,N2,NG};
  MPI_Type_create_subarray(3, sizes_pflag, subsizes3_pflag, starts_pflag,
               MPI_ORDER_C, flag_type, &pflag_face_type[2]);
  MPI_Type_commit(&pflag_face_type[2]);

  MPI_Barrier(comm);
}

void mpi_finalize()
{
  MPI_Finalize();
}

// Share face data
int sync_mpi_bound_X1(struct FluidState *S)
{

  // We don't check returns since MPI kindly crashes on failure
#if N1 > 1
  // First send right/receive left
  MPI_Sendrecv(&(S->P[0][NG][NG][N1]), 1, face_type[2], neighbors[1][1][2], 0,
           &(S->P[0][NG][NG][0]), 1, face_type[2], neighbors[1][1][0], 0, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(pflag[NG][NG][N1]), 1, pflag_face_type[2], neighbors[1][1][2], 6,
           &(pflag[NG][NG][0]), 1, pflag_face_type[2], neighbors[1][1][0], 6, comm, MPI_STATUS_IGNORE);
  // And back
  MPI_Sendrecv(&(S->P[0][NG][NG][NG]), 1, face_type[2], neighbors[1][1][0], 1,
           &(S->P[0][NG][NG][N1+NG]), 1, face_type[2], neighbors[1][1][2], 1, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(pflag[NG][NG][NG]), 1, pflag_face_type[2], neighbors[1][1][0], 7,
           &(pflag[NG][NG][N1+NG]), 1, pflag_face_type[2], neighbors[1][1][2], 7, comm, MPI_STATUS_IGNORE);
#endif

  return 0;
}

int sync_mpi_bound_X2(struct FluidState *S)
{

#if N2 > 1
  MPI_Sendrecv(&(S->P[0][NG][N2][0]), 1, face_type[1], neighbors[1][2][1], 2,
           &(S->P[0][NG][0][0]), 1, face_type[1], neighbors[1][0][1], 2, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(pflag[NG][N2][0]), 1, pflag_face_type[1], neighbors[1][2][1], 8,
           &(pflag[NG][0][0]), 1, pflag_face_type[1], neighbors[1][0][1], 8, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&(S->P[0][NG][NG][0]), 1, face_type[1], neighbors[1][0][1], 3,
           &(S->P[0][NG][N2+NG][0]), 1, face_type[1], neighbors[1][2][1], 3, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(pflag[NG][NG][0]), 1, pflag_face_type[1], neighbors[1][0][1], 9,
           &(pflag[NG][N2+NG][0]), 1, pflag_face_type[1], neighbors[1][2][1], 9, comm, MPI_STATUS_IGNORE);
#endif

  return 0;
}

int sync_mpi_bound_X3(struct FluidState *S)
{

#if N3 > 1
  MPI_Sendrecv(&(S->P[0][N3][0][0]), 1, face_type[0], neighbors[2][1][1], 4,
           &(S->P[0][0][0][0]), 1, face_type[0], neighbors[0][1][1], 4, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(pflag[N3][0][0]), 1, pflag_face_type[0], neighbors[2][1][1], 10,
           &(pflag[0][0][0]), 1, pflag_face_type[0], neighbors[0][1][1], 10, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(&(S->P[0][NG][0][0]), 1, face_type[0], neighbors[0][1][1], 5,
           &(S->P[0][N3+NG][0][0]), 1, face_type[0], neighbors[2][1][1], 5, comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(pflag[NG][0][0]), 1, pflag_face_type[0], neighbors[0][1][1], 11,
           &(pflag[N3+NG][0][0]), 1, pflag_face_type[0], neighbors[2][1][1], 11, comm, MPI_STATUS_IGNORE);
#endif

  return 0;
}

void mpi_barrier() {
  MPI_Barrier(comm);
}

int mpi_nprocs() {
  return numprocs;
}

int mpi_myrank() {
  return rank;
}

double mpi_max(double f)
{
  double fmax;
  MPI_Allreduce(&f, &fmax, 1, MPI_DOUBLE, MPI_MAX, comm);
  return fmax;
}

double mpi_min(double f)
{
  double fmin;
  MPI_Allreduce(&f, &fmin, 1, MPI_DOUBLE, MPI_MIN, comm);
  return fmin;
}

double mpi_reduce(double f) {

  double local;
  MPI_Allreduce(&f, &local, 1, MPI_DOUBLE, MPI_SUM, comm);
  return local;
}

int mpi_reduce_int(int f) {

  int local;
  MPI_Allreduce(&f, &local, 1, MPI_INT, MPI_SUM, comm);
  return local;
}

void mpi_reduce_vector(double *vec_send, double *vec_recv, int len)
{
  MPI_Allreduce(vec_send, vec_recv, len, MPI_DOUBLE, MPI_SUM, comm);
}

void mpi_int_broadcast(int *val)
{
  MPI_Bcast(val, 1, MPI_INT, 0, comm);
}

void mpi_dbl_broadcast(double *val)
{
  MPI_Bcast(val, 1, MPI_DOUBLE, 0, comm);
}

int mpi_io_proc()
{
  return (rank == 0 ? 1 : 0);
}
