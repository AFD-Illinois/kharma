// MPI wrappers
// Some convenient MPI calls, for things like global reductions that Parthenon doesn't cover
// This file has two different versions, depending on MPI_PARALLEL macro
// This way, the rest of the code can assume MPI is available and should be used,
// but consistent results are generated without it for free
// Trust me it makes everything 1000x more readable

// TODO is this how I want to do raw MPI comms?  Does Parthenon handle any of this? Should I hook into e.g. the comm?

#include "decs.hpp"

#ifdef MPI_PARALLEL

#include "globals.hpp"

#include <mpi.h>

static auto comm = MPI_COMM_WORLD;

void mpi_barrier()
{
    MPI_Barrier(comm);
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

double mpi_sum(double f)
{

    double local;
    MPI_Allreduce(&f, &local, 1, MPI_DOUBLE, MPI_SUM, comm);
    return local;
}

int mpi_sum_int(int f)
{
    int local;
    MPI_Allreduce(&f, &local, 1, MPI_INT, MPI_SUM, comm);
    return local;
}

void mpi_reduce_vector(double *vec_send, double *vec_recv, int len)
{
    MPI_Allreduce(vec_send, vec_recv, len, MPI_DOUBLE, MPI_SUM, comm);
}

void mpi_broadcast_int(int *val)
{
    MPI_Bcast(val, 1, MPI_INT, 0, comm);
}

void mpi_broadcast_dbl(double *val)
{
    MPI_Bcast(val, 1, MPI_DOUBLE, 0, comm);
}

bool mpi_io_proc()
{
    return (Globals::my_rank == 0 ? 1 : 0);
}
#else
// Dummy versions of calls

void mpi_barrier() {}
double mpi_max(double f) { return f; }
double mpi_min(double f) { return f; }
double mpi_sum(double f) { return f; }
int mpi_sum_int(int f) { return f; }
void mpi_reduce_vector(double *vec_send, double *vec_recv, int len)
{
    for (int i = 0; i < len; i++)
        vec_recv[i] = vec_send[i];
}
bool mpi_io_proc() { return 1; }
void mpi_broadcast_int(int *val) {}
void mpi_broadcast_dbl(double *val) {}
#endif // MPI_PARALLEL