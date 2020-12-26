// MPI wrappers
// Some convenient MPI calls, for things like global reductions that Parthenon doesn't cover
// This file has two different versions, depending on MPI_PARALLEL macro
// This way, the rest of the code can assume MPI is available and should be used,
// but consistent results are generated without it for free
// Trust me it makes everything 1000x more readable
#pragma once

// TODO are there alternative convenience functions in Parthenon I should use?
// TODO single/double overloads if I ever use floats

#ifdef MPI_PARALLEL

#include "globals.hpp"

#include <mpi.h>

static auto comm = MPI_COMM_WORLD;

inline void MPIBarrier()
{
    MPI_Barrier(comm);
}

inline double MPIMax(double f)
{
    double fmax;
    MPI_Allreduce(&f, &fmax, 1, MPI_DOUBLE, MPI_MAX, comm);
    return fmax;
}

inline double MPIMin(double f)
{
    double fmin;
    MPI_Allreduce(&f, &fmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    return fmin;
}

inline double MPISum(double f)
{

    double local;
    MPI_Allreduce(&f, &local, 1, MPI_DOUBLE, MPI_SUM, comm);
    return local;
}

inline int MPISumInt(int f)
{
    int local;
    MPI_Allreduce(&f, &local, 1, MPI_INT, MPI_SUM, comm);
    return local;
}

inline void MPIReduceVector(double *vec_send, double *vec_recv, int len)
{
    MPI_Allreduce(vec_send, vec_recv, len, MPI_DOUBLE, MPI_SUM, comm);
}

inline void MPIBroadcastInt(int *val)
{
    MPI_Bcast(val, 1, MPI_INT, 0, comm);
}

inline void MPIBroadcastDbl(double *val)
{
    MPI_Bcast(val, 1, MPI_DOUBLE, 0, comm);
}

inline bool MPIRank0()
{
    return (parthenon::Globals::my_rank == 0 ? 1 : 0);
}
#else
// Dummy versions of calls

inline void MPIBarrier() {}
inline double MPIMax(double f) { return f; }
inline double MPIMin(double f) { return f; }
inline double MPISum(double f) { return f; }
inline int MPISumInt(int f) { return f; }
inline void MPIReduceVector(double *vec_send, double *vec_recv, int len)
{
    for (int i = 0; i < len; i++)
        vec_recv[i] = vec_send[i];
}
inline bool MPIRank0() { return 1; }
inline void MPIBroadcastInt(int *val) {}
inline void MPIBroadcastDbl(double *val) {}
#endif // MPI_PARALLEL
