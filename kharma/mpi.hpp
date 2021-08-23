// MPI wrappers
// Some convenient MPI calls, for things like global reductions that Parthenon doesn't cover
// This file has two different versions, depending on MPI_PARALLEL macro
// This way, the rest of the code can assume MPI is available and should be used,
// but consistent results are generated without it for free
// Trust me it makes everything 1000x more readable
#pragma once

// TODO overloads for single

#ifdef MPI_PARALLEL

#include "globals.hpp"

#include <mpi.h>

static auto comm = MPI_COMM_WORLD;

// UNIVERSAL
inline bool MPIRank0()
{
    return (parthenon::Globals::my_rank == 0 ? true : false);
}
inline void MPIBarrier()
{
    MPI_Barrier(comm);
}

// DOUBLE
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
inline void MPIBroadcast(double val)
{
    MPI_Bcast(&val, 1, MPI_DOUBLE, 0, comm);
}

// FLOAT
inline float MPIMax(float f)
{
    float fmax;
    MPI_Allreduce(&f, &fmax, 1, MPI_FLOAT, MPI_MAX, comm);
    return fmax;
}
inline float MPIMin(float f)
{
    float fmin;
    MPI_Allreduce(&f, &fmin, 1, MPI_FLOAT, MPI_MIN, comm);
    return fmin;
}
inline float MPISum(float f)
{
    float local;
    MPI_Allreduce(&f, &local, 1, MPI_FLOAT, MPI_SUM, comm);
    return local;
}
inline void MPIBroadcast(float val)
{
    MPI_Bcast(&val, 1, MPI_FLOAT, 0, comm);
}

// INT
inline int MPIMax(int f)
{
    int fmax;
    MPI_Allreduce(&f, &fmax, 1, MPI_INT, MPI_MAX, comm);
    return fmax;
}
inline int MPIMin(int f)
{
    int fmin;
    MPI_Allreduce(&f, &fmin, 1, MPI_INT, MPI_MIN, comm);
    return fmin;
}
inline int MPISum(int f)
{
    int local;
    MPI_Allreduce(&f, &local, 1, MPI_INT, MPI_SUM, comm);
    return local;
}
inline void MPIBroadcast(int *val)
{
    MPI_Bcast(&val, 1, MPI_INT, 0, comm);
}

// VECTOR
inline void MPIReduceVector(double *vec_send, double *vec_recv, int len)
{
    MPI_Allreduce(vec_send, vec_recv, len, MPI_DOUBLE, MPI_SUM, comm);
}
#else
// Dummy versions of calls

inline void MPIBarrier() {}
inline bool MPIRank0() { return true; }

inline double MPIMax(double f) { return f; }
inline double MPIMin(double f) { return f; }
inline double MPISum(double f) { return f; }
inline void MPIBroadcast(double val) {}

inline float MPIMax(float f) { return f; }
inline float MPIMin(float f) { return f; }
inline float MPISum(float f) { return f; }
inline void MPIBroadcast(float val) {}

inline int MPIMax(int f) { return f; }
inline int MPIMin(int f) { return f; }
inline int MPISum(int f) { return f; }
inline void MPIBroadcast(int val) {}

inline void MPIReduceVector(double *vec_send, double *vec_recv, int len)
{
    for (int i = 0; i < len; i++)
        vec_recv[i] = vec_send[i];
}
#endif // MPI_PARALLEL
