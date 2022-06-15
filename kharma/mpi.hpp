// MPI wrappers
// Some convenient MPI calls, for things like global reductions that Parthenon doesn't cover
// This file has two different versions, depending on MPI_PARALLEL macro
// This way, the rest of the code can assume MPI is available and should be used,
// but consistent results are generated without it for free
// Trust me it makes everything 1000x more readable
#pragma once

#include <parthenon/parthenon.hpp>

#ifdef MPI_PARALLEL

#include <mpi.h>

static auto comm = MPI_COMM_WORLD;

// Wrappers to make Parthenon-scope MPI interface global,
// plus an easy barrier in case you need it for debugging
inline bool MPIRank()
{
    return parthenon::Globals::my_rank;
}
inline bool MPIRank0()
{
    return (parthenon::Globals::my_rank == 0 ? true : false);
}
inline void MPIBarrier()
{
    MPI_Barrier(comm);
}

/**
 * Perform a Parthenon MPI reduction. ONCE.
 * Parthenon does not clean up communicators, and
 * it is further bad practice to allocate a new one per-step.
 * So, avoid this function except in initialization.
 * 
 * Instead, copy/paste the contents & use a static declaration for
 * the AllReduce object.  Replace with a Reduce object if you don't
 * need the result on all ranks.
 */
template<typename T>
inline T MPIReduce_once(T f, MPI_Op O)
{
    AllReduce<T> reduction;
    reduction.val = f;
    reduction.StartReduce(O);
    // Wait on results
    while (reduction.CheckReduce() == TaskStatus::incomplete);
    return reduction.val;
}
#else
// Use Parthenon's MPI_Op workaround
//typedef MPI_Op parthenon::MPI_Op;

// Dummy versions of calls
inline void MPIBarrier() {}
inline bool MPIRank() { return 0; }
inline bool MPIRank0() { return true; }

template<typename T>
inline T MPIReduce_once(T f, MPI_Op O)
{
    return f;
}

#endif // MPI_PARALLEL
