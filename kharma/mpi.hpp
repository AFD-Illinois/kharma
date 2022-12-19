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
 * Perform a Parthenon MPI reduction.
 * Now that Parthenon cleans up communicators, this is basically
 * how all reductions should be done.
 * However, Reduction objects should have a longer lifetime if
 * possible, as they require some overhead to create/destroy.
 */
template<typename T>
inline T MPIReduce_once(T f, MPI_Op O)
{
    // TODO Can this borrow from an object with global lifetime?
    static parthenon::AllReduce<T> reduction;
    reduction.val = f;
    reduction.StartReduce(O);
    // Wait on results
    while (reduction.CheckReduce() == parthenon::TaskStatus::incomplete);
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
