
# Config for NCSA Delta, XSEDE's GPU resource

if [[ $HOST == *".delta.internal.ncsa.edu" ]]
then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=AMPERE80

  module load cmake
  if [[ $ARGS == *"cuda"* ]]
  then
    if [[ $ARGS == *"gcc"* ]]
    then
      module load gcc hdf5
    else
      # Most recent nvhpc.  Keeps system MPI but uses NVHPC's?
      module load nvhpc hdf5
      C_NATIVE=nvc
      CXX_NATIVE=nvc++
    fi
  else
    module load modtree/cpu gcc hdf5
  fi

  # MPI options
  MPI_EXE=mpirun
  MPI_EXTRA_ARGS="--map-by ppr:4:node:pe=16"
  MPI_NUM_PROCS=4
  KOKKOS_NUM_DEVICES=4

  module list
fi
