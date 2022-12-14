
# Config for NCSA Delta, XSEDE's GPU resource

# The Delta modules are in flux, YMMV
# If the default NVHPC compile does not work,
# try specifying 'gcc' to use the default stack

# Also note that Delta's hdf5 is no longer serviceable (?)
# So run './make.sh hdf5 clean cuda'

if [[ $HOST == *".delta.internal.ncsa.edu" ]]
then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=AMPERE80

  # Load common modules
  module purge
  module load modtree/gpu cmake
  MPI_EXE=mpirun

  if [[ $ARGS == *"cuda"* ]]
  then
    # GPU Compile
    # 4-device MPI
    MPI_EXTRA_ARGS="--map-by ppr:4:node:pe=16"
    MPI_NUM_PROCS=4

    if [[ $ARGS == *"nvhpc"* ]]; then
      # nvhpc only on request, MPI crashes
      module load nvhpc_latest openmpi-5.0_beta
      C_NATIVE=nvc
      CXX_NATIVE=nvc++
    else # TODO NVHPC not-latest
      C_NATIVE=gcc
      CXX_NATIVE=g++
    fi
  else
    # CPU Compile
    module load modtree/cpu gcc
    MPI_NUM_PROCS=1
  fi
fi
