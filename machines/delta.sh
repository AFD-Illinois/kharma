
# Config for NCSA Delta, XSEDE's GPU resource

# The Delta modules are in flux, YMMV
# If the default NVHPC compile does not work,
# try specifying 'gcc' to use the default stack

# Also note that Delta's hdf5 is no longer serviceable (?)
# So run './make.sh hdf5 clean cuda'

if [[ $HOST == *".delta.internal.ncsa.edu" || $HOST == *".delta.ncsa.illinois.edu" ]]
then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=AMPERE80
  MPI_EXE=mpirun

  module purge

  if [[ $ARGS == *"cuda"* ]]
  then
    # GPU Compile
    # 4-device MPI
    MPI_EXTRA_ARGS="--map-by ppr:1:node:pe=16"
    MPI_NUM_PROCS=1

    # Load common GPU modules
    module load modtree/gpu hdf5 cmake

    if [[ $ARGS == *"latest"* ]]; then
      # nvhpc only on request, MPI crashes
      module load nvhpc_latest openmpi-5.0_beta
      C_NATIVE=nvc
      CXX_NATIVE=nvc++
    elif [[ $ARGS == *"gcc"* ]]; then
      C_NATIVE=gcc
      CXX_NATIVE=g++
    else
      module load nvhpc
      #C_NATIVE=nvc
      #CXX_NATIVE=nvc++
    fi
  else
    # CPU Compile
    module load modtree/cpu gcc hdf5 cmake
    MPI_NUM_PROCS=1
  fi
fi
