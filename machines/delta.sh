
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
  NPROC=64

  module purge
  module load cmake

  if [[ $ARGS == *"cuda"* ]]
  then
    # GPU Compile
    # 4-device MPI
    MPI_EXTRA_ARGS="--map-by ppr:1:node:pe=16"
    MPI_NUM_PROCS=1

    # Device-side buffers are broken on some Nvidia machines
    EXTRA_FLAGS="-DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON $EXTRA_FLAGS"

    # Load common GPU modules
    module load gcc/11.4.0 cuda/11.8.0  openmpi/4.1.5+cuda

    if [[ $ARGS == *"latest"* ]]; then
      # nvhpc only on request, MPI crashes
      module load nvhpc_latest openmpi-5.0_beta
      C_NATIVE=nvc
      CXX_NATIVE=nvc++
    elif [[ $ARGS == *"gcc"* ]]; then
      C_NATIVE=gcc
      CXX_NATIVE=g++
    else
      module load nvhpc_latest/22.11 openmpi
      C_NATIVE=nvc
      CXX_NATIVE=nvc++
    fi
  else
    # CPU Compile
    module load modtree/cpu gcc
    MPI_NUM_PROCS=1
  fi
fi
