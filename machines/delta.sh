
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
    # 4-device MPI w/mapping, should play nice with different numbers
    MPI_NUM_PROCS=${MPI_NUM_PROCS:-4}
    MPI_EXTRA_ARGS="--map-by ppr:$MPI_NUM_PROCS:node:pe=16"

    if [[ "$ARGS" == *"hostside"* ]]; then
      # Device-side buffers are broken on some Nvidia machines
      EXTRA_FLAGS="-DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON $EXTRA_FLAGS"
    fi

    if [[ $ARGS == *"gcc"* ]]; then
      module load gcc/11.4.0 cuda/11.8.0 openmpi/4.1.5+cuda
      C_NATIVE=gcc
      CXX_NATIVE=g++
    elif [[ $ARGS == *"cray"* ]]; then
      module load PrgEnv-gnu cuda craype-x86-milan craype-accel-ncsa
      export MPICH_GPU_SUPPORT_ENABLED=1
      export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
      C_NATIVE=cc
      CXX_NATIVE=CC
    else
      # Default to gcc since we know that works
      module load gcc/11.4.0 cuda/11.8.0 openmpi/4.1.5+cuda
      C_NATIVE=gcc
      CXX_NATIVE=g++
    fi
  else
    # CPU Compile
    module load modtree/cpu gcc
    MPI_NUM_PROCS=1
  fi
fi
