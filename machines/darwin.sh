# LANL Darwin.  A little bit of everything

# Must list which node you're compiling for,
# from the options below

if [[ ($HOSTNAME == "cn"* || $HOSTNAME == "darwin"*) &&
      ("$PWD" == "/projects/jacamar-ci"* || "$PWD" == "/vast"*) ]]; then
  module purge
  module load cmake

  # Where we're going, we don't need system libraries
  ARGS="$ARGS hdf5"

  # 1. Load compiler stack
  if [[ "$ARGS" == *"gcc10"* ]]; then
    module load gcc/10.4.0
    C_NATIVE=gcc
    CXX_NATIVE=g++
  elif [[ "$ARGS" == *"gcc"* ]]; then
    # Default GCC
    module load gcc/12.1.0
    C_NATIVE=gcc
    CXX_NATIVE=g++
  elif [[ "$ARGS" == *"aocc"* ]]; then
    module load aocc
    C_NATIVE=clang
    CXX_NATIVE=clang++
  elif [[ "$ARGS" == *"nvhpc"* ]]; then
    module load nvhpc
    C_NATIVE="nvc"
    CXX_NATIVE="nvc++"
    # New NVHPC doesn't like CUDA_HOME
    export NVHPC_CUDA_HOME="$CUDA_HOME"
    unset CUDA_HOME
  elif [[ "$ARGS" == *"icc"* ]]; then
    module load intel-classic/2021.3.0
    C_NATIVE=icc
    CXX_NATIVE=icpc
  else
    # Default: NVHPC if cuda else IntelLLVM
    if [[ "$ARGS" == *"cuda"* ]]; then
      module load nvhpc
      C_NATIVE="nvc"
      CXX_NATIVE="nvc++"
      # New NVHPC doesn't like CUDA_HOME
      export NVHPC_CUDA_HOME="$CUDA_HOME"
      unset CUDA_HOME
    else
      module load intel
      C_NATIVE=icx
      CXX_NATIVE=icpx
    fi
  fi

  # 2. Load accelerator libraries
  if [[ "$ARGS" == *"cuda"* ]]; then
    module load cuda/12.0.0 nvhpc
    PREFIX_PATH=$NVHPC_ROOT
    EXTRA_FLAGS="-DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON $EXTRA_FLAGS"
  elif [[ "$ARGS" == *"hip"* ]]; then
    module load rocm/5.4.3 #openmpi/5.0.0rc11-gcc_13.1.0
    source ~/libs/env.sh
    C_NATIVE=hipcc
    CXX_NATIVE=hipcc
    export CXXFLAGS="-fopenmp $CXXFLAGS"
  else
    # load OpenMPI for CPU builds...
    module load openmpi
  fi

  # ... or if we force it (CI)
  if [[ "$ARGS" == *"ompi"* ]]; then
    module load openmpi
  fi

  # 3. Set architecture
  # These are orthogonal to above, so long as the hardware
  # supports the paradigm
  # Note this also specifies cores to use for compiling
  NPROC=$(($(nproc) / 2))
  if [[ "$ARGS" == *"arm-ampere"* ]]; then
    HOST_ARCH="ARMV81"
    DEVICE_ARCH="AMPERE80"
    MPI_NUM_PROCS_D=2
    NODE_SLICE=1
  elif [[ "$ARGS" == *"arm-hopper"* ]]; then
    HOST_ARCH="ARMV81"
    DEVICE_ARCH="HOPPER90"
    MPI_NUM_PROCS_D=1
    NODE_SLICE=1
  elif [[ "$ARGS" == *"ampere"* ]]; then
    HOST_ARCH="ZEN3"
    DEVICE_ARCH="AMPERE80"
    MPI_NUM_PROCS_D=2
    NODE_SLICE=1
  elif [[ "$ARGS" == *"volta"* ]]; then
    HOST_ARCH="HSW"
    DEVICE_ARCH="VOLTA70"
    MPI_NUM_PROCS_D=1
    # Some nodes have 2 GPUs, be conservative
    NODE_SLICE=2
  elif [[ "$ARGS" == *"knl"* ]]; then
    HOST_ARCH="KNL"
    MPI_NUM_PROCS_D=1
    # 4-way SMT, not 2
    NODE_SLICE=2
  elif [[ "$ARGS" == *"hsw"* ]]; then
    HOST_ARCH="HSW"
    MPI_NUM_PROCS_D=1
    NODE_SLICE=1
  elif [[ "$ARGS" == *"skx"* ]]; then
    HOST_ARCH="SKX"
    MPI_NUM_PROCS_D=1
    NODE_SLICE=1
  elif [[ "$ARGS" == *"zen2"* ]]; then
    HOST_ARCH=ZEN2
    MPI_NUM_PROCS_D=1
    NODE_SLICE=1
  elif [[ "$ARGS" == *"zen3"* ]]; then
    HOST_ARCH=ZEN3
    MPI_NUM_PROCS_D=1
    NODE_SLICE=1
  elif [[ "$ARGS" == *"mi250"* ]]; then
    HOST_ARCH=ZEN3
    DEVICE_ARCH=VEGA90A
    MPI_NUM_PROCS_D=8
    NODE_SLICE=2
  else
    echo "Must specify an architecture on Darwin!"
    exit
  fi
  MPI_NUM_PROCS=${MPI_NUM_PROCS:-$MPI_NUM_PROCS_D}

  # Runtime
  MPI_EXE="mpirun"
  # Lead MPI to water
  MPI_EXTRA_ARGS="--map-by ppr:${MPI_NUM_PROCS}:node:pe=$(($NPROC / $MPI_NUM_PROCS / $NODE_SLICE))"
fi
