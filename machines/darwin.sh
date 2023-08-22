# LANL Darwin.  A little bit of everything

# Must list which node you're compiling for:
# ampere for AMD/NVIDIA A100 nodes
# volta for x86/volta of all kinds
# Not working yet:
# arm-nv to compile for devkit ARM/NVIDIA nodes

if [[ $HOSTNAME == "cn"* || $HOSTNAME == "darwin"* ]]; then
  module purge
  module load cmake

  # Help Darwin find the right modules in automated jobs
  if [[ "$ARGS" == *"cuda"* ]]; then
    export MODULEPATH="/projects/darwin-nv/modulefiles/rhel8/aarch64:/projects/darwin-nv/modulefiles/rhel8/aarch64"
  fi

  # Load modules based on first argument...
  if [[ "$ARGS" == *"cuda"* ]]; then
    if [[ "$ARGS" == *"gcc12"* ]]; then
      module load cuda/12.0.0 openmpi gcc/12.1.0
      C_NATIVE=gcc
      CXX_NATIVE=g++
    elif [[ "$ARGS" == *"gcc"* ]]; then
      module load cuda openmpi gcc/10.2.0
      C_NATIVE=gcc
      CXX_NATIVE=g++
    else
      module load nvhpc/22.1 cuda/12.0.0
      C_NATIVE="nvc"
      CXX_NATIVE="nvc++"
      # New NVHPC doesn't like CUDA_HOME
      export NVHPC_CUDA_HOME="$CUDA_HOME"
      unset CUDA_HOME
    fi
  elif [[ "$ARGS" == *"hip"* ]]; then
    module load rocm/5.4.3 #openmpi/5.0.0rc11-gcc_13.1.0
    source ~/libs/env.sh
    C_NATIVE=hipcc
    CXX_NATIVE=hipcc
    export CXXFLAGS="-fopenmp $CXXFLAGS"
  else
    if [[ "$ARGS" == *"gcc"* ]]; then
      module load openmpi gcc/10.2.0
      C_NATIVE=gcc
      CXX_NATIVE=g++
      export CXXFLAGS="-fno-builtin-memset"
    else
      module load openmpi intel
      C_NATIVE=icx
      CXX_NATIVE=icpx
    fi
  fi

  # ...and set architecture according to second.
  # These are orthogonal to above, so long as the hardware
  # supports the paradigm
  NPROC=$(($(nproc) / 2))
  if [[ "$ARGS" == *"arm-ampere"* ]]; then
    HOST_ARCH="ARMV81"
    DEVICE_ARCH="AMPERE80"
    MPI_NUM_PROCS=2
    NODE_SLICE=2
  elif [[ "$ARGS" == *"arm-hopper"* ]]; then
    HOST_ARCH="ARMV81"
    DEVICE_ARCH="HOPPER90"
    MPI_NUM_PROCS=1
    NODE_SLICE=1
  elif [[ "$ARGS" == *"ampere"* ]]; then
    HOST_ARCH="ZEN3"
    DEVICE_ARCH="AMPERE80"
    MPI_NUM_PROCS=2
    NODE_SLICE=2
  elif [[ "$ARGS" == *"volta"* ]]; then
    HOST_ARCH="HSW"
    DEVICE_ARCH="VOLTA70"
    MPI_NUM_PROCS=1
    # Some nodes have 2 GPUs, be conservative
    NODE_SLICE=2
  elif [[ "$ARGS" == *"knl"* ]]; then
    HOST_ARCH="KNL"
    MPI_NUM_PROCS=1
    # 4-way SMT, not 2
    NODE_SLICE=2
  elif [[ "$ARGS" == *"hsw"* ]]; then
    HOST_ARCH="HSW"
    MPI_NUM_PROCS=1
    NODE_SLICE=1
  elif [[ "$ARGS" == *"mi250"* ]]; then
    HOST_ARCH=ZEN3
    DEVICE_ARCH=VEGA90A
    MPI_NUM_PROCS=8
    NODE_SLICE=16
  else
    echo "Must specify an architecture on Darwin!"
    exit
  fi

  # Runtime
  MPI_EXE="mpirun"
  # Lead MPI to water
  MPI_EXTRA_ARGS="--map-by ppr:${MPI_NUM_PROCS}:node:pe=$(($NPROC / $NODE_SLICE))"
fi
