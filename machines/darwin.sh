# LANL Darwin.  A little bit of everything

# Must list which node you're compiling for:
# ampere for AMD/NVIDIA A100 nodes
# volta for x86/volta of all kinds
# Not working yet:
# arm-nv to compile for devkit ARM/NVIDIA nodes

if [[ $HOSTNAME == "cn"* || $HOSTNAME == "darwin"* ]]; then
  module purge
  module load cmake

  # Always our own HDF5
  # Run ""./make.sh <usual args> hdf5" to build it
  PREFIX_PATH="$SOURCE_DIR/external/hdf5"

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
      module load nvhpc/23.3 cuda/11.7.0
      C_NATIVE="nvc"
      CXX_NATIVE="nvc++"
      # New NVHPC doesn't like CUDA_HOME
      export NVHPC_CUDA_HOME="$CUDA_HOME"
      unset CUDA_HOME
    fi
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

  # These are orthogonal to above.
  # Just don't compile for an nv arch without "cuda"
  NPROC=$(($(nproc) / 2)) # TODO robust?
  if [[ "$ARGS" == *"arm-nv"* ]]; then
    HOST_ARCH="ARMV81"
    DEVICE_ARCH="AMPERE80"
    MPI_NUM_PROCS=2
    MPI_EXTRA_ARGS="--map-by ppr:2:node:pe=$(($NPROC / 2))"
  elif [[ "$ARGS" == *"ampere"* ]]; then
    HOST_ARCH="ZEN3"
    DEVICE_ARCH="AMPERE80"
    MPI_NUM_PROCS=2
    MPI_EXTRA_ARGS="--map-by ppr:2:node:pe=$(($NPROC / 2))"
  elif [[ "$ARGS" == *"volta"* ]]; then
    HOST_ARCH="HSW"
    DEVICE_ARCH="VOLTA70"
    MPI_NUM_PROCS=1
    # Some nodes only have 1 GPU but be conservative
    MPI_EXTRA_ARGS="--map-by ppr:2:node:pe=$(($NPROC / 2))"
  else
    HOST_ARCH="HSW"
    MPI_NUM_PROCS=1
    MPI_EXTRA_ARGS="--map-by ppr:1:node:pe=$(($NPROC))"
  fi

  # Runtime
  MPI_EXE="mpirun"
fi
