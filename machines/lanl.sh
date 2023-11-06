# LANL Machines: HPC and IC

# Darwin.  A little bit of everything
# Must list which node you're compiling for:
# ampere for AMD/NVIDIA A100 nodes
# volta for x86/volta of all kinds
# Not working yet:
# arm-nv to compile for devkit ARM/NVIDIA nodes
if [[ $HOSTNAME == "darwin"* ]]; then
  module purge
  module load cmake

  # Always our own HDF5
  # Run ""./make.sh <usual args> hdf5" to build it
  PREFIX_PATH="$SOURCE_DIR/external/hdf5"

  # These are 
  if [[ "$ARGS" == *"arm-nv"* ]]; then
    HOST_ARCH="ARMV81"
    DEVICE_ARCH="AMPERE80"
    module load nvhpc/22.7 cuda/11.7.0
    C_NATIVE="nvc"
    CXX_NATIVE="nvc++"
    # New NVHPC doesn't like CUDA_HOME
    export NVHPC_CUDA_HOME="$CUDA_HOME"
    unset CUDA_HOME 
  elif [[ "$ARGS" == *"ampere"* ]]; then
    HOST_ARCH="ZEN3"
    DEVICE_ARCH="AMPERE80"
    module load nvhpc/22.7 cuda/11.7.0
    C_NATIVE="nvc"
    CXX_NATIVE="nvc++"
    # New NVHPC doesn't like CUDA_HOME
    export NVHPC_CUDA_HOME="$CUDA_HOME"
    unset CUDA_HOME
  elif [[ "$ARGS" == *"volta"* ]]; then
    HOST_ARCH="HSW"
    DEVICE_ARCH="VOLTA70"
    module load nvhpc/22.7 cuda/11.7.0
    C_NATIVE="nvc"
    CXX_NATIVE="nvc++"
    # New NVHPC doesn't like CUDA_HOME
    export NVHPC_CUDA_HOME="$CUDA_HOME"
    unset CUDA_HOME
  else
    echo "No target arch specified: must list a target arch for Darwin"
    exit
  fi

  # Runtime
  MPI_EXE="mpirun"
  MPI_NUM_PROCS=2
  MPI_EXTRA_ARGS="--map-by ppr:4:node:pe=8"
fi

# Chicoma
if [[ "$HOST" == "ch-fe"* || "$HOST" == "nid"* ]]; then
  HOST_ARCH="ZEN2"

  # Cray environments get confused easy
  # Make things as simple as possible
  module purge
  export CRAY_CPU_TARGET="x86-64"
  # Kokkos claims to need this but doesn't?
  #export CRAYPE_LINK_TYPE="dynamic"
  # I think this is for old true CLE systems, not Chicoma
  #EXTRA_FLAGS="-DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment $EXTRA_FLAGS"
  if [[ "$ARGS" == *"cuda"* ]]; then
    DEVICE_ARCH="AMPERE80"
    # No GPU-aware MPI (WTF)
    EXTRA_FLAGS="-DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON $EXTRA_FLAGS"
    # Runtime
    MPI_NUM_PROCS=4
    if [[ "$ARGS" == *"cray"* ]]; then
      # Cray's "nvidia" environment
      # Autodetect still ends up using nvc/++
      module load PrgEnv-nvidia cmake
    elif [[ "$ARGS" == *"ctk"* ]]; then
      module load PrgEnv-gnu gcc/11.2.0 cudatoolkit cmake
    elif [[ "$ARGS" == *"gnu"* ]]; then
      module load PrgEnv-gnu cpe-cuda cuda cmake
    elif [[ "$ARGS" == *"intel"* ]]; then
      module load PrgEnv-intel cmake
    else
      module load PrgEnv-nvhpc cmake
    fi
  else
    module load PrgEnv-aocc cmake
  fi

  # Runtime
  MPI_NUM_PROCS=4
  MPI_EXE=srun
fi
