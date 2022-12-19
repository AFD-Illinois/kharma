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
