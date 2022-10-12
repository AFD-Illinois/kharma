# LANL Darwin.  A little bit of everything

# Must list which node you're compiling for:
# ampere for AMD/NVIDIA A100 nodes
# Later:
# volta for x86/volta, either shared-gpu Titan V or volta-x86 queue
# arm-nv to compile for devkit ARM/NVIDIA nodes

if [[ $HOSTNAME == "cn"* || $HOSTNAME == "darwin"* ]]; then
  module purge
  module load cmake

  # Always our own HDF5
  PREFIX_PATH="$SOURCE_DIR/external/hdf5"

  if [[ "$ARGS" == *"arm-nv"* ]]; then
    echo "No"  
  elif [[ "$ARGS" == *"ampere"* ]]; then
    HOST_ARCH="ZEN3"
    DEVICE_ARCH="AMPERE80"
    module load nvhpc/22.7 cuda/11.7.0
    C_NATIVE="nvc"
    CXX_NATIVE="nvc++"
    # New NVHPC doesn't like CUDA_HOME
    export NVHPC_CUDA_HOME="$CUDA_HOME"
    unset CUDA_HOME
  else
    echo "No"
  fi

  # Runtime
  MPI_EXE="mpirun"
  MPI_NUM_PROCS=4
  MPI_EXTRA_ARGS="--map-by ppr:4:node:pe=32"
fi
