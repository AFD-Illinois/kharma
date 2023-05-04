
# INCITE resources

# ALCF: Polaris
if [[ $HOST == *".polaris.alcf.anl.gov" ]]; then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=AMPERE80

  module purge
  if [[ $ARGS == *"nvhpc233"* ]]; then
    # DOES NOT WORK: "CUDA 11.4 not installed with this NVHPC"
    module use /soft/compilers/nvhpc/modulefiles
    module load PrgEnv-nvhpc nvhpc/23.3
    # Guide new NVHPC to a working CUDA?
    # export NVHPC_CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/11.4"
    # export NVHPC_DEFAULT_CUDA=11.4
    # export NVCC_WRAPPER_CUDA_EXTRA_FLAGS="-gpu=cuda11.4"
    # EXTRA_FLAGS="-DCUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/11.4 $EXTRA_FLAGS"
  elif [[ $ARGS == *"nvhpc219"* ]]; then
    # DOES NOT WORK: compile errors in pmmintrin.h & AVX512 intrinsics headers
    module load PrgEnv-nvhpc
    # Correct some vars set by default PrgEnv-nvhpc
    unset CC CXX F77 F90 FC
    # Try not to require intrinsics?
    #HOST_ARCH=BDW
  elif [[ $ARGS == *"gcc"* ]]; then
    module load PrgEnv-gnu
    module load cudatoolkit-standalone
  else
    module load PrgEnv-nvhpc nvhpc/23.1
  fi
  # Common modules
  module load cray-hdf5-parallel cmake
  module load craype-accel-nvidia80
  
  # Since we ran 'module purge',
  # The Cray wrappers will warn unless we set this
  export CRAY_CPU_TARGET=x86-64
  # TODO(BSP) need to set CRAYPE_LINK_TYPE=dynamic long-term?

  EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON $EXTRA_FLAGS"
fi
