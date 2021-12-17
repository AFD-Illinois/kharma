
# BP's machines

# TODO toolbox break to discover enclosing hostname
if [[ "$HOST" == "toolbox" ]]; then
  HOST=ferrum
fi

if [[ $HOST == "fermium" ]]; then
  module purge
  module load nvhpc
  HOST_ARCH="AMDAVX"
  DEVICE_ARCH="TURING75"

  PREFIX_PATH="$HOME/libs/hdf5-nvhpc"
  export NVCC_WRAPPER_DEFAULT_COMPILER=nvc++
  # My CUDA installs are a bit odd
  EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/include/cuda $EXTRA_FLAGS"
fi

if [[ $HOST == "ferrum" ]]; then
  # Intel SYCL implementation "DPC++"
  module purge
  module load compiler mpi

  NPROC=6 # My kingdom for a RAM!

  HOST_ARCH="HSW"
  DEVICE_ARCH="INTEL_GEN"
  PREFIX_PATH="$HOME/libs/hdf5-oneapi"

  EXTRA_FLAGS="-DFUSE_FLUX_KERNELS=OFF -DFUSE_EMF_KERNELS=OFF -DFUSE_FLOOR_KERNELS=OFF $EXTRA_FLAGS"
fi

if [[ $HOST == "cinnabar"* ]]; then
  module purge # Handle modules inside this script

  if [[ "$*" == *"cuda"* ]]; then
    # Use NVHPC libraries (GPU-aware OpenMPI!)
    # but not nvc++ because Parthenon broke it
    HOST_ARCH="HSW"
    DEVICE_ARCH="KEPLER35"

    module load nvhpc
    PREFIX_PATH="$HOME/libs/hdf5-nvhpc"
    # Quash warning about my old gpus
    export NVCC_WRAPPER_CUDA_EXTRA_FLAGS="-Wno-deprecated-gpu-targets"

    # To use NVCC:
    if [[ "$*" == *"gcc"* ]]; then
      C_NATIVE="gcc"
      CXX_NATIVE="g++"
    else
      C_NATIVE="nvc"
      CXX_NATIVE="nvc++"
      export CXXFLAGS="-mp"
      #HOST_ARCH="SNB" # Kokkos doesn't detect/set -tp=haswell for nvc++
    fi

    # System CUDA
    EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/include/cuda $EXTRA_FLAGS"
  else
    # Intel
    module load compiler mpi
    HOST_ARCH="HSW"
    PREFIX_PATH="$HOME/libs/hdf5-oneapi"
  fi
fi
