
# BP's machines

if [[ "$HOST" == "toolbox" ]]; then
  HOST=fermium
fi

if [[ $HOST == "fermium" ]]; then
  module load nvhpc
  HOST_ARCH="AMDAVX"
  DEVICE_ARCH="TURING75"

  PREFIX_PATH="$HOME/libs/hdf5-nvhpc"
  export NVCC_WRAPPER_DEFAULT_COMPILER=nvc++
  # My CUDA installs are a bit odd
  EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/include/cuda $EXTRA_FLAGS"
fi

if [[ $HOST == "ferrum" ]]; then
  HOST_ARCH="HSW"
  DEVICE_ARCH="INTEL_GEN"
  PREFIX_PATH="$HOME/libs/hdf5-oneapi"
fi

if [[ $HOST == "cinnabar"* ]]; then
  if [[ "$*" == *"cuda"* ]]; then
    HOST_ARCH="HSW"
    DEVICE_ARCH="KEPLER35"

    # To use NVHPC:
    #export NVCC_WRAPPER_DEFAULT_COMPILER=nvc++
    #PREFIX_PATH="$HOME/libs/hdf5-nvhpc"
    #export CXXFLAGS="-mp"
    #HOST_ARCH="SNB" # Kokkos doesn't detect/set -tp=haswell for nvc++

    #export NVCC_WRAPPER_DEFAULT_COMPILER=clang++
    #export CXXFLAGS="-allow-unsupported-compiler"

    # NVHPC CUDA
    #export CUDA_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda
    #EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/21.5/cuda/include/ $EXTRA_FLAGS"
    # System CUDA
    EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/include/cuda $EXTRA_FLAGS"
    # This makes Nvidia chill about old GPUs, but requires a custom nvcc_wrapper
    #export CXXFLAGS="-Wno-deprecated-gpu-targets"
  else
    HOST_ARCH="HSW"
    DEVICE_ARCH="KEPLER35"
    PREFIX_PATH="$HOME/libs/hdf5-oneapi"
  fi
fi
