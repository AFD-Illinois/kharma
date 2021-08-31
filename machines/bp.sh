
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

  EXTRA_FLAGS="-DFUSE_FLUX_KERNELS=OFF $EXTRA_FLAGS"
fi

if [[ $HOST == "cinnabar"* ]]; then
  module purge # Handle modules inside this script

  if [[ "$*" == *"clanggpu"* ]]; then
    # Ill-fated clang GPU experiment.  Requires CUDA 10.1 or older,
    # which do not play nice with my std::variant tricks
    module load mpi/mpich-x86_64
    HOST_ARCH="HSW"
    DEVICE_ARCH="KEPLER35"
    export CXXFLAGS="--cuda-path=/usr/local/cuda-10.1"
    export CUDA_HOME="/usr/local/cuda-10.1"
    EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-10.1/include $EXTRA_FLAGS"

  elif [[ "$*" == *"cuda"* ]]; then
    # Use NVHPC libraries (GPU-aware OpenMPI!)
    # but not nvc++ because Parthenon broke it
    HOST_ARCH="HSW"
    DEVICE_ARCH="KEPLER35"

    module load nvhpc
    PREFIX_PATH="$HOME/libs/hdf5-nvhpc"

    # To use NVCC:
    if [[ "$*" == *"nvcc"* ]]; then
      export NVCC_WRAPPER_DEFAULT_COMPILER=nvc++
      export CXXFLAGS="-mp"
      HOST_ARCH="SNB" # Kokkos doesn't detect/set -tp=haswell for nvc++
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
