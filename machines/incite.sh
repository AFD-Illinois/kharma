
# INCITE resources
if [[ $HOST == *".summit.olcf.ornl.gov" ]]; then
  HOST_ARCH="POWER9"
  DEVICE_ARCH="VOLTA70"
  # Avoid sysadmin's wrath
  if [[ $(hostname) == "login"* ]]; then
    NPROC=16
  fi

  # All of these tested with Spectrum MPI 10.4.0.3
  if [[ "$ARGS" == *"gcc"* ]]; then
    module load gcc
    module load cuda
    PREFIX_PATH="$HOME/libs/hdf5-gcc10-spectrum"
  elif [[ "$ARGS" == *"xl"* ]]; then
    # xlC: OpenMP CXX problems
    module load xl
    module load cuda
    C_NATIVE='xlc'
    CXX_NATIVE='xlc++'
    export NVCC_WRAPPER_HOST_EXTRA_FLAGS='-O3 -qmaxmem=-1'
    export NVCC_WRAPPER_CUDA_EXTRA_FLAGS='-O3 -Xcompiler -qmaxmem=-1'
    #PREFIX_PATH="/sw/summit/hdf5/1.10.6_align/xl/16.1.1-5/"
  else
    # Use nvc++ compiler in NVHPC
    module unload cuda
    module load nvhpc/21.9
    C_NATIVE="nvc"
    CXX_NATIVE="nvc++"
    export CXXFLAGS="-mp"
    PREFIX_PATH="/gpfs/alpine/proj-shared/ast171/libs/hdf5-nvhpc-21.9"
  fi
fi

if [[ $HOST == *".alcf.anl.gov" ]]; then
  if [[ "$ARGS" == *"cuda"* ]]; then
    module purge
    module load Core/StdEnv cmake
    module load nvhpc/21.7
    #module load nvhpc
    module load openmpi
    #module load hdf5
    HOST_ARCH="AMDAVX"
    DEVICE_ARCH="AMPERE80"

    #CXXFLAGS="-mp"
    C_NATIVE="gcc"
    CXX_NATIVE="g++"
    #export CXXFLAGS="-g -pg"

    EXTRA_FLAGS="-DCUDAToolkit_ROOT_DIR=/soft/hpc-sdk/Linux_x86_64/21.7/cuda/11.4/ $EXTRA_FLAGS"
    EXTRA_FLAGS="-DCUDAToolkit_BIN_DIR=/soft/hpc-sdk/Linux_x86_64/21.7/cuda/11.4/bin $EXTRA_FLAGS"
    EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/soft/hpc-sdk/Linux_x86_64/21.7/cuda/11.4/include $EXTRA_FLAGS"
    PREFIX_PATH="$HOME/libs/hdf5-gcc-openmpi"
    #PREFIX_PATH="/soft/thetagpu/hpc-sdk/Linux_x86_64/21.7/"
  else
    echo "Compiling for KNL"
    HOST_ARCH="KNL"
    PREFIX_PATH="$MPICH_DIR"
  fi
fi
