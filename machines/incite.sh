
# INCITE resources
if [[ $HOST == *".alcf.anl.gov" ]]; then
  if [[ "$*" == *"cuda"* ]]; then
    module purge
    module load Core/StdEnv cmake
    module load nvhpc/21.7
    #module load nvhpc
    module load openmpi
    #module load hdf5
    HOST_ARCH="AMDAVX"
    DEVICE_ARCH="AMPERE80"

    #CXXFLAGS="-mp"
    export CC="gcc"
    export NVCC_WRAPPER_DEFAULT_COMPILER='g++'
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
if [[ $HOST == *".summit.olcf.ornl.gov" ]]; then
  HOST_ARCH="POWER9"
  DEVICE_ARCH="VOLTA70"

  # nvc++: nvcc *refuses* to use it and falls back to system GCC 4.8.5...
  #export NVCC_WRAPPER_DEFAULT_COMPILER='nvc++'
  #PREFIX_PATH="$HOME/libs/hdf5-nvhpc-21.2"

  # GCC 10.2, needs CUDA 11 probably
  #PREFIX_PATH="$HOME/libs/hdf5-gcc10-spectrum"

  # TODO make GCC 8 not crash
  #CXXFLAGS="-mno-float128 $CXXFLAGS"

  # GCC 6.4
  PREFIX_PATH="/sw/summit/hdf5/1.10.6_align/gcc/6.4.0/"

  # xlC: OpenMP CXX problems
  #export NVCC_WRAPPER_DEFAULT_COMPILER='xlC'
  #PREFIX_PATH="/sw/summit/hdf5/1.10.6_align/xl/16.1.1-5/"
fi
