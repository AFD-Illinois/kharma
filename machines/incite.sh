
# INCITE resources
if [[ $HOST == *".summit.olcf.ornl.gov" ]]; then
  HOST_ARCH="POWER9"
  DEVICE_ARCH="VOLTA70"
  # Avoid sysadmin's wrath
  NPROC=8
  # Runtime options for one-node test runs
  MPI_EXE="jsrun --smpiargs="-gpu" -r 6 -a 1 -g 1 -c 6 -d packed -b packed:6"
  OMP_NUM_THREADS=24
  KOKKOS_NUM_DEVICES=1
  MPI_NUM_PROCS=6

  # ONLY GCC WORKS: There are C++17 compile issues with most other combos/stacks
  # Tested with Spectrum MPI 10.4.0.3
  module load cmake
  if [[ "$ARGS" == *"xl"* ]]; then
    # xlC: OpenMP CXX problems
    #module load xl cuda
    C_NATIVE='xlc'
    CXX_NATIVE='xlc++'
    export NVCC_WRAPPER_HOST_EXTRA_FLAGS='-O3 -qmaxmem=-1'
    export NVCC_WRAPPER_CUDA_EXTRA_FLAGS='-O3 -Xcompiler -qmaxmem=-1'
    #PREFIX_PATH="/sw/summit/hdf5/1.10.6_align/xl/16.1.1-5/"
  elif [[ "$ARGS" == *"nvhpc"* ]]; then
    # Use nvc++ compiler in NVHPC
    module load cuda/11.5.2 nvhpc/22.5 spectrum-mpi hdf5/1.10.7

    C_NATIVE="nvc"
    CXX_NATIVE="nvc++"
    export CXXFLAGS="-mp"
    PREFIX_PATH="/gpfs/alpine/proj-shared/ast171/libs/hdf5-nvhpc-21.9"
  else
    # Use default GCC
    module load gcc/11.1.0 hdf5/1.10.7 cuda/11.5.2
    C_NATIVE='gcc'
    CXX_NATIVE='g++'
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
