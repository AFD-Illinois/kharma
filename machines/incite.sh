
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

  # Summit *hates* C++17.
  # Use GCC with 14
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
    module load gcc cuda hdf5
    C_NATIVE='gcc'
    CXX_NATIVE='g++'
  fi
fi

if [[ $HOST == *".alcf.anl.gov" ]]; then
  HOST_ARCH=HSW
  DEVICE_ARCH=AMPERE80
  module load PrgEnv-gnu
  module load cudatoolkit-standalone
  #module load PrgEnv-nvhpc
  module load cray-hdf5-parallel cmake
  #export CRAY_CPU_TARGET=x86-64

  # Correct some vars set by default PrgEnv-nvhpc
  #unset CC
  #unset F77
  #unset CXX
  #unset FC
  #unset F90

  EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON $EXTRA_FLAGS"
fi

