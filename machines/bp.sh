
# BP's machines

if [[ $HOST == "cheshire"* ]]; then
  HOST_ARCH="HSW"
  DEVICE_ARCH="PASCAL61"
  export OMP_NUM_THREADS=24

  if [[ "$ARGS" == *"cuda"* ]]; then
    # NVHPC. Compiler is chosen automatically now
    module load nvhpc
    NPROC=8 # so much memory
  else
    # Intel oneAPI
    module load compiler mpi/2021
    NPROC=24
  fi
  # Even CPU kharma is unkillable without this
  MPI_EXE=mpirun
fi

if [[ $HOST == "toolbox"* || $HOST == "nvhpc"* ]]; then
  METAL_HOSTNAME=$(cat ~/.config/hostname)
fi

if [[ $METAL_HOSTNAME == "fermium" ]]; then
  HOST_ARCH="AMDAVX"
  # We patch Kokkos to make this gfx1101==rx7800xt
  DEVICE_ARCH="AMD_GFX1100"
  # MPI & Kokkos separately dislike running the bin alone
  #MPI_EXE=mpirun
  NPROC=24

  if [[ "$ARGS" == *"hip"* ]]; then
    # AMD for GPUs (this will be run in container, no modules)
    C_NATIVE=hipcc
    CXX_NATIVE=hipcc
  else
    # AMD for CPUs
    module load aocc-compiler-4.1.0 mpi
    CXX_NATIVE=clang++
    C_NATIVE=clang
  fi
fi

if [[ $METAL_HOSTNAME == "ferrum" ]]; then
  HOST_ARCH="HSW"
  NPROC=6

  if [[ "$ARGS" == *"gcc"* ]]; then
    module load mpi/mpich-x86_64
    C_NATIVE="gcc"
    CXX_NATIVE="g++"
  elif [[ "$ARGS" == *"icc"* ]]; then
    # Intel compiler
    module purge
    module load compiler mpi
    C_NATIVE="icc"
    CXX_NATIVE="icpc"
  else
    # Intel SYCL implementation "DPC++"
    module purge
    module load compiler mpi
    C_NATIVE="icx"
    CXX_NATIVE="icpx"
  fi
fi

if [[ $HOST == "cinnabar"* ]]; then
  # All my MPI stacks can use this as the call
  MPI_EXE=mpirun

  module purge # Handle modules inside this script
  HOST_ARCH="HSW" # This won't change
  DEVICE_ARCH="TURING75"

  # Runtime
  MPI_NUM_PROCS=1

  # TODO container:
  # module swap nvhpc-hpcx nvhpc

  if [[ "$ARGS" == *"cuda"* ]]; then
    # Runtime
    MPI_EXTRA_ARGS="--map-by ppr:1:numa:pe=14"

    # System CUDA path
    EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/include/cuda $EXTRA_FLAGS"

    # Switch between g++/NVC++:
    if [[ "$ARGS" == *"gcc"* ]]; then
      module load mpi/mpich-x86_64 nvhpc-nompi
      C_NATIVE="gcc"
      CXX_NATIVE="g++"
    else
      module load nvhpc
      PREFIX_PATH="$HOME/libs/hdf5-nvhpc"
      C_NATIVE="nvc"
      CXX_NATIVE="nvc++"
      #export CXXFLAGS="-mp"
    fi
  else
    if [[ "$ARGS" == *"gcc"* ]]; then
      # GCC
      module load mpi/mpich-x86_64
      C_NATIVE="gcc"
      CXX_NATIVE="g++"
    else
      # Intel by default
      module load compiler mpi
      C_NATIVE="icx"
      CXX_NATIVE="icpx"
    fi
  fi
fi
