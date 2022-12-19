
# BP's machines

if [[ $HOST == "cheshire"* ]]; then
  HOST_ARCH="HSW"
  DEVICE_ARCH="PASCAL61"
  export OMP_NUM_THREADS=24

  if [[ "$ARGS" == *"cuda"* ]]; then
    # NVHPC. Compiler is chosen automatically now
    module load nvhpc
  else
    # Intel oneAPI
    module load compiler mpi/2021
  fi

  NPROC=24
  MPI_EXE=mpirun
fi

if [[ $HOST == "toolbox"* || $HOST == "nvhpc"* ]]; then
  METAL_HOSTNAME=$(cat ~/.config/hostname)
fi

if [[ $METAL_HOSTNAME == "fermium" ]]; then
  HOST_ARCH="AMDAVX"
  DEVICE_ARCH="TURING75"
  # Nvidia MPI hangs unless I do this
  MPI_EXE=mpirun

  if [[ "$ARGS" == *"cuda"* ]]; then
    module purge
    module load nvhpc
    PREFIX_PATH="$HOME/libs/hdf5-nvhpc"
    MPI_NUM_PROCS=1

    if [[ "$ARGS" == *"gcc"* ]]; then
      C_NATIVE=gcc
      CXX_NATIVE=g++
    else
      C_NATIVE=nvc
      CXX_NATIVE=nvc++
      export CFLAGS="-mp"
      export CXXFLAGS="-mp"
    fi
  else
    # To experiment with AMD NUMA
    #MPI_EXTRA_ARGS="--map-by ppr:2:socket:pe=12"
    #MPI_NUM_PROCS=2
    if [[ "$ARGS" == *"gcc"* ]]; then
      module purge
      #module load mpi/mpich-x86_64
      C_NATIVE=gcc
      CXX_NATIVE=g++
    elif [[ "$ARGS" == *"clang"* ]]; then
      module purge
      module load mpi/mpich-x86_64
      C_NATIVE=clang
      CXX_NATIVE=clang++
      PREFIX_PATH="$HOME/libs/hdf5-clang14"
    else
      module purge
      module load mpi/mpich-x86_64
      source /opt/AMD/aocc-compiler-3.2.0/setenv_AOCC.sh
      PREFIX_PATH="$HOME/libs/hdf5-aocc"
      C_NATIVE=clang
      CXX_NATIVE=clang++
    fi
  fi
fi

if [[ $METAL_HOSTNAME == "ferrum" ]]; then
  HOST_ARCH="HSW"
  DEVICE_ARCH="INTEL_GEN"
  NPROC=6

  if [[ "$ARGS" == *"gcc"* ]]; then
    module load mpi/mpich-x86_64
    C_NATIVE="gcc"
    CXX_NATIVE="g++"
  elif [[ "$ARGS" == *"icc"* ]]; then
    # Intel compiler
    module purge
    module load compiler mpi
    PREFIX_PATH="$HOME/libs/hdf5-oneapi"
  else
    # Intel SYCL implementation "DPC++"
    module purge
    module load compiler mpi
    PREFIX_PATH="$HOME/libs/hdf5-oneapi"
    C_NATIVE="icx"
    CXX_NATIVE="icpx"
  fi
fi

if [[ $HOST == "cinnabar"* ]]; then
  # All my MPI stacks can use this as the call
  MPI_EXE=mpirun

  module purge # Handle modules inside this script
  HOST_ARCH="HSW" # This won't change

  if [[ "$ARGS" == *"cuda"* ]]; then
    # Use NVHPC libraries (GPU-aware OpenMPI!)
    DEVICE_ARCH="KEPLER35"
    MPI_NUM_PROCS=2
    MPI_EXTRA_ARGS="--map-by ppr:1:numa:pe=14"

    # Quash warning about my old gpus
    export NVCC_WRAPPER_CUDA_EXTRA_FLAGS="-Wno-deprecated-gpu-targets"
    # System CUDA path
    EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/include/cuda $EXTRA_FLAGS"

    # Switch between g++/NVC++:
    if [[ "$ARGS" == *"gcc"* ]]; then
      module load mpi/mpich-x86_64 nvhpc-nompi
      C_NATIVE="gcc"
      CXX_NATIVE="g++"
      # Uses system GCC, which is old
      EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON $EXTRA_FLAGS"
    else
      module load nvhpc
      PREFIX_PATH="$HOME/libs/hdf5-nvhpc"
      C_NATIVE="nvc"
      CXX_NATIVE="nvc++"
      export CXXFLAGS="-mp"
    fi
  else
    MPI_NUM_PROCS=1
    if [[ "$ARGS" == *"gcc"* ]]; then
      # GCC
      module load mpi/mpich-x86_64
      C_NATIVE="gcc"
      CXX_NATIVE="g++"
      # Uses system GCC, which is old
      EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON $EXTRA_FLAGS"
    else
      # Intel by default
      module load compiler mpi
      PREFIX_PATH="$HOME/libs/hdf5-oneapi"
    fi
  fi
fi
