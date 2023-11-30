
# Config for OLCF Frontier

if [[ $HOST == *".frontier.olcf.ornl.gov" ]]
then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=VEGA90A

  MPI_EXE=srun
  NPROC=64

  if [[ $ARGS == *"hip"* ]]; then
    # HIP compile for AMD GPUs

    if [[ $ARGS == *"cray"* ]]; then
      module load PrgEnv-cray
      module load craype-accel-amd-gfx90a
      module load amd-mixed
    else
      module load PrgEnv-amd
      module load craype-accel-amd-gfx90a
    fi

    module load cray-hdf5-parallel

    if [[ $ARGS == *"hipcc"* ]]; then
      # TODO LINK MPI RIGHT
      CXX_NATIVE=hipcc
      C_NATIVE=hipcc
      export CXXFLAGS="-I$CRAY_HDF5_PARALLEL_PREFIX/include -L$CRAY_HDF5_PARALLEL_PREFIX/lib -l:libhdf5_parallel.a"
      #export PATH="$CRAY_HDF5_PARALLEL_PREFIX/bin:$PATH"
    else
      CXX_NATIVE=CC
      C_NATIVE=cc
      export CXXFLAGS="-noopenmp -mllvm -amdgpu-function-calls=false $CXXFLAGS"
    fi

    # Runtime
    MPI_NUM_PROCS=8
    MPI_EXTRA_ARGS="-c1 --gpus-per-node=8 --gpu-bind=closest"
    export MPICH_GPU_SUPPORT_ENABLED=1

   # Old workaround, for non-GPU MPI only!
   #export MPICH_SMP_SINGLE_COPY_MODE=NONE
  else
    # CPU Compile
    # TODO -c etc etc
    MPI_NUM_PROCS=1
  fi
fi
