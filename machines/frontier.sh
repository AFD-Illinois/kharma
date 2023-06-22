
# Config for OLCF Frontier

if [[ $HOST == *".frontier.olcf.ornl.gov" ]]
then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=VEGA90A

  MPI_EXE=srun

  if [[ $ARGS == *"hip"* ]]; then
    # HIP compile for AMD GPUs
    if [[ $ARGS == *"hipcc"* ]]; then
      module load PrgEnv-cray amd-mixed
      CXX_NATIVE=hipcc
      C_NATIVE=cc
      export CXXFLAGS="-fopenmp $CXXFLAGS"
    else
      module load PrgEnv-amd cray-hdf5-parallel
      CXX_NATIVE=CC
      C_NATIVE=cc
      export CXXFLAGS="-fopenmp -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false $CXXFLAGS"
    fi

    # Runtime
    MPI_NUM_PROCS=8
    MPI_EXTRA_ARGS="-c2 --gpus-per-node=8 --gpu-bind=closest"
    export MPICH_SMP_SINGLE_COPY_MODE=NONE
  else
    # CPU Compile
    # TODO -c etc etc
    MPI_NUM_PROCS=1
  fi
fi
