
# Config for OLCF Frontier

if [[ $HOST == *".frontier.olcf.ornl.gov" ]]
then
  HOST_ARCH=ZEN3
  DEVICE_ARCH=VEGA90A

  if [[ $ARGS == *"hip"* ]]; then
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
    MPI_NUM_PROCS=4
  else
    # CPU Compile
    MPI_NUM_PROCS=1
  fi
fi
