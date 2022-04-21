# Illinois campus cluster
# Always compile on a compute node!
if [[ $HOST == *".campuscluster.illinois.edu" ]]; then               
  HOST_ARCH="SKX"
  DEVICE_ARCH="VOLTA70"

  MPI_EXE="mpirun"
  MPI_NUM_PROCS=2
  MPI_EXTRA_ARGS=""
  KOKKOS_NUM_DEVICES=2

  # Exactly this stack works. Experimentation is futile
  C_NATIVE="/usr/local/gcc/8.2.0/bin/gcc"
  CXX_NATIVE="/usr/local/gcc/8.2.0/bin/g++"
  export OMPI_CC="/usr/local/gcc/8.2.0/bin/gcc"
  export OMPI_CXX="/usr/local/gcc/8.2.0/bin/g++"

  PREFIX_PATH="$HOME/libs/hdf5-nvhpc;/usr/local/cuda/11.1"
  EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda/11.1/include $EXTRA_FLAGS"
fi
