# Illinois campus cluster                     
if [[ $HOST == *".campuscluster.illinois.edu" ]]; then               
  HOST_ARCH="SKX"
  DEVICE_ARCH="VOLTA70"

  MPI_EXE="mpirun"
  MPI_NUM_PROCS=2
  MPI_EXTRA_ARGS=""
  KOKKOS_NUM_DEVICES=2

  C_NATIVE="/usr/local/gcc/8.2.0/bin/gcc"
  CXX_NATIVE="/usr/local/gcc/8.2.0/bin/g++"
  export OMPI_CC="/usr/local/gcc/8.2.0/bin/gcc"
  export OMPI_CXX="/usr/local/gcc/8.2.0/bin/g++"

  #C_NATIVE="nvc"
  #CXX_NATIVE="nvc++"

  PREFIX_PATH="$HOME/libs/hdf5-nvhpc;/usr/local/cuda/11.1"
  #PREFIX_PATH="$HOME/libs/hdf5-gcc7-cuda11-openmpi/"
  #PREFIX_PATH="$HOME/libs/hdf5-gcc7-cuda11-mvapich2/"                                                       
  EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda/11.1/include $EXTRA_FLAGS"
fi
