# Illinois campus cluster                     
if [[ $HOST == *".campuscluster.illinois.edu" ]]; then               
  HOST_ARCH="SKX"
  DEVICE_ARCH="VOLTA70"

  export NVCC_WRAPPER_DEFAULT_COMPILER='/usr/local/gcc/8.2.0/bin/g++'
  PREFIX_PATH="$HOME/libs/hdf5-nvhpc;/usr/local/cuda/11.1"
  #PREFIX_PATH="$HOME/libs/hdf5-gcc7-cuda11-openmpi/"
  #PREFIX_PATH="$HOME/libs/hdf5-gcc7-cuda11-mvapich2/"                                                       
  EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda/11.1/include $EXTRA_FLAGS"
fi
