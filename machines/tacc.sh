
# TACC resources
# Generally you want latest Intel/IMPI/phdf5 modules,
# On longhorn use gcc7, mvapich2-gdr, and manually-compiled PHDF5

if [[ $HOST == *".frontera.tacc.utexas.edu" ]]; then
  HOST_ARCH="SKX"
fi

if [[ $HOST == *".stampede2.tacc.utexas.edu" ]]; then
  NPROC=16
  if [[ "$ARGS" == *"skx"* ]]; then
    HOST_ARCH="SKX"
  else
    HOST_ARCH="KNL"
    EXTRA_FLAGS="-DFUSE_FLUX_KERNELS=OFF -DFUSE_EMF_KERNELS=OFF -DFUSE_FLOOR_KERNELS=OFF $EXTRA_FLAGS"
  fi
fi

if [[ $HOST == *".longhorn.tacc.utexas.edu" ]]; then
  HOST_ARCH="POWER9"
  DEVICE_ARCH="VOLTA70"
  PREFIX_PATH="$HOME/libs/hdf5-gcc9-spectrum"
  export NVCC_WRAPPER_HOST_EXTRA_FLAGS="-mno-float128"
  export NVCC_WRAPPER_CUDA_EXTRA_FLAGS="-Xcompiler -mno-float128"
fi

