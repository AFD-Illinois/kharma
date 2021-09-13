
# TACC resources
# Generally you want latest Intel/IMPI/phdf5 modules,
# On longhorn use gcc7, mvapich2-gdr, and manually-compiled PHDF5

if [[ $HOST == *".frontera.tacc.utexas.edu" ]]; then
  HOST_ARCH="SKX"
fi

if [[ $HOST == *".stampede2.tacc.utexas.edu" ]]; then
  NPROC=16
  if [[ "$*" == *"skx"* ]]; then
    HOST_ARCH="SKX"
  else
    HOST_ARCH="KNL"
    EXTRA_FLAGS="-DFUSE_FLUX_KERNELS=OFF -DFUSE_EMF_KERNELS=OFF -DFUSE_FLOOR_KERNELS=OFF $EXTRA_FLAGS"
  fi
fi

if [[ $HOST == *".longhorn.tacc.utexas.edu" ]]; then
  HOST_ARCH="POWER9"
  DEVICE_ARCH="VOLTA70"
  PREFIX_PATH="$HOME/libs/hdf5-gcc7-mvapich2"
fi
