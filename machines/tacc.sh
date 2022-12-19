
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
    EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON"
  else
    HOST_ARCH="KNL"
    EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON -DFUSE_FLUX_KERNELS=OFF -DFUSE_FLOOR_KERNELS=OFF $EXTRA_FLAGS"
  fi
fi

if [[ $HOST == *".longhorn.tacc.utexas.edu" ]]; then
  HOST_ARCH="POWER9"
  DEVICE_ARCH="VOLTA70"

  # Volta is still touchy about kernel sizes
  EXTRA_FLAGS="-DFUSE_FLUX_KERNELS=OFF $EXTRA_FLAGS"

  # TACC does NOT ship PHDF5 on Longhorn
  PREFIX_PATH="$SOURCE_DIR/external/hdf5"

  if [[ $ARGS == *"gcc9"* ]]; then
    module load gcc/9.1.0 cuda spectrum_mpi
    # GCC9 has problems with POWER9's float128
    export NVCC_WRAPPER_HOST_EXTRA_FLAGS="-mno-float128"
    export NVCC_WRAPPER_CUDA_EXTRA_FLAGS="-Xcompiler -mno-float128"
    CXX_NATIVE='g++'
    C_NATIVE='gcc'
  elif [[ $ARGS == *"gcc7"* ]]; then
    # Fun chasing the dragon
    module load gcc/7.3.0 cuda mvapich2-gdr
    CXX_NATIVE='g++'
    C_NATIVE='gcc'
  else # THESE ARE NOT WORKING
    # "Load" the GCC 9.1 module to get newer libc
    # LD_LIBRARY_PATH must also be set when running!
    export PATH="/opt/apps/gcc/9.1.0/bin:$PATH"
    export LIBRARY_PATH="/opt/apps/gcc/9.1.0/lib:/opt/apps/gcc/9.1.0/lib64:$LIBRARY_PATH"
    export LD_LIBRARY_PATH="/opt/apps/gcc/9.1.0/lib:/opt/apps/gcc/9.1.0/lib64:$LD_LIBRARY_PATH"
    export INCLUDE="/opt/apps/gcc/9.1.0/include:$INCLUDE"
    export PREFIX_PATH="/opt/apps/gcc/9.1.0;$PREFIX_PATH"
    if [[ $ARGS == *"pgi"* ]]; then
      module load pgi cuda spectrum_mpi
      CXX_NATIVE='pgc++'
      C_NATIVE='pgcc'
      export NVCC_WRAPPER_HOST_EXTRA_FLAGS="-Mnostdinc"
    elif [[ $ARGS == *"xl"* ]]; then
      module load xl cuda mvapich2-gdr
      C_NATIVE='xlc'
      CXX_NATIVE='xlC'
      export NVCC_WRAPPER_HOST_EXTRA_FLAGS="-nostdinc++"
    fi
  fi
fi

