
if [[ $HOST == "lmc.astro.illinois.edu" ]]; then
  # LMC: Haswell
  HOST_ARCH="HSW"
  # When we can compile HDF5 successfully
  #PREFIX_PATH="$HOME/libs/hdf5"
  # Until then, disable MPI & static HDF5 as bad versions are installed
  PREFIX_PATH=
  EXTRA_FLAGS="-DPARTHENON_DISABLE_MPI=ON"
elif [[ $HOST == *".astro.illinois.edu" ]]; then
  if [[ $HOST == "bh29"* ]]; then
    # BH29: Zen2 AMD EPYC 7742
    HOST_ARCH="ZEN2"
  else
    # Other machines are Skylake
    HOST_ARCH="SKX"
  fi

  # Compile our own HDF5 by default
  PREFIX_PATH="$SOURCE_DIR/external/hdf5"

  if [[ $ARGS == *"icc"* ]]; then
    # Intel ICC
    module purge
    source /opt/intel/oneapi/setvars.sh
    C_NATIVE="icc"
    CXX_NATIVE="icpc"

  elif [[ $ARGS == *"aocc"* ]]; then
    # AMD AOCC (BH29 only)
    source /opt/AMD/aocc-compiler-3.1.0.sles15/setenv_AOCC.sh
    C_NATIVE="clang"
    CXX_NATIVE="clang++"

  else
    # GNU GCC
    if [[ $HOST == "bh29"* ]]; then
      # Older GCC has no flag for ZEN2
      HOST_ARCH="ZEN"
    fi
    module load gnu mpich phdf5
    # System HDF5 location
    PREFIX_PATH="$MPI_DIR"
  fi
fi
# BH29 additions
if [[ $HOST == "bh29.astro.illinois.edu" ]]; then
  HOST_ARCH="ZEN1"

  # AOCC Requires system libc++, like icpx
  #source /opt/AMD/aocc-compiler-3.0.0/setenv_AOCC.sh
  #CXX_NATIVE="clang++"
fi
