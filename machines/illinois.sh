
# LMC is special
if [[ $HOST == "lmc.astro.illinois.edu" ]]; then
  #conda deactivate
  HOST_ARCH="HSW"
  # When we can compile HDF5 successfully
  #PREFIX_PATH="$HOME/libs/hdf5"
  # Until then, disable MPI & static HDF5 as bad versions are installed
  PREFIX_PATH=
  EXTRA_FLAGS="-DPARTHENON_DISABLE_MPI=ON"
# So is BH29
elif [[ $HOST == "bh29.astro.illinois.edu" ]]; then
  HOST_ARCH="ZEN2"

  # Compile our own HDF5
  PREFIX_PATH="$SOURCE_DIR/external/hdf5"

  if  [[ $ARGS == *"icc"* ]]; then
    source /opt/intel/oneapi/setvars.sh
    C_NATIVE="icc"
    CXX_NATIVE="icpc"
  elif [[ $ARGS == *"gcc"* ]]; then
    # Older GCC has no flag for ZEN2
    HOST_ARCH="ZEN1"
    # Modules?
  else
    # AOCC Requires system libstdc++
    PREFIX_PATH="/usr/lib64"
    source /opt/AMD/aocc-compiler-3.1.0.sles15/setenv_AOCC.sh
    C_NATIVE="clang"
    CXX_NATIVE="clang++"
  fi

elif [[ $HOST == *".astro.illinois.edu" ]]; then
  HOST_ARCH="SKX"
  #module purge

  # Uncomment to use full intel stack: MPI crashes
  #source /opt/intel/oneapi/setvars.sh
  #PREFIX_PATH="$HOME/libs/hdf5-oneapi/"

  # To load GNU stuff
  module load gnu mpich phdf5
  PREFIX_PATH="$MPI_DIR"

  # Add back just the intel compilers, not MPI
  #C_NATIVE="/opt/intel/oneapi/compiler/2021.4.0/linux/bin/intel64/icc"
  #CXX_NATIVE="/opt/intel/oneapi/compiler/2021.4.0/linux/bin/intel64/icpc"
  #export CXXFLAGS="-Wno-unknown-pragmas $CXXFLAGS"
  # New compiler. Should be faster, but default linker can't find LLVM libc++?
  #C_NATIVE="icx"
  #CXX_NATIVE="icpx"

  #MPI_EXE=mpirun
fi
