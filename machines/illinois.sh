
# LMC is special
if [[ $HOST == "lmc.astro.illinois.edu" ]]; then
  #conda deactivate
  HOST_ARCH="HSW"
  # When we can compile HDF5 successfully
  #PREFIX_PATH="$HOME/libs/hdf5"
  # Until then, disable MPI & static HDF5 as bad versions are installed
  PREFIX_PATH=
  EXTRA_FLAGS="-DPARTHENON_DISABLE_MPI=ON"
else

# Illinois BH cluster
if [[ $HOST == *".astro.illinois.edu" ]]; then
  HOST_ARCH="SKX"
  module purge

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
# BH29 additions
if [[ $HOST == "bh29.astro.illinois.edu" ]]; then
  HOST_ARCH="ZEN2"

  # AOCC Requires system libc++, like icpx
  #source /opt/AMD/aocc-compiler-3.0.0/setenv_AOCC.sh
  #CXX_NATIVE="clang++"
fi

fi #LMC
