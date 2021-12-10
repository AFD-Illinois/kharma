
# Illinois BH cluster
if [[ $HOST == *".astro.illinois.edu" ]]; then
  HOST_ARCH="SKX"

  # When all of oneAPI works
  #source /opt/intel/oneapi/setvars.sh
  #PREFIX_PATH="$HOME/libs/hdf5-oneapi"

  # Load system MPI
  module load gnu mpich phdf5
  PREFIX_PATH="$MPI_DIR"
  # To try to use Intel icpc
  # Currently can't access Intel OpenMP library with system MPI easily
  #CXX_NATIVE=/opt/intel/oneapi/compiler/2021.1.2/linux/bin/intel64/icpc
  #export CXXFLAGS="-Wno-unknown-pragmas"

  # New compiler. Should be faster, but requires LLVM libc++ on system
  #CXX_NATIVE="icpx"
fi
# Except BH27/9
if [[ $HOST == "bh29.astro.illinois.edu" ]]; then
  HOST_ARCH="ZEN2"

  # AOCC Requires system libc++, like icpx
  #source /opt/AMD/aocc-compiler-3.0.0/setenv_AOCC.sh
  #CXX_NATIVE="clang++"
fi
