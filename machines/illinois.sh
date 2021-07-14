
# Illinois BH cluster
if [[ $HOST == *".astro.illinois.edu" ]]; then
  # When oneAPI works
  #source /opt/intel/oneapi/setvars.sh
  #PREFIX_PATH="$HOME/libs/hdf5-oneapi"

  module load gnu mpich phdf5
  PREFIX_PATH="$MPI_DIR"

  HOST_ARCH="SKX"
fi
# Except BH27/9
if [[ $HOST == "bh29.astro.illinois.edu" ]]; then
  source /opt/AMD/aocc-compiler-3.0.0/
  HOST_ARCH="AMDAVX"
fi
if [[ $HOST == "bh27.astro.illinois.edu" ]]; then
  HOST_ARCH="WSM"
fi
