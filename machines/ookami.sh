
# Ookami. ARM compiler seems the most compatible,
# "Vendor" would be Fujitsu or Cray CC
if [[ $HOST == *".cm.cluster" ]]; then
  HOST_ARCH="A64FX"
  PREFIX_PATH=$PWD/external/hdf5

  module purge
  module load cmake

  module load gcc/11.1.0
  module load openmpi/gcc11/4.1.1
  #module load mvapich2/gcc11/2.3.6
  C_NATIVE="gcc"
  CXX_NATIVE="g++"

  #module load arm-modules/21
  #module load openmpi/arm21/4.1.1
  #module load mvapich2/arm21/2.3.6
  #C_NATIVE="armclang"
  #CXX_NATIVE="armclang++"
  #PREFIX_PATH="$HOME/libs/hdf5-arm21-mvapich2"

  #PREFIX_PATH="$HOME/libs/hdf5-cray-mvapich2_nogpu_svealpha"
  #CXX_NATIVE="CC"
  #CRAYPE_LINK_TYPE="dynamic"
fi
