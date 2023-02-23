# Harvard Cannon

#if [[ $HOST == *"rc.fas.harvard.edu" ]]; then
if [[ $(hostname -f) == *"rc.fas.harvard.edu" ]]; then
    echo CANNON
    HOST_ARCH=HSW
    EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON"
    echo $ARGS
    echo "after printing out"
    module unload hdf5
    module unload Anaconda3/2020.11

    #module load gcc/12.1.0-fasrc01
    #module load openmpi/4.1.3-fasrc01
    module load cmake/3.17.3-fasrc01 # newer versions are usually better
    #C_NATIVE=gcc
    #CXX_NATIVE=g++
    #module load cmake/3.23.2-fasrc01 # newer versions are usually better
  if [[ "$ARGS" == *"cuda"* ]]; then
    #DEVICE_ARCH=VOLTA70 ## test, (old GPUs)
    DEVICE_ARCH=AMPERE80 ## blackhole_gpu, itc_gpu
    module load gcc/9.3.0-fasrc01
    module load openmpi/4.0.5-fasrc01
    #module load cuda/11.1.0-fasrc01
    module load cuda/11.6.2-fasrc01
    export PATH=/n/home09/hyerincho/packages/hdf5-openmpi4.1.1:$PATH
  else
    module load intel/19.0.5-fasrc01
    module load openmpi/4.0.1-fasrc01
    export PATH=/n/home09/hyerincho/packages/hdf5-openmpi4.0.1:$PATH
    export PATH=/n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin:$PATH
    export LIBRARY_PATH=/n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/lib64:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/lib64:$LD_LIBRARY_PATH
  fi

fi

