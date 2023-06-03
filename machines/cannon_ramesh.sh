# Harvard Cannon

#if [[ $HOST == *"rc.fas.harvard.edu" ]]; then
if [[ "$ARGS" == *"rocky"* ]]; then
    echo $(hostname -f)
    echo ROCKY
    HOST_ARCH=HSW
    EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON"
    echo $ARGS
    echo "after printing out"
    module purge
    module load gcc/10.2.0-fasrc01 openmpi/4.1.0-fasrc01 cmake/3.25.2-fasrc01
    #module load gcc/12.2.0-fasrc01 mpich/4.1-fasrc01 cmake/3.25.2-fasrc01
    export MPICH_GPU_SUPPORT_ENABLED=1
    #module load intel/23.0.0-fasrc01 openmpi/4.1.4-fasrc01 cmake/3.25.2-fasrc01

    #source /n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/spack/share/spack/setup-env.sh
    #spack clean -m
    #spack install gcc@10.2.0 openmpi@4.1.3 cmake@3.23.2

    C_NATIVE=gcc
    CXX_NATIVE=g++
    export PATH=/n/home09/hyerincho/packages/hdf5-openmpi4.1.1:$PATH
    if [[ "$ARGS" == *"cuda"* ]]; then
      DEVICE_ARCH=AMPERE80 ## rocky_gpu
      module load cuda/12.0.1-fasrc01
    fi
elif [[ $(hostname -f) == *"rc.fas.harvard.edu" ]]; then
    echo CANNON
    HOST_ARCH=HSW
    EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON"
    echo $ARGS
    echo "after printing out"
    module purge
    #module unload hdf5
    #module unload Anaconda3/2020.11

    if [[ "$ARGS" == *"gcc10"* ]]; then
      # try 1: weird B cleaning phi dependence
      module load gcc/10.2.0-fasrc01 # test
      module load openmpi/4.1.3-fasrc03 # test
    fi

    # try 2: doesn't compile! (03/27/23), (03/29/23) but Ben says it's possible when using CUDA 12.0
    #module load gcc/12.1.0-fasrc01
    #module load openmpi/4.1.3-fasrc02

    # try 3: 
    #module load gcc/9.3.0-fasrc01
    #module load openmpi/4.0.5-fasrc01

    # try 4: phi dependence gone (03/29/23)
    if [[ "$ARGS" == *"gcc08"* ]]; then
      module load gcc/8.2.0-fasrc01
      module load openmpi/4.1.1-fasrc02
    fi

    #module load cmake/3.17.3-fasrc01 # newer versions are usually better
    module load cmake/3.23.2-fasrc01 # with nvhpc
  if [[ "$ARGS" == *"cuda"* ]]; then
    #DEVICE_ARCH=VOLTA70 ## test, (old GPUs)
    DEVICE_ARCH=AMPERE80 ## blackhole_gpu, itc_gpu

    if [[ "$ARGS" == *"nvhpc"* ]]; then 
      # use nvhpc instead (03/28/23)
      # https://github.com/fasrc/User_Codes/blob/master/Documents/Software/Spack.md
      source /n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/spack/share/spack/setup-env.sh
      #spack load nvhpc/tkncmer # with mpi
      spack load nvhpc@22.7
      spack find --loaded
      #PREFIX_PATH=/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/nvhpc-22.9-tkncmerau6ssssdbw6gpgusvqy6r4grc/Linux_x86_64/22.9/comm_libs/mpi/
      C_NATIVE=nvc #mpicc #
      CXX_NATIVE=nvc++ #mpicxx #
      export CXXFLAGS="-mp"
      export CFLAGS="-mp"
    fi

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

