#!/bin/bash

# TODO if these things are necessary then...
unset CPATH
conda deactivate
source scl_source enable devtoolset-8

# Make script for KHARMA
# Used to decide flags and call cmake
# TODO autodetection?  Machinefiles?

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $SCRIPT_DIR
if [[ "$*" == *"clean"* ]]; then
  rm -rf build kharm.* core.*
fi
mkdir -p build

cd build

if [[ "$*" == *"clean"* ]]; then
  if false; then # CUDA BUILD
    # TODO unify MPI flags
    cmake3 ..\
    -DCMAKE_CXX_COMPILER=$PWD/../external/parthenon/external/Kokkos/bin/nvcc_wrapper \
    -DUSE_MPI=OFF \
    -DDISABLE_MPI=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_WSM=OFF \
    -DKokkos_ARCH_HSW=ON \
    -DKokkos_ARCH_BDW=OFF \
    -DKokkos_ARCH_SKX=OFF \
    -DKokkos_ARCH_POWER9=OFF \
    -DKokkos_ARCH_ARM
    -DKokkos_ARCH_KEPLER35=ON \
    -DKokkos_ARCH_MAXWELL52=OFF \
    -DKokkos_ARCH_PASCAL60=OFF \
    -DKokkos_ARCH_VOLTA70=OFF \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON
  else #OpenMP BUILD
    cmake3 ..\
    -DCMAKE_CXX_FLAGS="-I/opt/apps/intel18/hdf5/1.10.4/x86_64/include/ \
			-L/opt/apps/intel18/hdf5/1.10.4/x86_64/lib/ \
            -I/usr/include/mpich-x86_64/ \
            -L/usr/lib64/mpich/lib/" \
    -DUSE_MPI=OFF \
    -DDISABLE_MPI=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=ON \
    -DKokkos_ARCH_BDW=OFF \
    -DKokkos_ARCH_KNL=OFF
  fi
fi

make -j
cp kharma/kharma.* ..
