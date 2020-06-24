#!/bin/bash

# TODO if these things are necessary then...
unset CPATH
conda deactivate

# Make script for KHARMA
# Used to decide flags and call cmake
# TODO autodetection?  Machinefiles?

if [[ "$*" == *"debug"* ]]; then
  TYPE=Debug
else
  TYPE=Release
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $SCRIPT_DIR
if [[ "$*" == *"clean"* ]]; then
  rm -rf build kharm.* core.*
fi
mkdir -p build

cd build

if [[ "$*" == *"clean"* ]]; then
  if [[ "$*" == *"cuda"* ]]; then # CUDA BUILD
    # TODO unify MPI flags
    cmake3 ..\
    -DCMAKE_CXX_COMPILER=$PWD/../external/parthenon/external/Kokkos/bin/nvcc_wrapper \
    -DCMAKE_BUILD_TYPE=$TYPE \
    -DCMAKE_PREFIX_PATH=/usr/lib64/mpich \
    -DPAR_LOOP_LAYOUT="MANUAL1D_LOOP" \
    -DPAR_LOOP_INNER_LAYOUT="SIMDFOR_INNER_LOOP" \
    -DENABLE_UNIT_TESTS=OFF \
    -DENABLE_INTEGRATION_TESTS=OFF \
    -DENABLE_REGRESSION_TESTS=OFF \
    -DENABLE_EXAMPLES=OFF \
    -DPARTHENON_DISABLE_MPI=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_WSM=OFF \
    -DKokkos_ARCH_HSW=ON \
    -DKokkos_ARCH_BDW=OFF \
    -DKokkos_ARCH_SKX=OFF \
    -DKokkos_ARCH_POWER9=OFF \
    -DKokkos_ARCH_KEPLER35=ON \
    -DKokkos_ARCH_PASCAL60=OFF \
    -DKokkos_ARCH_VOLTA70=OFF \
    -DKokkos_ARCH_TURING75=OFF \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON
  else #OpenMP BUILD
    cmake3 ..\
    -DCMAKE_BUILD_TYPE=$TYPE \
    -DCMAKE_PREFIX_PATH=/usr/lib64/mpich \
    -DPAR_LOOP_LAYOUT="MANUAL1D_LOOP" \
    -DPAR_LOOP_INNER_LAYOUT="SIMDFOR_INNER_LOOP" \
    -DENABLE_UNIT_TESTS=OFF \
    -DENABLE_INTEGRATION_TESTS=OFF \
    -DENABLE_REGRESSION_TESTS=OFF \
    -DENABLE_EXAMPLES=OFF \
    -DPARTHENON_DISABLE_MPI=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=ON \
    -DKokkos_ARCH_BDW=OFF \
    -DKokkos_ARCH_SKX=OFF \
    -DKokkos_ARCH_KNL=OFF \
    -DKokkos_ARCH_ARMV8_THUNDERX2=OFF \
    -DKokkos_ARCH_AMDAVX=OFF \
    -DKokkos_ARCH_EPYC=OFF
  fi
fi

make -j
cp kharma/kharma.* ..
