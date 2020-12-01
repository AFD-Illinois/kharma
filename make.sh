#!/bin/bash

# Make script for KHARMA
# Used to decide flags and call cmake
# TODO autodetection would be fun here, Parthenon may do some in future too.

# Set the correct compiler on Fedora machines
if [[ $(hostname) == "toolbox" ]]; then
  module load mpi
  #export NVCC_WRAPPER_DEFAULT_COMPILER=cuda-g++
  export PATH="/usr/local/cuda/bin/:$PATH"
fi

# Make conda go away.  Bad libraries. Bad.
echo "If this is a Conda path deactivate your environment: $(which python)"

# Only use icc on Stampede
if [[ $(hostname) == *"stampede2"* ]]; then
  CC_NATIVE=icc
  CXX_NATIVE=icpc
else
  CC_NATIVE=gcc
  CXX_NATIVE=g++
fi

if [[ "$*" == *"debug"* ]]; then
  TYPE=Debug
else
  TYPE=Release
fi

# "Clean" here is 
SCRIPT_DIR=$( dirname "$0" )
cd $SCRIPT_DIR
if [[ "$*" == *"clean"* ]]; then
  rm -rf build
fi

mkdir -p build
cd build

# CUDA loop options: MANUAL1D_LOOP > MDRANGE_LOOP, TPTTR_LOOP & TPTTRTVR_LOOP don't compile
# Inner loop must be TVR_INNER_LOOP
# OpenMP loop options for KNL:
# Outer: SIMDFOR_LOOP;MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTVR_LOOP;TPTTRTVR_LOOP
# Inner: SIMDFOR_INNER_LOOP;TVR_INNER_LOOP

if [[ "$*" == *"clean"* ]]; then
  if [[ "$*" == *"cuda"* ]]; then # CUDA BUILD
    # TODO unify MPI flags
    cmake ..\
    -DCMAKE_CXX_COMPILER=$PWD/../external/parthenon/external/Kokkos/bin/nvcc_wrapper \
    -DCMAKE_BUILD_TYPE=$TYPE \
    -DCMAKE_PREFIX_PATH=/usr/lib64/openmpi \
    -DPAR_LOOP_LAYOUT="MANUAL1D_LOOP" \
    -DPAR_LOOP_INNER_LAYOUT="TVR_INNER_LOOP" \
    -DBUILD_TESTING=OFF \
    -DPARTHENON_DISABLE_EXAMPLES=ON \
    -DPARTHENON_DISABLE_MPI=OFF \
    -DPARTHENON_NGHOST=4 \
    -DPARTHENON_LINT_DEFAULT=OFF \
    -DENABLE_COMPILER_WARNINGS=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_WSM=OFF \
    -DKokkos_ARCH_HSW=OFF \
    -DKokkos_ARCH_BDW=OFF \
    -DKokkos_ARCH_SKX=OFF \
    -DKokkos_ARCH_AMDAVX=ON \
    -DKokkos_ARCH_POWER9=OFF \
    -DKokkos_ARCH_KEPLER35=OFF \
    -DKokkos_ARCH_VOLTA70=OFF \
    -DKokkos_ARCH_TURING75=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_CUDA_CONSTEXPR=ON
  else #OpenMP BUILD
    cmake ..\
    -DCMAKE_CXX_COMPILER=$CXX_NATIVE \
    -DCMAKE_BUILD_TYPE=$TYPE \
    -DCMAKE_PREFIX_PATH=/usr/lib64/openmpi \
    -DPAR_LOOP_LAYOUT="MANUAL1D_LOOP" \
    -DPAR_LOOP_INNER_LAYOUT="SIMDFOR_INNER_LOOP" \
    -DBUILD_TESTING=OFF \
    -DPARTHENON_DISABLE_EXAMPLES=ON \
    -DPARTHENON_DISABLE_MPI=OFF \
    -DPARTHENON_NGHOST=4 \
    -DPARTHENON_LINT_DEFAULT=OFF \
    -DENABLE_COMPILER_WARNINGS=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=OFF \
    -DKokkos_ARCH_BDW=OFF \
    -DKokkos_ARCH_SKX=OFF \
    -DKokkos_ARCH_KNL=OFF \
    -DKokkos_ARCH_ARMV8_THUNDERX2=OFF \
    -DKokkos_ARCH_AMDAVX=ON
  fi
fi

make -j12
cp kharma/kharma.* ..
