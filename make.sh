#!/bin/bash

# Make script for KHARMA
# Used to decide flags and call cmake
# Usage:
# ./make.sh [option1] [option2]

# clean: BUILD by re-running cmake, restarting the make process from nothing.
#        That is, "./make.sh clean" == "make clean" + "make"
#        Always use 'clean' when switching Release<->Debug or OpenMP<->CUDA
# cuda:  Build for GPU with CUDA. Must have 'nvcc' in path
# sycl:  Build for GPU with SYCL. Must have 'icpx' in path
# debug: Configure with debug flags: mostly array bounds checks
#        Note, though, many sanity checks during the run are
#        actually *runtime* parameters e.g. verbose, flag_verbose, etc
# trace: Configure with execution tracing: print at the beginning and end
#        of most host-side function calls during a step
# skx:   Compile specifically for Skylake nodes on Stampede2

# Processors to use.  Leave blank for all.  Be a good citizen.
NPROC=

### Machine-specific configurations ###
# This segment sources a series of machine-specific
# definitions from the machines/ directory.
# If the current machine isn't listed, this script
# and/or Kokkos will attempt to guess the host architecture,
# which should suffice to compile but may not provide optimal
# performance.

# See e.g. tacc.sh for an example to get started writing one,
# or specify any options you need manually below

# Kokkos_ARCH options:
# CPUs: WSM, HSW, BDW, SKX, AMDAVX
# ARM: ARMV8, ARMV81, ARMV8_THUNDERX2, A64FX
# POWER: POWER8, POWER9
# MIC: KNC, KNL
# GPUs: KEPLER35, VOLTA70, TURING75, AMPERE80

# HOST_ARCH=
# DEVICE_ARCH=

# Less common options:
# PREFIX_PATH=
# EXTRA_FLAGS=
# export NVCC_WRAPPER_DEFAULT_COMPILER=

HOST=$(hostname -f)
for machine in machines/*.sh
do
  source $machine
done


# If we haven't special-cased already, guess an architecture
# This ends up pretty much optimal on x86 architectures which don't have
# 1. AVX512 (Intel on HPC or Gen10+ consumer)
# 2. GPUs
if [[ -z "$HOST_ARCH" ]]; then
  if grep GenuineIntel /proc/cpuinfo >/dev/null 2>&1; then
    HOST_ARCH="HSW"
  fi
  if grep AuthenticAMD /proc/cpuinfo >/dev/null 2>&1; then
    HOST_ARCH="AMDAVX"
  fi
fi

# Add some flags only if they're set
if [[ -v HOST_ARCH ]]; then
  EXTRA_FLAGS="-DKokkos_ARCH_${HOST_ARCH}=ON $EXTRA_FLAGS"
fi
if [[ -v DEVICE_ARCH ]]; then
  EXTRA_FLAGS="-DKokkos_ARCH_${DEVICE_ARCH}=ON $EXTRA_FLAGS"
fi
if [[ -v PREFIX_PATH ]]; then
  EXTRA_FLAGS="-DCMAKE_PREFIX_PATH=$PREFIX_PATH $EXTRA_FLAGS"
fi
if [[ "$*" == *"trace"* ]]; then
  EXTRA_FLAGS="-DTRACE=1 $EXTRA_FLAGS"
fi

### Check environment ###
if [[ "$(which python3 2>/dev/null)" == *"conda"* ]]; then
  echo "It looks like you have Anaconda loaded."
  echo "Anaconda forces a serial version of HDF5 which makes this compile impossible."
  echo "Deactivate your environment with 'conda deactivate'"
fi

if [[ "$*" == *"debug"* ]]; then
  TYPE=Debug
else
  TYPE=Release
fi

### Build HDF5 ###
if [[ "$*" == *"hdf5"* ]]; then
  cd external
  if [ ! -f hdf5-* ]; then
    wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_0/source/hdf5-1.12.0.tar.gz
    tar xf hdf5-1.12.0.tar.gz
  fi
  cd hdf5-1.12.0/
  make clean
  CC=mpicc ./configure --enable-parallel --prefix=$PWD/../hdf5
  make -j$NPROC
  make install
  make clean
  exit
fi

### Build KHARMA ###
SCRIPT_DIR=$( dirname "$0" )
cd $SCRIPT_DIR
SCRIPT_DIR=$PWD

# Strongly prefer icc for OpenMP compiles
# I would try clang but it would break all Macs
if [[ -z "$CXX_NATIVE" ]]; then
  if which icpc >/dev/null 2>&1; then
    CXX_NATIVE=icpc
    C_NATIVE=icc
    # Avoid warning on nvcc pragmas Intel doesn't like
    export CXXFLAGS="-Wno-unknown-pragmas"
    #export CFLAGS="-qopenmp"
  elif which cc >/dev/null 2>&1; then
    CXX_NATIVE=CC
    C_NATIVE=cc
    #export CXXFLAGS="-Wno-unknown-pragmas" # TODO if Cray->Intel in --version
  elif which xlC >/dev/null 2>&1; then
    CXX_NATIVE=xlC
    C_NATIVE=xlc
  else
    CXX_NATIVE=g++
    C_NATIVE=gcc
  fi
fi

# CUDA loop options: MANUAL1D_LOOP > MDRANGE_LOOP, TPTTR_LOOP & TPTTRTVR_LOOP don't compile
# Inner loop must be TVR_INNER_LOOP
# OpenMP loop options for KNL:
# Outer: SIMDFOR_LOOP;MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTVR_LOOP;TPTTRTVR_LOOP
# Inner: SIMDFOR_INNER_LOOP;TVR_INNER_LOOP
if [[ "$*" == *"sycl"* ]]; then
  export CXX=icpx
  EXTRA_FLAGS="-DCMAKE_C_COMPILER=icx $EXTRA_FLAGS"
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="ON"
  ENABLE_HIP="OFF"
elif [[ "$*" == *"hip"* ]]; then
  export CXX=hipcc
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="ON"
elif [[ "$*" == *"cuda"* ]]; then
  export CXX="$SCRIPT_DIR/bin/nvcc_wrapper"
  if [[ "$*" == *"dryrun"* ]]; then
    export CXXFLAGS="-dryrun $CXXFLAGS"
    echo "Dry-running with $CXXFLAGS"
  fi
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="ON"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
elif [[ "$*" == *"clanggpu"* ]]; then
  export CXX="clang++"
  export CC="clang"
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="ON"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
else
  export CXX="$CXX_NATIVE"
  export CC="$C_NATIVE"
  OUTER_LAYOUT="MDRANGE_LOOP"
  INNER_LAYOUT="SIMDFOR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
fi

# Make build dir. Recall "clean" means "clean and build"
if [[ "$*" == *"clean"* ]]; then
  rm -rf build
fi
mkdir -p build
cd build

if [[ "$*" == *"clean"* ]]; then
#set -x
  cmake ..\
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH:$CMAKE_PREFIX_PATH" \
    -DCMAKE_BUILD_TYPE=$TYPE \
    -DPAR_LOOP_LAYOUT=$OUTER_LAYOUT \
    -DPAR_LOOP_INNER_LAYOUT=$INNER_LAYOUT \
    -DKokkos_ENABLE_OPENMP=$ENABLE_OPENMP \
    -DKokkos_ENABLE_CUDA=$ENABLE_CUDA \
    -DKokkos_ENABLE_SYCL=$ENABLE_SYCL \
    -DKokkos_ENABLE_HIP=$ENABLE_HIP \
    $EXTRA_FLAGS
#set +x
fi

make -j$NPROC

cp kharma/kharma.* ..
