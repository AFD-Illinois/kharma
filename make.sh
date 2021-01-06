#!/bin/bash

# Make script for KHARMA
# Used to decide flags and call cmake
# Usage:
# ./make.sh [clean] [cuda] [debug]

# clean: BUILD by re-running cmake, restarting the make process from nothing
#        That is, "./make.sh clean" == "make clean" + "make"
#        Always use 'clean' when switching Release->Debug or OpenMP->CUDA
# cuda:  Build for GPU with CUDA. Must have 'nvcc' in path
# debug: Configure with debug flags: mostly array bounds checks
#        Note most prints/fluid sanity checks are actually *runtime* parameters

### Machine-specific configurations ###
if [[ $HOSTNAME == "toolbox" ]]; then
  HOST=fermium
else
  HOST=$(hostname -f)
fi

# Kokkos_ARCH options:
# CPUs: WSM, HSW, BDW, SKX, AMDAVX
# ARM: ARMV8, ARMV81, ARMV8_THUNDERX2
# POWER: POWER8, POWER9
# MIC: KNC, KNL
# GPUs: KEPLER35, VOLTA70, TURING75

# TACC resources
# Generally you want latest Intel/IMPI/phdf5 modules,
# On longhorn use gcc/7, cuda, manually-compiled PHDF5
if [[ $HOST == *".frontera.tacc.utexas.edu" ]]; then
  HOST_ARCH="SKX"
fi
if [[ $HOST == *".stampede2.tacc.utexas.edu" ]]; then
  HOST_ARCH="KNL"
  #HOST_ARCH="SKX"
fi
if [[ $HOST == *".longhorn.tacc.utexas.edu" ]]; then
  HOST_ARCH="POWER9"
  DEVICE_ARCH="VOLTA70"
fi

# Illinois BH cluster
if [[ $HOST == *".astro.illinois.edu" ]]; then
  # When oneAPI works
  #source /opt/intel/oneapi/setvars.sh
  #PREFIX_PATH="~/libs/hdf5-oneapi"

  module load gnu mpich phdf5
  PREFIX_PATH="$MPI_DIR"

  HOST_ARCH="SKX"
fi
# Except BH27/9
if [[ $HOST == "bh29.astro.illinois.edu" ]]; then
  HOST_ARCH="AMDAVX"
fi
if [[ $HOST == "bh27.astro.illinois.edu" ]]; then
  HOST_ARCH="WSM"
fi

if [[ $HOST == "fermium" ]]; then
  HOST_ARCH="AMDAVX"
  DEVICE_ARCH="TURING75"
  EXTRA_FLAGS="-DCUDAToolkit_INCLUDE_DIR=/usr/include/cuda $EXTRA_FLAGS"
fi

# If we haven't special-cased, guess
if [[ -z "$HOST_ARCH" ]]; then
  if grep Intel /proc/cpuinfo >/dev/null 2>&1; then
    # Probably okay to default to HSW here but being cautious
    HOST_ARCH="WSM"
  fi
  if grep AMD /proc/cpuinfo >/dev/null 2>&1; then
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
  EXTRA_FLAGS="-DCMAKE_PREFIX_PATH=\"$PREFIX_PATH\" $EXTRA_FLAGS"
fi

### Environment ###
if [[ "$(which python)" == *"conda"* ]]; then
  echo "It looks like you have Anaconda loaded."
  echo "Anaconda forces a serial version of HDF5 which makes this compile impossible."
  echo "Deactivate your environment with 'conda deactivate'"
  exit
fi
echo "If this is your Anaconda version of Python, deactivate your environment:"
echo "$(which python)"
echo

if [[ "$*" == *"debug"* ]]; then
  TYPE=Debug
else
  TYPE=Release
fi

### Build ###
SCRIPT_DIR=$( dirname "$0" )
cd $SCRIPT_DIR
SCRIPT_DIR=$PWD

# Strongly prefer icc for OpenMP compiles
if which icpc >/dev/null 2>&1; then
  CXX_NATIVE=icpc
  # Avoid warning on nvcc pragmas Intel doesn't like
  export CXXFLAGS=-Wno-unknown-pragmas
else
  CXX_NATIVE=g++
fi

# CUDA loop options: MANUAL1D_LOOP > MDRANGE_LOOP, TPTTR_LOOP & TPTTRTVR_LOOP don't compile
# Inner loop must be TVR_INNER_LOOP
# OpenMP loop options for KNL:
# Outer: SIMDFOR_LOOP;MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTVR_LOOP;TPTTRTVR_LOOP
# Inner: SIMDFOR_INNER_LOOP;TVR_INNER_LOOP
if [[ "$*" == *"cuda"* ]]; then
  CXX="$SCRIPT_DIR/external/parthenon/external/Kokkos/bin/nvcc_wrapper"
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_CUDA="ON"
else
  CXX="$CXX_NATIVE"
  OUTER_LAYOUT="MDRANGE_LOOP"
  INNER_LAYOUT="SIMDFOR_INNER_LOOP"
  ENABLE_CUDA="OFF"
fi

# Make build dir. Recall "clean" means "clean and build"
if [[ "$*" == *"clean"* ]]; then
  rm -rf build
fi
mkdir -p build
cd build

if [[ "$*" == *"clean"* ]]; then
  cmake ..\
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=$TYPE \
    -DPAR_LOOP_LAYOUT=$OUTER_LAYOUT \
    -DPAR_LOOP_INNER_LAYOUT=$INNER_LAYOUT \
    -DKokkos_ENABLE_CUDA=$ENABLE_CUDA \
    -DKokkos_ARCH_${HOST_ARCH}=ON \
    $EXTRA_FLAGS
fi

#  if [[ "$*" == *"cuda"* ]]; then # CUDA BUILD
#    cmake ..\
#    -DCMAKE_CXX_COMPILER=$PWD/../external/parthenon/external/Kokkos/bin/nvcc_wrapper \
#    -DCMAKE_PREFIX_PATH=/usr/lib64/openmpi \
#    -DCUDAToolkit_INCLUDE_DIR=/usr/include/cuda \

make -j12
cp kharma/kharma.* ..
