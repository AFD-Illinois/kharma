#!/bin/bash

#export NVCC_WRAPPER_DEFAULT_COMPILER="g++"

MAKE="make -f ../Makefile"

mkdir -p build
cd build

if command -v nvcc >/dev/null 2>&1; then
  # Add CUDA
  $MAKE clean
  export KOKKOS_DEVICES="Cuda,OpenMP"
  $MAKE $@
  cp *.cuda ..
fi

# Add OpenMP
$MAKE clean
export KOKKOS_DEVICES="OpenMP"
$MAKE $@
cp *.host ..
