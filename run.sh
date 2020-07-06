#!/bin/bash

# OpenMP
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
# Cuda
export CUDA_LAUNCH_BLOCKING=0

# Use GPU 0, 1 is flaky on cinnabar
export KOKKOS_DEVICE_ID=0

if [ -f kharma.cuda ]; then
  ./kharma.cuda "$@"
elif [ -f kharma.host ]; then
  ./kharma.host "$@"
else
  echo "KHARMA executable not found!"
fi
