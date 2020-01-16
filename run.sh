#!/bin/bash

exe=${1:-host}

# Alternately
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# CUDA UVM Setup (TODO use non-UVM in future?)
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

#numactl --interleave=all ./ngrain.${exe}
./kharm.${exe} input.h5
