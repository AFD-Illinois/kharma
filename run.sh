#!/bin/bash

exe=${1:-cuda}

# OpenMP Setup
#export OMP_PROC_BIND=true
# Alternately
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads

# To use all threads on cinn
export GOMP_CPU_AFFINITY=0-55
export OMP_NUM_THREADS=56
export OMP_PROC_BIND=true

# CUDA UVM Setup (TODO use non-UVM in future?)
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

#numactl --interleave=all ./ngrain.${exe}
./ngrain.${exe}
