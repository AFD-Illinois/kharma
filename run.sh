#!/bin/bash



# Alternately
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# CUDA UVM Setup (TODO use non-UVM in future?)
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

if [ -f kharm.cuda ]; then
  ./kharm.cuda
elif [ -f kharm.host ]; then
  ./kharm.host
else
  echo "K/HARM executable not found!"
fi
