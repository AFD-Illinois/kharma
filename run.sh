#!/bin/bash

# OpenMP
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Cuda: 1 device
export CUDA_LAUNCH_BLOCKING=0
#export KOKKOS_DEVICE_ID=0

# Attempt at a personal 2-gpu config
export KOKKOS_NUM_DEVICES=2

KHARMA_DIR="$(dirname $0)"
if [ -f $KHARMA_DIR/kharma.cuda ]; then
  EXE_NAME=kharma.cuda
elif [ -f $KHARMA_DIR/kharma.host ]; then
  EXE_NAME=kharma.host
else
  echo "KHARMA executable not found!"
  exit
fi

# Optionally use the Kokkos tools to profile
#export KOKKOS_PROFILE_LIBRARY=$KHARMA_DIR/../kokkos-tools/kp_kernel_timer.so

# TODO options based on hostname etc here
#$KHARMA_DIR/external/parthenon/external/Kokkos/bin/hpcbind --whole-system -- $KHARMA_DIR/$EXE_NAME "$@"
mpirun -n 2 $KHARMA_DIR/$EXE_NAME "$@"
#mpirun -n 1 $KHARMA_DIR/$EXE_NAME "$@"
#$KHARMA_DIR/$EXE_NAME "$@"
