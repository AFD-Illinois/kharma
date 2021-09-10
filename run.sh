#!/bin/bash

# OpenMP directives: use all available threads
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Force a number of OpenMP threads if it doesn't autodetect
#export OMP_NUM_THREADS=28
# Number of GPUs on the node (doesn't matter for CPU runs):
export KOKKOS_NUM_DEVICES=2

# If you see weird GPU race conditions, setting this
# to 1 *might* fix them. Maybe.
export CUDA_LAUNCH_BLOCKING=0
# Kokkos can be forced to a particular device:
#export KOKKOS_DEVICE_ID=0

KHARMA_DIR="$(dirname $0)"
if [ -f $KHARMA_DIR/kharma.cuda ]; then
  EXE_NAME=kharma.cuda
elif [ -f $KHARMA_DIR/kharma.sycl ]; then
  EXE_NAME=kharma.sycl
elif [ -f $KHARMA_DIR/kharma.host ]; then
  EXE_NAME=kharma.host
else
  echo "KHARMA executable not found!"
  exit
fi

# Optionally use the Kokkos tools to profile
#export KOKKOS_PROFILE_LIBRARY=$KHARMA_DIR/../kokkos-tools/kp_kernel_timer.so
#export KOKKOS_PROFILE_LIBRARY=$KHARMA_DIR/../kokkos-tools/kp_nvprof_cnnector.so

# TODO options based on hostname etc here
#$KHARMA_DIR/external/parthenon/external/Kokkos/bin/hpcbind --whole-system -- $KHARMA_DIR/$EXE_NAME "$@"
#mpirun -n 8 $KHARMA_DIR/$EXE_NAME "$@"
#mpirun -n 4 $KHARMA_DIR/$EXE_NAME "$@"
#mpirun -n 2 $KHARMA_DIR/$EXE_NAME "$@"
#mpirun -n 1 $KHARMA_DIR/$EXE_NAME "$@"
#mpirun -n 2 --map-by ppr:1:numa:pe=14 $KHARMA_DIR/$EXE_NAME "$@"
$KHARMA_DIR/$EXE_NAME "$@"
