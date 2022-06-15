#!/bin/bash
# NCSA Delta run script: interactive w/mpirun

# OpenMP directives: use all available threads
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# If you see weird GPU race conditions, setting this
# to 1 *might* fix them. Maybe.
export CUDA_LAUNCH_BLOCKING=0
# Kokkos can be forced to a particular device:
#export KOKKOS_DEVICE_ID=0

# KHARMA directory 
KHARMA_DIR="$(realpath $(dirname $(realpath "${BASH_SOURCE[0]}"))/../..)"
echo $KHARMA_DIR

# Optionally use the Kokkos tools to profile kernels
#export KOKKOS_PROFILE_LIBRARY=$KHARMA_DIR/../kokkos-tools/kp_kernel_timer.so
#export KOKKOS_PROFILE_LIBRARY=$KHARMA_DIR/../kokkos-tools/kp_nvprof_cnnector.so

# Load any defaults/modules from the machine file
HOST=$(hostname -f)
ARGS=$(cat $KHARMA_DIR/make_args)
for machine in $KHARMA_DIR/machines/*.sh
do
  source $machine
done

export KOKKOS_NUM_DEVICES=$SLURM_NTASKS_PER_NODE

# Run with mpirun
mpirun $KHARMA_DIR/kharma.cuda -d dumps_kharma "$@"
