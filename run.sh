#!/bin/bash

### System-specific parameters
# Override these with your compile file in machines/!
# For running different configs on the fly, you can use the options
# -n (number of MPI procs)
# -nt (number of OpenMP threads)
# Note these options must be FIRST and IN ORDER!

# Optionally use the Kokkos tools to profile kernels
#export KOKKOS_PROFILE_LIBRARY=$KHARMA_DIR/../kokkos-tools/kp_kernel_timer.so
#export KOKKOS_PROFILE_LIBRARY=$KHARMA_DIR/../kokkos-tools/kp_nvprof_cnnector.so

# Default MPI parameters: don't use MPI or run with 1 process
MPI_EXE=${MPI_EXE:-}
MPI_NUM_PROCS=${MPI_NUM_PROCS:-1}
MPI_EXTRA_ARGS=${MPI_EXTRA_ARGS:-}

# Default OpenMP directives: use all available threads
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-threads}
# Force a number of OpenMP threads if it doesn't autodetect
#export OMP_NUM_THREADS=28



### General run script

# Map each MPI rank to one device with Kokkos
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
# If you see weird GPU race conditions, setting this
# to 1 *might* fix them. Maybe.
export CUDA_LAUNCH_BLOCKING=0
# Kokkos can be forced to use only a particular device:
#export KOKKOS_DEVICE_ID=0

# Choose the kharma binary from compiled options in order of preference
KHARMA_DIR="$(dirname "${BASH_SOURCE[0]}")"
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

# Load environment from the same files as the compile process
HOST=$(hostname -f)
ARGS=$(cat $KHARMA_DIR/make_args)
for machine in $KHARMA_DIR/machines/*.sh
do
  source $machine
done

# Override MPI_NUM_PROCS at user option "-n"
# and OMP_NUM_THREADS at option "-nt"
if [[ "$1" == "-n" ]]; then
  MPI_NUM_PROCS="$2"
  if [[ -z $MPI_EXE && $(( $MPI_NUM_PROCS > 1 )) ]]; then
    MPI_EXE="mpirun"
  fi
  shift
  shift
fi
if [[ "$1" == "-nt" ]]; then
  export OMP_NUM_THREADS="$2"
  shift
  shift
fi

# Run based on preferences
if [ -z "$MPI_EXE" ]; then
  echo "Running $KHARMA_DIR/$EXE_NAME $@"
  exec $KHARMA_DIR/$EXE_NAME "$@"
else
  echo "Running $MPI_EXE -n $MPI_NUM_PROCS $MPI_EXTRA_ARGS $KHARMA_DIR/$EXE_NAME $@"
  exec $MPI_EXE -n $MPI_NUM_PROCS $MPI_EXTRA_ARGS $KHARMA_DIR/$EXE_NAME "$@"
fi
