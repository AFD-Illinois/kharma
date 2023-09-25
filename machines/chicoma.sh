# LANL Machines: HPC and IC

# Chicoma
if [[ "$HOST" == "ch-fe"* || "$HOST" == "nid00"* ]]; then
  HOST_ARCH="ZEN2"
  NPROC=64

  # Cray environments get confused easy
  # Make things as simple as possible
  # TODO version with Cray wrappers?
  module purge
  export CRAY_CPU_TARGET="x86-64"
  if [[ "$ARGS" == *"cuda"* ]]; then
    DEVICE_ARCH="AMPERE80"
    if [[ "$ARGS" == *"gnu"* ]]; then
      module load PrgEnv-gnu
    elif [[ "$ARGS" == *"intel"* ]]; then
      module load PrgEnv-intel
    elif [[ "$ARGS" == *"nvc++"* ]]; then
      module load PrgEnv-nvhpc
      EXTRA_FLAGS="-DCMAKE_CUDA_COMPILER=$HOME/bin/nvc++-wrapper -DCMAKE_CUDA_COMPILER_ID=NVHPC -DCMAKE_CUDA_COMPILER_VERSION=11.6 $EXTRA_FLAGS"
    else
      module load PrgEnv-nvhpc
    fi
    module load cpe-cuda cuda craype-accel-nvidia80
    # GPU runtime opts
    MPI_NUM_PROCS=4
    MPI_EXTRA_ARGS="--cpu-bind=mask_cpu:0x0*16,0x1*16,0x2*16,0x3*16 $SOURCE_DIR/bin/select_gpu_chicoma"
    unset OMP_NUM_THREADS
    unset OMP_PROC_BIND
    unset OMP_PLACES
  else
    module load PrgEnv-aocc
  fi
  module load cray-hdf5-parallel cmake
  # System HDF5 can't use compression
  EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON $EXTRA_FLAGS"
  export MPICH_GPU_SUPPORT_ENABLED=1

  # Runtime opts
  MPI_EXE=srun
fi
