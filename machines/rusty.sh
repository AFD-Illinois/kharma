# ------------------------------------------------------------------ #
# Config for Flatiron's Rusty cluster
#
# Rusty has three generations of NVIDIA GPUs - H100, A100, V100
# The CPU-GPU combination are,
#           Icelake-H100
#           Icelake-A100
#           Skylake-V100
#
# 'h100', 'a100', or 'v100' can be specified while building KHARMA
# to specify which arch the code should compile for.
# The default is 'a100'.
# 'cpu' specifies a CPU build on Icelake arch
#
# ------------------------------------------------------------------ #
if [[ $HOST == "rustyamd"* || $HOST == "worker"* ]]
then
    # Get host and device arch
    NPROC=64
    if [[ $ARGS == *"h100"* ]]; then
        HOST_ARCH="ICX"
        DEVICE_ARCH="HOPPER90"
    elif [[ $ARGS == *"a100"* ]]; then
        HOST_ARCH="ICX"
        DEVICE_ARCH="AMPERE80"
    elif [[ $ARGS == *"v100"* ]]; then
        HOST_ARCH="SKX"
        DEVICE_ARCH="VOLTA70"
    elif [[ $ARGS == *"cpu"* ]]
    then
        HOST_ARCH="ICX"
    fi

    # GPU compile
    if [[ $ARGS == *"cuda"* ]]; then
        echo "GPU build"
        # Load required modules
        module --force purge
        module load modules/2.3-20240529 slurm cmake gcc/11.4.0 cuda gsl openmpi hdf5 openblas
        module list
        # 8-way compile for H100s
        if [[ $ARGS == *"h100"* ]]; then
            MPI_EXTRA_ARGS="--map-by ppr:8:node:pe=8"
            MPI_NUM_PROCS=8
        # 4-way compile for A100s and V100s
        elif [[ $ARGS == *"a100"* ]]; then
            # 4-way compile for A100s
            MPI_EXTRA_ARGS="--map-by ppr:4:node:pe=16"
            MPI_NUM_PROCS=4
        else
            # 4-way compile for V100s
            MPI_EXTRA_ARGS="--map-by ppr:4:node:pe=9"
            MPI_NUM_PROCS=4
        fi
        C_NATIVE=gcc
        CXX_NATIVE=g++
    # CPU compile
    else
        echo "CPU build"
        module --force purge
        module load modules/2.3-20240529 slurm gcc/11.4.0 cmake gsl openmpi openblas hdf5
        module list
        if [[ "$ARGS" == *"skx"* ]]; then
            HOST_ARCH="SKX"
            NPROC=64
        elif [[ "$ARGS" == *"rome"* ]]; then
            HOST_ARCH="ZEN2"
            NPROC=128
        else
            # Default to Icelake
            HOST_ARCH="SKX"
            NPROC=64
        fi
        C_NATIVE=gcc
        CXX_NATIVE=g++
    fi
fi
