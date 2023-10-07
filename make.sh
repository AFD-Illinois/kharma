#!/bin/bash

# Make script for KHARMA
# Used to set sensible default flags and call cmake/make
# Usage:
# ./make.sh [option1] [option2]
#
# clean: BUILD by re-running cmake, restarting the make process from nothing.
#        That is, "./make.sh clean" == "make clean" + "make"
#        Always use 'clean' when switching Release<->Debug or OpenMP<->CUDA
# cuda:  Build for GPU with CUDA
# sycl:  Build for GPU with SYCL
# hip:   Build for GPU with HIP
# debug: Configure with debug flags: mostly array bounds checks
#        Note, though, many sanity checks during the run are
#        actually *runtime* parameters e.g. verbose, flag_verbose, etc
# trace: Configure with execution tracing: print at the beginning and end
#        of most host-side function calls during a step
# hdf5:  Download & compile HDF5, rather than looking for a system version
# cleanhdf5:  Reconfigure HDF5 from scratch, rather than just recompiling
# nompi:      Disable MPI and don't search/link it
# noimplicit: Disable implicit solver, avoids pulling in Kokkos-kernels
# nocleanup:  Disable magnetic field cleaning code for resizing, avoids
#             pulling in some unofficial Parthenon code.
# Many machine files have additional options, check machines/machinename.sh

# Make processes to use
# Set conservatively as nvcc/nvc++ uses a *lot* of memory
# Set in environment or override in machine file
NPROC=${NPROC:-8}

### Load machine-specific configurations ###
HOST=$(hostname -f)
if [ -z $HOST ]; then
  HOST=$(hostname)
fi
ARGS="$*"
SOURCE_DIR=$(dirname "$(readlink -f "$0")")
for machine in machines/*.sh
do
  source $machine
done

# If we haven't special-cased already, guess an architecture
# This only works with newer Kokkos, it's always best to
# specify HOST_ARCH in a machine file once you know it.
if [[ -z "$HOST_ARCH" ]]; then
  HOST_ARCH="NATIVE"
fi
EXTRA_FLAGS="-DKokkos_ARCH_${HOST_ARCH}=ON $EXTRA_FLAGS"

# Kokkos does *not* support compiling for multiple devices!
# But if they ever do, you can separate a list of DEVICE_ARCH
# with commas.
if [[ -v DEVICE_ARCH ]]; then
  readarray -t arch_array < <(awk -F',' '{ for( i=1; i<=NF; i++ ) print $i }' <<<"$DEVICE_ARCH")
  for arch in "${arch_array[@]}"; do
    EXTRA_FLAGS="-DKokkos_ARCH_${arch}=ON $EXTRA_FLAGS"
  done
fi
if [[ "$ARGS" == *"trace"* ]]; then
  EXTRA_FLAGS="-DKHARMA_TRACE=1 $EXTRA_FLAGS"
fi
if [[ "$ARGS" == *"nompi"* ]]; then
  EXTRA_FLAGS="-DKHARMA_DISABLE_MPI=1 $EXTRA_FLAGS"
fi
if [[ "$ARGS" == *"noimplicit"* ]]; then
  EXTRA_FLAGS="-DKHARMA_DISABLE_IMPLICIT=1 $EXTRA_FLAGS"
fi
if [[ "$ARGS" == *"nocleanup"* ]]; then
  EXTRA_FLAGS="-DKHARMA_DISABLE_CLEANUP=1 $EXTRA_FLAGS"
fi

### Enivoronment Prep ###
if [[ "$(which python3 2>/dev/null)" == *"conda"* ]]; then
  echo
  echo "make.sh note:"
  echo "It looks like you have Anaconda loaded."
  echo "Anaconda loads a serial version of HDF5 which may make this compile impossible."
  echo "If you run into trouble, deactivate your environment with 'conda deactivate'"
fi
# Save arguments if we've changed them
# Used in run.sh for loading the same modules/etc.
if [[ "$ARGS" == *"clean"* ]]; then
  echo "$ARGS" > $SOURCE_DIR/make_args
fi
# Choose configuration
if [[ "$ARGS" == *"debug"* ]]; then
  TYPE=Debug
else
  TYPE=Release
fi

### Set KHARMA Flags ###
SCRIPT_DIR=$( dirname "$0" )
cd $SCRIPT_DIR
SCRIPT_DIR=$PWD

# Generally best to set CXX_NATIVE yourself if you want to be sure,
# but we try to be smart about loading the most specific/advanced/
# capable compiler available in PATH.
if [[ -z "$CXX_NATIVE" ]]; then
  # If Cray environment is loaded, use their wrappers
  if which CC >/dev/null 2>&1; then
    CXX_NATIVE=CC
    C_NATIVE=cc
    # In case this isn't Cray, use the more common flag
    OMP_FLAG="-fopenomp"
  # Prefer Intel oneAPI compiler over legacy, both over generic
  elif which icpx >/dev/null 2>&1; then
    CXX_NATIVE=icpx
    C_NATIVE=icx
    OMP_FLAG="-fiopenmp"
  elif which icpc >/dev/null 2>&1; then
    CXX_NATIVE=icpc
    C_NATIVE=icc
    OMP_FLAG="-qopenmp"
  # Prefer NVHPC over generic compilers
  elif which nvc++ >/dev/null 2>&1; then
    CXX_NATIVE=nvc++
    C_NATIVE=nvc
    OMP_FLAG="-mp"
  # Maybe we overwrote 'c++' to point to something
  # Usually this is GCC on Linux systems, which is fine
  elif which cpp >/dev/null 2>&1; then
    CXX_NATIVE=c++
    C_NATIVE=cc
    OMP_FLAG="-fopenmp"
  # Otherwise, trusty system GCC
  else
    CXX_NATIVE=g++
    C_NATIVE=gcc
    OMP_FLAG="-fopenmp"
  fi
  # clang/++ will never be used automatically;
  # blame Apple, who don't support OpenMP
fi
# Disable OpenMP for HIP compiles, it gets confused
# and thinks we want to use OMP 5.0 offload stuff
if [[ "$ARGS" != *"hip"* ]]; then
  export CXXFLAGS="$OMP_FLAG $CXXFLAGS"
fi

# Set compilers
# Options are named different so we can override w/wrapper for CUDA
export CXX="$CXX_NATIVE"
export CC="$C_NATIVE"

# CUDA loop options: MANUAL1D_LOOP > MDRANGE_LOOP, TPTTR_LOOP & TPTTRTVR_LOOP don't compile
# Inner loop must be TVR_INNER_LOOP
# OpenMP loop options for KNL:
# Outer: SIMDFOR_LOOP;MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTVR_LOOP;TPTTRTVR_LOOP
# Inner: SIMDFOR_INNER_LOOP;TVR_INNER_LOOP
if [[ "$ARGS" == *"sycl"* ]]; then
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="ON"
  ENABLE_HIP="OFF"
elif [[ "$ARGS" == *"hip"* ]]; then
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="ON"
elif [[ "$ARGS" == *"cuda"* ]]; then
  export CXX="$SCRIPT_DIR/bin/nvcc_wrapper"
  if [[ "$ARGS" == *"wrapper_dryrun"* ]]; then
    export CXXFLAGS="-dryrun $CXXFLAGS"
    echo "Dry-running the nvcc wrapper with $CXXFLAGS"
  fi
  export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX_NATIVE"
  # Generally Kokkos sets this, so we don't need to
  #export CXXFLAGS="--expt-relaxed-constexpr $CXXFLAGS"
  # New NVHPC complains if we don't set this
  export NVHPC_CUDA_HOME=$CUDA_HOME
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="ON"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
elif [[ "$ARGS" == *"nvc++"* ]]; then
  OUTER_LAYOUT="MANUAL1D_LOOP"
  INNER_LAYOUT="TVR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="ON"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
else
  OUTER_LAYOUT="MDRANGE_LOOP"
  INNER_LAYOUT="SIMDFOR_INNER_LOOP"
  ENABLE_OPENMP="ON"
  ENABLE_CUDA="OFF"
  ENABLE_SYCL="OFF"
  ENABLE_HIP="OFF"
fi

# Allow for a custom linker program, but use CXX by
# default as system linker may be older/incompatible
if [[ -v LINKER ]]; then
  EXTRA_FLAGS="-DCMAKE_LINKER=$LINKER"
fi
if [[ "$ARGS" == *"special_link_line"* ]]; then
  EXTRA_FLAGS="-DCMAKE_CXX_LINK_EXECUTABLE='<CMAKE_LINKER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>'"
fi

# Avoid warning on nvcc pragmas Intel doesn't like
if [[ $CXX == "icpc" ]]; then
  export CXXFLAGS="-Wno-unknown-pragmas $CXXFLAGS"
fi
# Avoid icpx's astonishing DEFAULT -ffast-math
if [[ $CXX == "icpx" ]]; then
  export CXXFLAGS="-fno-fast-math $CXXFLAGS"
fi

### Build HDF5 ###
# If we're building HDF5, do it after we set *all flags*
if [[ "$ARGS" == *"hdf5"* && "$ARGS" == *"clean"* ]]; then
  H5VER=1.12.2
  H5VERU=1_12_2
  cd external
  # Allow complete reconfigure (for switching compilers, takes longer)
  if [[ "$ARGS" == *"cleanhdf5"* ]]; then
    rm -rf hdf5-${H5VER}/
  fi
  # Download if needed
  if [ ! -f hdf5-${H5VER}.tar.gz ]; then
    curl https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_${H5VERU}/source/hdf5-${H5VER}.tar.gz -o hdf5-${H5VER}.tar.gz
  fi
  # Unpack if needed (or deleted)
  if [ ! -d hdf5-${H5VER}/ ]; then
    tar xf hdf5-${H5VER}.tar.gz
  fi
  cd hdf5-${H5VER}/
  # TODO better ensure we're using C_NATIVE underneath.  e.g. MPI_CFLAGS with -cc
  if  [[ "$ARGS" == *"nompi"* ]]; then
    HDF_CC=$C_NATIVE
    HDF_EXTRA=""
  else
    if [[ "$ARGS" == *"icc"* ]]; then
      HDF_CC=mpiicc
      HDF_EXTRA="--enable-parallel"
    else
      # Cray wrappers include MPI
      if [[ "$C_NATIVE" == "cc" ]]; then
        HDF_CC=cc
      else
        HDF_CC=mpicc
      fi
      HDF_EXTRA="--enable-parallel"
    fi
  fi

  echo Configuring HDF5...

  export CFLAGS="-fPIC $CFLAGS"
  CC=$HDF_CC sh configure -C $HDF_EXTRA --prefix=$SOURCE_DIR/external/hdf5 --enable-build-mode=production \
  --disable-dependency-tracking --disable-hl --disable-tests --disable-tools --disable-shared --disable-deprecated-symbols > build-hdf5.log
  sleep 1

  echo "Building HDF5 (probably 30s-2min)"
  # Compiling C takes less memory
  if [[ -v $NPROC ]]; then
    make -j$(( $NPROC * 2 )) >> build-hdf5.log 2>&1
  else
    make -j >> build-hdf5.log 2>&1
  fi
  make install >> build-hdf5.log 2>&1
  make clean >> build-hdf5.log 2>&1
  cd ../..

  echo Built HDF5
fi
if [[ "$ARGS" == *"hdf5"* ]]; then
  PREFIX_PATH="$SOURCE_DIR/external/hdf5;$PREFIX_PATH"
fi

### Build KHARMA ###
# If we're doing a clean build, prep the source and
# delete the build directory
if [[ "$ARGS" == *"clean"* ]]; then

  cd external/parthenon
  if [[ $(( $(git --version | cut -d '.' -f 2) > 35 )) ]]; then
    git apply --quiet ../patches/parthenon-*.patch
  else
    echo "make.sh note: You may see errors applying patches below. These are normal."
    git apply ../patches/parthenon-*.patch
  fi
  cd -

  rm -rf build
fi
mkdir -p build
cd build

if [[ "$ARGS" == *"clean"* ]]; then

  if [[ "$ARGS" == *"dryrun"* ]]; then
    set -x
  fi

  cmake ..\
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH;$CMAKE_PREFIX_PATH" \
    -DCMAKE_BUILD_TYPE=$TYPE \
    -DPAR_LOOP_LAYOUT=$OUTER_LAYOUT \
    -DPAR_LOOP_INNER_LAYOUT=$INNER_LAYOUT \
    -DKokkos_ENABLE_OPENMP=$ENABLE_OPENMP \
    -DKokkos_ENABLE_CUDA=$ENABLE_CUDA \
    -DKokkos_ENABLE_SYCL=$ENABLE_SYCL \
    -DKokkos_ENABLE_HIP=$ENABLE_HIP \
    $EXTRA_FLAGS

  if [[ "$ARGS" == *"dryrun"* ]]; then
    set +x
    exit
  fi
fi

if [[ "$ARGS" != *"dryrun"* ]]; then
  make -j$NPROC
  cp kharma/kharma.* ..
fi
