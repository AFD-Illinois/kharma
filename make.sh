# Make script for K/HARM
# Used to decide flags and call cmake
# TODO autodetection?  Machinefiles at least?

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $SCRIPT_DIR
if [[ "$*" == *"clean"* ]]; then
  rm -rf build kharm.* core.*
fi
mkdir -p build

cd build

# TODO Faaaancy autodetection
export CXX=icpc
export CC=icc

if [[ "$*" == *"clean"* ]]; then
  if false; then # CUDA BUILD
    cmake3 ..\
    -DCMAKE_CXX_COMPILER=$PWD/../external/kokkos/bin/nvcc_wrapper \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MPI=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=ON \
    -DKokkos_ARCH_BDW=OFF \
    -DKokkos_ARCH_POWER9=OFF \
    -DKokkos_ARCH_KEPLER35=ON \
    -DKokkos_ARCH_MAXWELL52=OFF \
    -DKokkos_ARCH_VOLTA70=OFF \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON
  else #KNL BUILD
    cmake ..\
    -DCMAKE_CXX_COMPILER=icpc \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-I/opt/apps/intel18/hdf5/1.10.4/x86_64/include \
                        -L/opt/apps/intel18/hdf5/1.10.4/x86_64/lib -g" \
    -DUSE_MPI=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=OFF \
    -DKokkos_ARCH_BDW=OFF \
    -DKokkos_ARCH_KNL=ON \
    -DKokkos_ARCH_POWER9=OFF
  fi
fi

make -j
cp kharm/kharm.* ..

if [[ "$*" == *"run"* ]]; then
    cd ..
    ./run.sh
fi
