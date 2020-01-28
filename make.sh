# Make script for K/HARM
# Used to decide flags and call cmake
# TODO autodetection?  Machinefiles at least?

if [[ "$*" == *"clean"* ]]; then
  rm -rf build
  mkdir build
fi

cd build

if [[ "$*" == *"clean"* ]]; then
cmake3 ..\
    -DCMAKE_CXX_COMPILER=$PWD/../external/kokkos/bin/nvcc_wrapper \
    -DUSE_MPI=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=OFF \
    -DKokkos_ARCH_BDW=ON \
    -DKokkos_ARCH_KNL=OFF \
    -DKokkos_ARCH_POWER9=OFF \
    -DKokkos_ARCH_KEPLER35=OFF \
    -DKokkos_ARCH_MAXWELL52=OFF \
    -DKokkos_ARCH_VOLTA70=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON
fi

make -j
cp kharm/kharm.* ..

if [[ "$*" == *"run"* ]]; then
    cd ..
    ./run.sh
fi