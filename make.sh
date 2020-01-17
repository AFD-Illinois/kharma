rm -rf build
mkdir build
cd build
if false; then
cmake3 ..\
    -DCMAKE_CXX_COMPILER=$PWD/../external/kokkos/bin/nvcc_wrapper \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=ON \
    -DKokkos_ARCH_KEPLER35=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON
else
cmake3 ..\
    -DCMAKE_CXX_COMPILER=g++ \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=ON \
    -DKokkos_ARCH_KEPLER35=OFF \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON
fi
make -j
