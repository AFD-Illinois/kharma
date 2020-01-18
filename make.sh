rm -rf build
mkdir build
cd build

if true; then
export MPICH_CXX=clang++
cmake3 ..\
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_FLAGS="--cuda-path=/usr/local/cuda-9.2 -nocudalib" \
    -DUSE_MPI=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=ON \
    -DKokkos_ARCH_KEPLER35=OFF \
    -DKokkos_ARCH_MAXWELL52=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON
else
export MPICH_CXX=clang++
cmake3 ..\
    -DCMAKE_CXX_COMPILER=clang++ \
    -DUSE_MPI=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ARCH_HSW=ON
fi
make -j VERBOSE=1
