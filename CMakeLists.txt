# KHARMA CMake top file.
# Only for general options, not KHARMA build specifics.
# That is, what would still be relevant if we were building
# multiple codes in this directory?

cmake_minimum_required(VERSION 3.10)
project(kharma LANGUAGES C CXX)

# We follow Parthenon in requiring C++17 going forward
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")
#set(PARTHENON_ENABLE_CPP17 ON CACHE BOOL "KHARMA Override")
set(CMAKE_CXX_STANDARD 14)

# Parthenon options
set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "KHARMA Override")
set(PARTHENON_LINT_DEFAULT OFF CACHE BOOL "KHARMA Override")
# Attempt HDF5 compression, requires recent/standard HDF5. YMMV
set(PARTHENON_DISABLE_HDF5_COMPRESSION OFF CACHE BOOL "KHARMA Override")
# Don't build sparse (selectively-allocated) variable support
set(PARTHENON_DISABLE_SPARSE ON CACHE BOOL "KHARMA Override")

# Parthenon internal build options
set(BUILD_TESTING OFF CACHE BOOL "KHARMA Override")
set(ENABLE_COMPILER_WARNINGS OFF CACHE BOOL "KHARMA Override")
# TODO upstream Parthenon needs support before we do this
#set(COORDINATE_TYPE GRCoordinates)
# Always use static HDF5
set(HDF5_USE_STATIC_LIBRARIES ON CACHE BOOL "KHARMA Override")

# Kokkos options
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "KHARMA Override")
set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "KHARMA Override")
set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "KHARMA Override")
set(Kokkos_ENABLE_HWLOC OFF CACHE BOOL "KHARMA Override") # Possible speed improvement?
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "KHARMA Override")

# If we build KokkosKernels at all, disable the extras
set(KokkosKernels_ENABLE_TPL_CUSPARSE OFF CACHE BOOL "KHARMA Override")
set(KokkosKernels_ENABLE_TPL_CUBLAS OFF CACHE BOOL "KHARMA Override")


# Offer a KHARMA option to disable the MPI requirement
# The only difference from setting PARTHENON_DISABLE_MPI is that
# the configure step no longer searches for/fails without it
if (KHARMA_DISABLE_MPI)
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "KHARMA Override")
else()
  set(PARTHENON_DISABLE_MPI OFF CACHE BOOL "KHARMA Override")
  find_package(MPI REQUIRED)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif()

# OpenMP is strictly required
find_package(OpenMP REQUIRED)

# Build Parthenon
add_subdirectory(external/parthenon)
include_directories(external/parthenon/src)
# mpark::variant is header only, don't build anything
include_directories(external/variant/include)
# Kokkos kernels: don't build them (very slow), just import all headers
# Requires KokkosKernels_config.h shipped with KHARMA, YMMV
# In case of issues, uncomment the following line to build them
#add_subdirectory(external/kokkos-kernels)
include_directories(external/kokkos-kernels/src)
include_directories(external/kokkos-kernels/src/batched)
include_directories(external/kokkos-kernels/src/common)
include_directories(external/kokkos-kernels/src/batched/dense)
include_directories(external/kokkos-kernels/src/batched/dense/impl)
include_directories(external/kokkos-kernels/src/blas)
include_directories(external/kokkos-kernels/src/blas/impl)

# Finally, build KHARMA
add_subdirectory(kharma)
