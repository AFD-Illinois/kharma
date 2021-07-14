# KHARMA CMake top file. Basically exists to include Parthenon and KHARMA CMake files in 1 build with the same options

cmake_minimum_required(VERSION 3.10)
project(kharma LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

# Force C++17
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
# ensure correct flag is used as deprecated c++1z is not supported by nvcc
#set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")

#foreach(path ${CMAKE_PREFIX_PATH})
#  include_directories(${path})
#endforeach()

# Parthenon options
set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "KHARMA Override")
set(PARTHENON_LINT_DEFAULT OFF CACHE BOOL "KHARMA Override")
# To use old Summit built-in HDF5
set(PARTHENON_DISABLE_HDF5_COMPRESSION ON CACHE BOOL "KHARMA Override")
set(PARTHENON_ENABLE_CPP17 ON CACHE BOOL "KHARMA Override")

# Parthenon internal build options
set(BUILD_TESTING OFF CACHE BOOL "KHARMA Override")
set(ENABLE_COMPILER_WARNINGS OFF CACHE BOOL "KHARMA Override")

# Kokkos options
set(Kokkos_ENABLE_OPENMP ON)
set(Kokkos_ENABLE_CUDA_LAMBDA ON)
set(Kokkos_ENABLE_CUDA_CONSTEXPR ON)
set(Kokkos_ENABLE_HWLOC OFF) # Possible speed improvement?
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON)

# TODO set this here when I upstream or otherwise unfork
#set(COORDINATE_TYPE GRCoordinates)

# Parthenon says it doesn't need MPI.  It just *strongly prefers* it, and so do we.
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
find_package(OpenMP REQUIRED)

# TODO don't build parthenon unit tests etc just the library
add_subdirectory(external/parthenon)
include_directories(external/parthenon/src)
# mpark::variant is header only, don't build anything
include_directories(external/variant/include)

# KHARMA folder
add_subdirectory(kharma)
