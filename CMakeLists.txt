# KHARMA CMake top file. Basically exists to include Parthenon and KHARMA CMake files in 1 build with the same options

cmake_minimum_required(VERSION 3.13)
project(kharma LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 14)

# Parthenon says it doesn't need MPI.  It lies.
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
