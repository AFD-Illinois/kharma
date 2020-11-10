# KHARMA CMake top file. Basically exists to include Parthenon and KHARMA CMake files in 1 build with the same options

cmake_minimum_required(VERSION 3.11)
project(kharma LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 14)

# Parthenon options
set(NUMBER_GHOST_CELLS 4) # TODO: Make this a CMake option and check it
set(COORDINATE_TYPE UniformCartesian) # TODO: Make this an option when more are available

# MDRANGE_LOOP, MANUAL1D_LOOP, SIMDFOR_LOOP
set(PAR_LOOP_LAYOUT "MDRANGE_LOOP")
set(PAR_LOOP_INNER_LAYOUT "SIMDFOR_INNER_LOOP")

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
