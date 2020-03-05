# KHARMA CMake top file. Basically exists to include Parthenon and KHARMA CMake files in 1 build with the same options

cmake_minimum_required(VERSION 3.13)
project(kharma LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 14)

# TODO don't build parthenon unit tests? Move to external/?
add_subdirectory(parthenon)
add_subdirectory(kharma)
