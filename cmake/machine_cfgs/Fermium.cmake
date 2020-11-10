#Personal machinefile for 

message(STATUS "Loading machine configuration for OLCF's Summit.\n"
  "Supported MACHINE_VARIANT includes 'cuda', 'mpi', and 'cuda-mpi'\n"
  "This configuration has been tested using the following modules: "
  "module load cuda gcc cmake/3.14.2 python hdf5\n")

# common options
set(Kokkos_ARCH_AMDAVX ON CACHE BOOL "CPU architecture")

# variants
if (${MACHINE_VARIANT} MATCHES "cuda")
  set(Kokkos_ARCH_TURING75 ON CACHE BOOL "GPU architecture")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Cuda")
  set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon/external/Kokkos/bin/nvcc_wrapper CACHE STRING "Use nvcc_wrapper")
endif()
