# include header for main application
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# kharm executable
#
if (Kokkos_ENABLE_CUDA)
  set(EXE_NAME "kharm.cuda")
else()
  set(EXE_NAME "kharm.host")
endif()

set (EXE_NAME_SRC
  kharm.cpp
  decs.hpp
  diffuse.hpp
  )

add_executable(${EXE_NAME}
  ${EXE_NAME_SRC})

target_link_libraries(${EXE_NAME} ${MPI_C_LIBRARIES})
target_link_libraries(${EXE_NAME} kokkos)
target_link_libraries(${EXE_NAME} hwloc)
if (KOKKOS_ENABLE_HWLOC)
  list(APPEND CMAKE_PREFIX_PATH "/usr/lib64")
  target_link_libraries(${EXE_NAME} hwloc)
endif()
