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

AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR} EXE_NAME_SRC)

add_executable(${EXE_NAME}
    ${EXE_NAME_SRC})

target_link_libraries(${EXE_NAME} kokkos)
target_link_libraries(${EXE_NAME} hdf5)

# FLAGS
set(CMAKE_CXX_FLAGS_DEBUG "-Wall")
if(CMAKE_BUILD_TYPE)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        message("Debug build")
        target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=1)
    else()
        message("Explicit non-Debug build")
        target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=0)
    endif()
else()
    message("Implicit non-Debug build")
    target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=0)
endif()

if (USE_MPI)
    target_compile_definitions(${EXE_NAME} PUBLIC USE_MPI)
    target_link_libraries(${EXE_NAME} ${MPI_C_LIBRARIES})
endif()

if (Kokkos_ENABLE_CUDA)
    target_compile_definitions(${EXE_NAME} PUBLIC Kokkos_ENABLE_CUDA)
endif()

if (Kokkos_ENABLE_HWLOC)
    target_compile_definitions(${EXE_NAME} PUBLIC Kokkos_ENABLE_HWLOC)
    target_link_libraries(${EXE_NAME} hwloc)
endif()
