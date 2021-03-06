#
# kharma executable
#
if (Kokkos_ENABLE_CUDA)
    set(EXE_NAME "kharma.cuda")
else()
    set(EXE_NAME "kharma.host")
endif()

AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR} EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/prob EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/coordinates EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/grmhd EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/b_none EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/b_flux_ct EXE_NAME_SRC)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/b_cd_glm EXE_NAME_SRC)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/prob)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/coordinates)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/grmhd)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/b_none)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/b_flux_ct)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/b_cd_glm)

add_executable(${EXE_NAME} ${EXE_NAME_SRC})

target_link_libraries(${EXE_NAME} PUBLIC parthenon)

# FLAGS
if(CMAKE_BUILD_TYPE)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        message("Debug build")
        target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=1)
        # Only for very verbose output
        #target_compile_definitions(${EXE_NAME} PUBLIC DEBUG_TRACE=1)
    else()
        message("Explicit non-Debug build")
        target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=0)
    endif()
else()
    message("Implicit non-Debug build")
    target_compile_definitions(${EXE_NAME} PUBLIC DEBUG=0)
endif()

if (Kokkos_ENABLE_HWLOC)
    target_compile_definitions(${EXE_NAME} PUBLIC Kokkos_ENABLE_HWLOC)
    target_link_libraries(${EXE_NAME} PUBLIC hwloc)
endif()
