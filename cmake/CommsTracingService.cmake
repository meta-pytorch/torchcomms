# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Build the CommsTracingService thrift library.
# This module sets up the thrift service library that NCCLX depends on.

# This requires thrift to be built first
if(NOT TARGET third_party_deps)
    message(FATAL_ERROR "ThirdParty.cmake must be included before CommsTracingService.cmake")
endif()

# Determine install prefix
if(DEFINED ENV{CONDA_PREFIX})
    set(COMMS_TRACING_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE PATH "CommsTracingService install prefix")
else()
    set(COMMS_TRACING_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/conda" CACHE PATH "CommsTracingService install prefix")
endif()

set(COMMS_TRACING_BUILD_DIR "${CMAKE_BINARY_DIR}/comms_tracing_service")
set(COMMS_TRACING_LIB_DIR "${COMMS_TRACING_INSTALL_PREFIX}/lib")
set(COMMS_TRACING_INCLUDE_DIR "${COMMS_TRACING_INSTALL_PREFIX}/include")

# Skip if USE_SYSTEM_LIBS is set
if(USE_SYSTEM_LIBS)
    message(STATUS "USE_SYSTEM_LIBS is set, skipping comms_tracing_service build")
    add_custom_target(comms_tracing_service_deps)
    return()
endif()

# Skip if NCCL_BUILD_SKIP_DEPS is set
if(DEFINED ENV{NCCL_BUILD_SKIP_DEPS})
    message(STATUS "NCCL_BUILD_SKIP_DEPS is set, skipping comms_tracing_service build")
    add_custom_target(comms_tracing_service_deps)
    return()
endif()

message(STATUS "Building comms_tracing_service to ${COMMS_TRACING_INSTALL_PREFIX}")

# Common CMake args
set(COMMS_TRACING_CMAKE_ARGS
    -DCMAKE_PREFIX_PATH=${COMMS_TRACING_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX=${COMMS_TRACING_INSTALL_PREFIX}
    -DCMAKE_MODULE_PATH=${COMMS_TRACING_INSTALL_PREFIX}
    -DCMAKE_INSTALL_LIBDIR=${COMMS_TRACING_LIB_DIR}
    -DCMAKE_INSTALL_INCLUDEDIR=${COMMS_TRACING_INCLUDE_DIR}
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_STANDARD=20
    -DCMAKE_BUILD_TYPE=Release
)

# Build comms_tracing_service
# This is more complex because we need to set up the directory structure and copy files
ExternalProject_Add(comms_tracing_service
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${COMMS_TRACING_BUILD_DIR}/src
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E make_directory ${COMMS_TRACING_BUILD_DIR}/src
        COMMAND ${CMAKE_COMMAND} -E make_directory ${COMMS_TRACING_BUILD_DIR}/src/comms/analyzer/if
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/comms/analyzer/if
            ${COMMS_TRACING_BUILD_DIR}/src/comms/analyzer/if
        COMMAND ${CMAKE_COMMAND} -E copy
            ${COMMS_TRACING_BUILD_DIR}/src/comms/analyzer/if/CMakeLists.txt
            ${COMMS_TRACING_BUILD_DIR}/src/CMakeLists.txt
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_BINARY_DIR}/third-party/thrift/src/thrift-build
            ${COMMS_TRACING_BUILD_DIR}/src/build
        COMMAND ${CMAKE_COMMAND} -S ${COMMS_TRACING_BUILD_DIR}/src
            -B ${COMMS_TRACING_BUILD_DIR}/build
            ${COMMS_TRACING_CMAKE_ARGS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${COMMS_TRACING_BUILD_DIR}/build
    INSTALL_COMMAND ${CMAKE_COMMAND} --install ${COMMS_TRACING_BUILD_DIR}/build
    PREFIX ${COMMS_TRACING_BUILD_DIR}
    DEPENDS third_party_deps
)

add_custom_target(comms_tracing_service_deps
    DEPENDS comms_tracing_service
)
