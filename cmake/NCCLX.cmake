# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Build NCCLX using the make-based build system.
# This module wraps the NCCLX Makefile build via add_custom_target.

# NCCLX source directory
set(NCCLX_HOME "${CMAKE_SOURCE_DIR}/comms/ncclx/stable" CACHE PATH "NCCLX source directory")

# NCCLX build directory
set(NCCLX_BUILD_DIR "${CMAKE_BINARY_DIR}/ncclx" CACHE PATH "NCCLX build output directory")

# CUDA settings
if(DEFINED ENV{CUDA_HOME})
    set(NCCLX_CUDA_HOME "$ENV{CUDA_HOME}")
else()
    set(NCCLX_CUDA_HOME "/usr/local/cuda")
endif()

# Get CUDA version to determine arch support
execute_process(
    COMMAND ${NCCLX_CUDA_HOME}/bin/nvcc --version
    OUTPUT_VARIABLE NVCC_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" CUDA_VERSION_MATCH "${NVCC_VERSION_OUTPUT}")
set(CUDA_MAJOR "${CMAKE_MATCH_1}")
set(CUDA_MINOR "${CMAKE_MATCH_2}")

# Determine NVCC architectures
if(DEFINED ENV{NVCC_ARCH})
    set(NCCLX_NVCC_ARCH "$ENV{NVCC_ARCH}")
else()
    set(NCCLX_NVCC_ARCH "a100,h100")
    # Add b200 support if CUDA 12.8+
    if(CUDA_MAJOR GREATER 12 OR (CUDA_MAJOR EQUAL 12 AND CUDA_MINOR GREATER_EQUAL 8))
        set(NCCLX_NVCC_ARCH "${NCCLX_NVCC_ARCH},b200")
    endif()
endif()

# Convert arch names to NVCC gencode flags
set(NVCC_GENCODE "")
string(REPLACE "," ";" ARCH_LIST "${NCCLX_NVCC_ARCH}")
foreach(arch ${ARCH_LIST})
    if(arch STREQUAL "p100")
        string(APPEND NVCC_GENCODE " -gencode=arch=compute_60,code=sm_60")
    elseif(arch STREQUAL "v100")
        string(APPEND NVCC_GENCODE " -gencode=arch=compute_70,code=sm_70")
    elseif(arch STREQUAL "a100")
        string(APPEND NVCC_GENCODE " -gencode=arch=compute_80,code=sm_80")
    elseif(arch STREQUAL "h100")
        string(APPEND NVCC_GENCODE " -gencode=arch=compute_90,code=sm_90")
    elseif(arch STREQUAL "b200")
        string(APPEND NVCC_GENCODE " -gencode=arch=compute_100,code=sm_100")
    endif()
endforeach()
string(STRIP "${NVCC_GENCODE}" NVCC_GENCODE)

# Use environment variable if set
if(DEFINED ENV{NVCC_GENCODE})
    set(NVCC_GENCODE "$ENV{NVCC_GENCODE}")
endif()

# Other settings
if(DEFINED ENV{NCCL_FP8})
    set(NCCLX_FP8 "$ENV{NCCL_FP8}")
else()
    set(NCCLX_FP8 "1")
endif()

if(DEFINED ENV{NCCL_ENABLE_IN_TRAINER_TUNE})
    set(NCCLX_IN_TRAINER_TUNE "$ENV{NCCL_ENABLE_IN_TRAINER_TUNE}")
else()
    set(NCCLX_IN_TRAINER_TUNE "")
endif()

# Get dev signature
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SHORT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(DEFINED ENV{DEV_SIGNATURE})
    set(NCCLX_DEV_SIGNATURE "$ENV{DEV_SIGNATURE}")
elseif(GIT_SHORT_HASH)
    set(NCCLX_DEV_SIGNATURE "git-${GIT_SHORT_HASH}")
else()
    set(NCCLX_DEV_SIGNATURE "")
endif()

# Install prefix - use CONDA_PREFIX if available
if(DEFINED ENV{CONDA_PREFIX})
    set(NCCLX_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
else()
    set(NCCLX_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/conda")
endif()

set(NCCLX_LIB_SUFFIX "lib" CACHE STRING "Library directory suffix")
set(NCCLX_LIB_DIR "${NCCLX_INSTALL_PREFIX}/${NCCLX_LIB_SUFFIX}")
set(NCCLX_INCLUDE_DIR "${NCCLX_INSTALL_PREFIX}/include")

# Build THIRD_PARTY_LDFLAGS - base flags that don't require pkg-config
set(THRIFT_SERVICE_LDFLAGS "-l:libcomms_tracing_service.a -Wl,--start-group -l:libasync.a -l:libconcurrency.a -l:libthrift-core.a -l:libthriftanyrep.a -l:libthriftcpp2.a -l:libthriftmetadata.a -l:libthriftprotocol.a -l:libthrifttype.a -l:libthrifttyperep.a -l:librpcmetadata.a -l:libruntime.a -l:libserverdbginfo.a -l:libtransport.a -l:libcommon.a -Wl,--end-group -l:libwangle.a -l:libfizz.a -l:libxxhash.a")
set(STATIC_LDFLAGS "-l:libglog.a -l:libgflags.a -l:libboost_context.a -l:libfmt.a -l:libssl.a -l:libcrypto.a")

# Create a build script that handles pkg-config at build time
set(NCCLX_BUILD_SCRIPT "${CMAKE_BINARY_DIR}/build_ncclx.sh")
file(WRITE ${NCCLX_BUILD_SCRIPT} "#!/bin/bash
set -ex

export PKG_CONFIG_PATH=\"${NCCLX_LIB_DIR}/pkgconfig\"
export LDFLAGS=\"-Wl,--allow-shlib-undefined\"

# Get folly LDFLAGS via pkg-config
FOLLY_LDFLAGS=\$(pkg-config --libs --static libfolly 2>/dev/null || echo \"\")

# Build THIRD_PARTY_LDFLAGS
THIRD_PARTY_LDFLAGS=\"${THRIFT_SERVICE_LDFLAGS} \${FOLLY_LDFLAGS} ${STATIC_LDFLAGS}\"

make -C \"${NCCLX_HOME}\" VERBOSE=1 -j \\
    \"\$1\" \\
    BUILDDIR=\"${NCCLX_BUILD_DIR}\" \\
    NVCC_GENCODE=\"${NVCC_GENCODE}\" \\
    CUDA_HOME=\"${NCCLX_CUDA_HOME}\" \\
    NCCL_HOME=\"${NCCLX_HOME}\" \\
    NCCL_SUFFIX=\"x-${NCCLX_DEV_SIGNATURE}\" \\
    NCCL_FP8=\"${NCCLX_FP8}\" \\
    BASE_DIR=\"${CMAKE_SOURCE_DIR}\" \\
    CONDA_INCLUDE_DIR=\"${NCCLX_INCLUDE_DIR}\" \\
    CONDA_LIB_DIR=\"${NCCLX_LIB_DIR}\" \\
    THIRD_PARTY_LDFLAGS=\"\${THIRD_PARTY_LDFLAGS}\" \\
    NCCL_ENABLE_IN_TRAINER_TUNE=\"${NCCLX_IN_TRAINER_TUNE}\" \\
    CUDARTLIB=cudart_static
")
file(CHMOD ${NCCLX_BUILD_SCRIPT} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Build NCCLX using the generated script
add_custom_target(ncclx_build
    COMMAND ${NCCLX_BUILD_SCRIPT} src.build
    WORKING_DIRECTORY ${NCCLX_HOME}
    COMMENT "Building NCCLX with make"
    DEPENDS comms_tracing_service_deps
)

# Install target
add_custom_target(ncclx_install
    COMMAND ${NCCLX_BUILD_SCRIPT} src.install
    WORKING_DIRECTORY ${NCCLX_HOME}
    COMMENT "Installing NCCLX"
    DEPENDS ncclx_build
)

# Export paths for consumers
set(NCCLX_INCLUDE "${NCCLX_BUILD_DIR}/include" CACHE PATH "NCCLX include directory" FORCE)
set(NCCLX_STATIC_LIB "${NCCLX_BUILD_DIR}/lib/libnccl_static.a" CACHE FILEPATH "NCCLX static library" FORCE)
set(NCCLX_SHARED_LIB "${NCCLX_BUILD_DIR}/lib/libnccl.so" CACHE FILEPATH "NCCLX shared library" FORCE)

message(STATUS "NCCLX will be built to: ${NCCLX_BUILD_DIR}")
message(STATUS "NCCLX architectures: ${NCCLX_NVCC_ARCH}")
message(STATUS "NCCLX CUDA home: ${NCCLX_CUDA_HOME}")
