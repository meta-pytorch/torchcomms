# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# CMake build for the standalone `torchcomms-ncclx` wheel.
#
# This produces a single Python extension `torchcomms_ncclx._comms_ncclx.so`
# that links against the already-installed torchcomms package (specifically
# its `libtorchcomms.so`) and the vendored NCCLX static library built by
# `build_ncclx.sh`.

cmake_minimum_required(VERSION 3.22)
project(torchcomms_ncclx_ext LANGUAGES C CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(FETCHCONTENT_BASE_DIR "build/fetchcontent")

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    add_link_options("LINKER:--push-state,--no-as-needed" "-latomic" "LINKER:--pop-state")
endif()

# Resolve the torchcomms repo root. When invoked from setup.py this is
# passed as -DTORCHCOMMS_ROOT=...; fall back to three levels above this
# CMakeLists for a direct cmake invocation.
if(NOT DEFINED TORCHCOMMS_ROOT)
    get_filename_component(TORCHCOMMS_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
set(ROOT "${TORCHCOMMS_ROOT}")
message(STATUS "TORCHCOMMS_ROOT : ${ROOT}")

# CUDA arch auto-detect (mirrors top-level CMakeLists.txt).
if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
else()
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _smi_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _smi_result
    )
    if(_smi_result EQUAL 0 AND _smi_output)
        string(REPLACE "\n" ";" _cc_list "${_smi_output}")
        list(REMOVE_DUPLICATES _cc_list)
        list(JOIN _cc_list ";" TORCH_CUDA_ARCH_LIST)
        message(STATUS "Auto-detected CUDA architectures from nvidia-smi: ${TORCH_CUDA_ARCH_LIST}")
    else()
        set(TORCH_CUDA_ARCH_LIST "9.0")
        message(STATUS "nvidia-smi not available, defaulting CUDA architecture to 9.0")
    endif()
endif()
message(STATUS "TORCH_CUDA_ARCH_LIST : ${TORCH_CUDA_ARCH_LIST}")

set(USE_SYSTEM_LIBS $ENV{USE_SYSTEM_LIBS})

if(DEFINED ENV{BUILD_PREFIX})
    set(CONDA_PREFIX $ENV{PREFIX})
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
else()
    set(CONDA_PREFIX "${ROOT}/build/conda")
    set(ENV{CONDA_PREFIX} "${CONDA_PREFIX}")
endif()
set(CONDA_LIB "${CONDA_PREFIX}/lib")
set(CONDA_INCLUDE "${CONDA_PREFIX}/include")

# Re-use the torchcomms third-party CMake helpers. These pull in glog/gflags/
# fmt/nlohmann_json via find_package + FetchContent fallback, and (when
# USE_SYSTEM_LIBS is unset) run build_ncclx.sh to build the NCCLX static
# lib + supporting third-party deps into $CONDA_PREFIX.
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
set(CMAKE_MODULE_PATH
    ${ROOT}/comms/torchcomms/cmake/third_party
    ${CMAKE_MODULE_PATH})
set(USE_NCCLX ON CACHE BOOL "" FORCE)
include(build_ncclx)
include(nlohmann_json)
include(gflags)
include(glog)
include(fmt)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Auto-detect PyTorch cmake path and locate the installed torchcomms package.
if(NOT DEFINED CMAKE_PREFIX_PATH OR NOT CMAKE_PREFIX_PATH MATCHES "torch")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_CMAKE_PREFIX_PATH)
        list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
        message(STATUS "Auto-detected Torch cmake path: ${TORCH_CMAKE_PREFIX_PATH}")
    endif()
endif()
find_package(Torch REQUIRED)
set(TORCH_PYTHON_LIB "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")

# Locate the installed torchcomms package directory so we can link against
# its libtorchcomms.so. This must be installed before building the ncclx
# wheel (it provides core symbols required by _comms_ncclx).
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import os, torchcomms; print(os.path.dirname(torchcomms.__file__))"
    OUTPUT_VARIABLE TORCHCOMMS_PKG_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _tc_result
)
if(NOT _tc_result EQUAL 0)
    message(FATAL_ERROR
        "torchcomms must be installed before building torchcomms-ncclx. "
        "Install it first: `pip install torchcomms` (or build from source).")
endif()
message(STATUS "Found torchcomms package at: ${TORCHCOMMS_PKG_DIR}")

set(LIBTORCHCOMMS "${TORCHCOMMS_PKG_DIR}/libtorchcomms.so")
if(NOT EXISTS "${LIBTORCHCOMMS}")
    message(FATAL_ERROR
        "libtorchcomms.so not found at ${LIBTORCHCOMMS}. The installed "
        "torchcomms wheel does not include the shared library.")
endif()

# ----- Build the _comms_ncclx extension. -----
enable_language(CUDA)
find_package(CUDA)

file(GLOB TORCHCOMMS_NCCLX_SOURCES "${ROOT}/comms/torchcomms/ncclx/*.cpp")
file(GLOB TORCHCOMMS_CUDA_DEVICE_SOURCES
    "${ROOT}/comms/torchcomms/device/cuda/*.cpp"
    "${ROOT}/comms/torchcomms/device/cuda/*.cu")

if(USE_SYSTEM_LIBS)
    set(NCCLX_INCLUDE "${CONDA_INCLUDE}")
    set(NCCLX_STATIC_LIB "${CONDA_LIB}/libnccl_static.a")
    set(NCCLX_SHARED_LIB "${CONDA_LIB}/libnccl.so")
else()
    set(NCCLX_BUILD_DIR $ENV{BUILDDIR})
    if(NOT NCCLX_BUILD_DIR)
        set(NCCLX_BUILD_DIR "${ROOT}/build/ncclx")
    endif()
    set(NCCLX_INCLUDE "${NCCLX_BUILD_DIR}/include")
    set(NCCLX_STATIC_LIB "${NCCLX_BUILD_DIR}/lib/libnccl_static.a")
    set(NCCLX_SHARED_LIB "${NCCLX_BUILD_DIR}/lib/libnccl.so")
endif()

set(ENV{PKG_CONFIG_PATH} "${CONDA_LIB}/pkgconfig")
execute_process(
    COMMAND pkg-config --libs --static libfolly
    OUTPUT_VARIABLE FOLLY_LDFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PKG_RESULT
)
if(NOT PKG_RESULT EQUAL 0)
    message(FATAL_ERROR "pkg-config for libfolly failed")
endif()
separate_arguments(FOLLY_LDFLAGS NATIVE_COMMAND "${FOLLY_LDFLAGS_RAW}")
list(FILTER FOLLY_LDFLAGS EXCLUDE REGEX "glog")
list(FILTER FOLLY_LDFLAGS EXCLUDE REGEX "gflags")
list(FILTER FOLLY_LDFLAGS EXCLUDE REGEX "libfmt")

if(NOT EXISTS "${NCCLX_INCLUDE}")
    message(FATAL_ERROR "NCCLX include not found at ${NCCLX_INCLUDE}")
endif()

if(EXISTS "${NCCLX_INCLUDE}/nccl_device/core.h")
    message(STATUS "NCCL Device API headers found, enabling device API support")
    set(TORCHCOMMS_HAS_NCCL_DEVICE_API ON)
    file(GLOB TORCHCOMMS_DEVICE_NCCLX_SOURCE
        "${ROOT}/comms/torchcomms/device/ncclx/*.cpp")
else()
    set(TORCHCOMMS_HAS_NCCL_DEVICE_API OFF)
    set(TORCHCOMMS_DEVICE_NCCLX_SOURCE "")
endif()

if("$ENV{ENABLE_PIPES}" STREQUAL "1")
    set(TORCHCOMMS_ENABLE_PIPES ON)
    if(TORCHCOMMS_HAS_NCCL_DEVICE_API)
        file(GLOB TORCHCOMMS_DEVICE_PIPES_SOURCE
            "${ROOT}/comms/torchcomms/device/pipes/*.cpp")
    else()
        set(TORCHCOMMS_DEVICE_PIPES_SOURCE "")
    endif()
else()
    set(TORCHCOMMS_ENABLE_PIPES OFF)
    set(TORCHCOMMS_DEVICE_PIPES_SOURCE "")
endif()

add_library(torchcomms_comms_ncclx MODULE
    ${TORCHCOMMS_NCCLX_SOURCES}
    ${TORCHCOMMS_CUDA_DEVICE_SOURCES}
    ${TORCHCOMMS_DEVICE_NCCLX_SOURCE}
    ${TORCHCOMMS_DEVICE_PIPES_SOURCE}
    ${ROOT}/comms/utils/CudaRAII.cc
    ${ROOT}/comms/utils/GraphCaptureSideStream.cc
)
set_target_properties(torchcomms_comms_ncclx PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_ncclx"
    SUFFIX ".${Python3_SOABI}.so"
)
target_include_directories(torchcomms_comms_ncclx PRIVATE
    ${ROOT}
    ${NCCLX_INCLUDE}
    ${CONDA_INCLUDE}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_comms_ncclx PRIVATE cxx_std_20)

if(TORCHCOMMS_HAS_NCCL_DEVICE_API)
    target_compile_definitions(torchcomms_comms_ncclx
        PRIVATE TORCHCOMMS_HAS_NCCL_DEVICE_API=1)
endif()
if(TORCHCOMMS_ENABLE_PIPES)
    target_compile_definitions(torchcomms_comms_ncclx PRIVATE ENABLE_PIPES=1)
endif()

target_link_directories(torchcomms_comms_ncclx PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms_ncclx PRIVATE
    fmt::fmt
    ${LIBTORCHCOMMS}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
)

if(USE_SYSTEM_LIBS)
    target_link_libraries(torchcomms_comms_ncclx PRIVATE
        ${NCCLX_SHARED_LIB}
        ${FOLLY_LDFLAGS}
        "-lboost_program_options"
        "-lboost_filesystem"
    )
else()
    target_link_libraries(torchcomms_comms_ncclx PRIVATE
        ${NCCLX_STATIC_LIB}
        "-l:libcomms_tracing_service.a"
        "-Wl,--start-group"
        "-l:libasync.a"
        "-l:libconcurrency.a"
        "-l:libthrift-core.a"
        "-l:libthriftanyrep.a"
        "-l:libthriftcpp2.a"
        "-l:libthriftmetadata.a"
        "-l:libthriftprotocol.a"
        "-l:libthrifttype.a"
        "-l:libthrifttyperep.a"
        "-l:librpcmetadata.a"
        "-l:libruntime.a"
        "-l:libserverdbginfo.a"
        "-l:libtransport.a"
        "-l:libcommon.a"
        "-Wl,--end-group"
        "-l:libfolly.a"
        "-l:libwangle.a"
        "-l:libfizz.a"
        "-l:libcrypto.a"
        "-l:libssl.a"
        "-l:libxxhash.a"
        "-l:libboost_filesystem.a"
        "-l:libboost_context.a"
        ${FOLLY_LDFLAGS}
    )
    add_custom_command(
        TARGET torchcomms_comms_ncclx
        PRE_LINK
        COMMAND "${ROOT}/rename_symbols.sh"
                '$<TARGET_OBJECTS:torchcomms_comms_ncclx>'
    )
    add_custom_command(
        TARGET torchcomms_comms_ncclx
        PRE_LINK
        COMMAND "${ROOT}/rename_symbols.sh" "${NCCLX_STATIC_LIB}"
    )
endif()

if(CUDA_FOUND)
    target_link_libraries(torchcomms_comms_ncclx PRIVATE CUDA::cudart)
endif()

# No install(...) — the standalone setup.py drives the build directly with
# CMAKE_LIBRARY_OUTPUT_DIRECTORY pointing at the target wheel location.
