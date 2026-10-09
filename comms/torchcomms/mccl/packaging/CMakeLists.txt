# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.22)
project(torchcomms_mccl_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)

foreach(required_variable
        TORCHCOMMS_SOURCE_ROOT
        MCCL_SOURCE_ROOT
        TORCHCOMMS_DEPS_ROOT
        TORCHCOMMS_LIBRARY
        MCCL_LIBRARY
        TORCHCOMMS_MCCL_OUTPUT_DIRECTORY)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

if(NOT EXISTS "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/TorchComm.hpp")
    message(FATAL_ERROR "TORCHCOMMS_SOURCE_ROOT is not a TorchComms source tree")
endif()
if(NOT EXISTS "${MCCL_SOURCE_ROOT}/comms/mccl/mccl.h")
    message(FATAL_ERROR "MCCL_SOURCE_ROOT is not an MCCL source tree")
endif()
if(NOT EXISTS "${TORCHCOMMS_LIBRARY}")
    message(FATAL_ERROR "Installed libtorchcomms does not exist")
endif()
if(NOT EXISTS "${MCCL_LIBRARY}")
    message(FATAL_ERROR "Installed libmccl does not exist")
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(ENV{PKG_CONFIG_PATH} "${TORCHCOMMS_DEPS_ROOT}/lib/pkgconfig")
execute_process(
    COMMAND pkg-config --libs --static libfolly
    OUTPUT_VARIABLE TORCHCOMMS_FOLLY_LDFLAGS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCHCOMMS_FOLLY_PKG_RESULT
)
if(NOT TORCHCOMMS_FOLLY_PKG_RESULT EQUAL 0)
    message(FATAL_ERROR "pkg-config for libfolly failed")
endif()
separate_arguments(TORCHCOMMS_FOLLY_LDFLAGS NATIVE_COMMAND
    "${TORCHCOMMS_FOLLY_LDFLAGS_RAW}")
set(TORCHCOMMS_FOLLY_LIBRARY
    "${TORCHCOMMS_DEPS_ROOT}/lib/libfolly.a")
if(NOT EXISTS "${TORCHCOMMS_FOLLY_LIBRARY}")
    message(FATAL_ERROR
        "The staged TorchComms Folly archive does not exist: "
        "${TORCHCOMMS_FOLLY_LIBRARY}")
endif()
# libtorchcomms owns and exports the logging implementation. Keep the plugin's
# private Folly support archive from introducing a second logging runtime. Link
# that archive by exact path so an ambient toolchain cannot substitute another
# Folly ABI after CMake normalizes pkg-config's -L flag.
list(FILTER TORCHCOMMS_FOLLY_LDFLAGS EXCLUDE REGEX "^-L")
list(FILTER TORCHCOMMS_FOLLY_LDFLAGS EXCLUDE REGEX "^-lfolly$")
list(FILTER TORCHCOMMS_FOLLY_LDFLAGS EXCLUDE REGEX "glog")
list(FILTER TORCHCOMMS_FOLLY_LDFLAGS EXCLUDE REGEX "gflags")

set(TORCHCOMMS_CVARS_SOURCE_DIR
    "${TORCHCOMMS_SOURCE_ROOT}/comms/utils/cvars")
set(TORCHCOMMS_GENERATED_CVARS_ROOT "${CMAKE_BINARY_DIR}/generated")
set(TORCHCOMMS_GENERATED_CVARS_DIR
    "${TORCHCOMMS_GENERATED_CVARS_ROOT}/comms/utils/cvars")
file(MAKE_DIRECTORY "${TORCHCOMMS_GENERATED_CVARS_DIR}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "NCCL_CVARS_OUTPUT_DIR=${TORCHCOMMS_GENERATED_CVARS_DIR}"
        "${Python3_EXECUTABLE}"
        "${TORCHCOMMS_CVARS_SOURCE_DIR}/extractcvars.py"
    WORKING_DIRECTORY "${TORCHCOMMS_CVARS_SOURCE_DIR}"
    RESULT_VARIABLE TORCHCOMMS_CVARS_RESULT
    OUTPUT_VARIABLE TORCHCOMMS_CVARS_STDOUT
    ERROR_VARIABLE TORCHCOMMS_CVARS_STDERR
)
if(NOT TORCHCOMMS_CVARS_RESULT EQUAL 0)
    message(FATAL_ERROR
        "TorchComms CVAR generation failed (${TORCHCOMMS_CVARS_RESULT}):\n"
        "${TORCHCOMMS_CVARS_STDOUT}\n${TORCHCOMMS_CVARS_STDERR}")
endif()
foreach(TORCHCOMMS_GENERATED_CVAR nccl_cvars.cc nccl_cvars.h)
    if(NOT EXISTS
       "${TORCHCOMMS_GENERATED_CVARS_DIR}/${TORCHCOMMS_GENERATED_CVAR}")
        message(FATAL_ERROR
            "TorchComms CVAR generation did not produce "
            "${TORCHCOMMS_GENERATED_CVAR}")
    endif()
endforeach()

function(add_build_prefix_map prefix)
    foreach(kind file debug macro)
        set(flag "-f${kind}-prefix-map=${prefix}=.")
        add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:${flag}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>
        )
    endforeach()
endfunction()

add_build_prefix_map("${CMAKE_SOURCE_DIR}")
add_build_prefix_map("${CMAKE_BINARY_DIR}")
add_build_prefix_map("${TORCHCOMMS_SOURCE_ROOT}")
add_build_prefix_map("${MCCL_SOURCE_ROOT}")
add_build_prefix_map("${TORCHCOMMS_DEPS_ROOT}")
add_build_prefix_map("${TORCH_INSTALL_PREFIX}")

set(TORCHCOMMS_MCCL_SOURCES
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/mccl/McclGraphEventTracker.cpp"
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/mccl/TorchCommMCCL.cpp"
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/mccl/TorchCommMCCLCCA.cpp"
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/mccl/TorchCommMCCLPy.cpp"
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/mccl/TorchCommMCCLUtils.cpp"
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/mccl/TorchWorkMCCL.cpp"
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/mccl/TorchWorkMCCLQueue.cpp"
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/device/cuda/CudaApi.cpp"
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/device/cuda/DeviceCounter.cpp"
    "${TORCHCOMMS_SOURCE_ROOT}/comms/torchcomms/device/cuda/AtomicAddKernel.cu"
)

add_library(torchcomms_comms_mccl MODULE ${TORCHCOMMS_MCCL_SOURCES})
set_target_properties(torchcomms_comms_mccl PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../mccl/lib;$ORIGIN/../torch/lib"
    LIBRARY_OUTPUT_DIRECTORY "${TORCHCOMMS_MCCL_OUTPUT_DIRECTORY}"
    OUTPUT_NAME "_comms_mccl"
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".${Python3_SOABI}.so"
)

target_include_directories(torchcomms_comms_mccl PRIVATE
    "${TORCHCOMMS_GENERATED_CVARS_ROOT}"
    "${TORCHCOMMS_SOURCE_ROOT}"
    "${MCCL_SOURCE_ROOT}"
    "${TORCHCOMMS_DEPS_ROOT}/include"
    "${TORCH_INSTALL_PREFIX}/include"
    "${Python3_INCLUDE_DIRS}"
)
if(DEFINED TORCHCOMMS_PYBIND11_INCLUDE_DIR)
    target_include_directories(torchcomms_comms_mccl BEFORE PRIVATE
        "${TORCHCOMMS_PYBIND11_INCLUDE_DIR}"
    )
endif()
if(DEFINED CUDA_MATH_INCLUDE AND NOT "${CUDA_MATH_INCLUDE}" STREQUAL "")
    target_include_directories(torchcomms_comms_mccl PRIVATE
        "${CUDA_MATH_INCLUDE}"
    )
endif()

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(torchcomms_comms_mccl PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_compile_definitions(torchcomms_comms_mccl PRIVATE
    # Keep the plugin's private fmt calls self-contained instead of relying on
    # another extension to export the same fmt ABI into the process namespace.
    FMT_HEADER_ONLY
    FMT_USE_CONSTEXPR
    MCCL_BUILD_PRIMS=1
    MCCL_OSS_BUILD
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO
    SPDLOG_FMT_EXTERNAL
)
get_target_property(TORCHCOMMS_CUDART_LIBRARY CUDA::cudart IMPORTED_LOCATION)
if(NOT TORCHCOMMS_CUDART_LIBRARY)
    message(FATAL_ERROR "Could not resolve the CUDA runtime library")
endif()
get_filename_component(TORCHCOMMS_CUDA_LIBRARY_DIR
    "${TORCHCOMMS_CUDART_LIBRARY}" DIRECTORY)
target_link_options(torchcomms_comms_mccl PRIVATE
    "-L${TORCHCOMMS_CUDA_LIBRARY_DIR}"
    "LINKER:--exclude-libs,ALL"
    "LINKER:-Bsymbolic"
)
target_link_libraries(torchcomms_comms_mccl PRIVATE
    "${TORCHCOMMS_LIBRARY}"
    "${MCCL_LIBRARY}"
    "${TORCHCOMMS_FOLLY_LIBRARY}"
    ${TORCHCOMMS_FOLLY_LDFLAGS}
    ${TORCH_LIBRARIES}
    "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so"
    CUDA::cudart
    atomic
)
