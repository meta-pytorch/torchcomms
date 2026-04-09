# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.22)
project(torchcomms_ncclx)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(FETCHCONTENT_BASE_DIR "build/fetchcontent")

# Options
option(USE_NCCL "Whether to build NCCL or not" ON)
option(USE_NCCLX "Whether to build NCCLX or not" ON)
option(USE_GLOO "Whether to build Gloo or not" ON)
option(USE_RCCL "Whether to build RCCL or not" OFF)
option(USE_RCCLX "Whether to build RCCLX or not" OFF)
option(USE_XCCL "Whether to build XCCL or not" OFF)
option(USE_UCC "Whether to build UCC or not" OFF)
option(USE_MPI "Whether to build MPI or not" OFF)
option(USE_TRANSPORT "Whether to build TRANSPORT or not" ON)
option(USE_TRITON "Whether to build Triton device bitcode or not" OFF)
option(BUILD_TESTS "Whether to build tests or not" OFF)
message(STATUS "  USE_NCCL : ${USE_NCCL}")
message(STATUS "  USE_NCCLX : ${USE_NCCLX}")
message(STATUS "  USE_GLOO  : ${USE_GLOO}")
message(STATUS "  USE_RCCL  : ${USE_RCCL}")
message(STATUS "  USE_RCCLX  : ${USE_RCCLX}")
message(STATUS "  USE_XCCL  : ${USE_XCCL}")
message(STATUS "  USE_UCC   : ${USE_UCC}")
message(STATUS "  USE_MPI   : ${USE_MPI}")
message(STATUS "  USE_TRANSPORT  : ${USE_TRANSPORT}")
message(STATUS "  USE_TRITON     : ${USE_TRITON}")
message(STATUS "  BUILD_TESTS    : ${BUILD_TESTS}")

if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
else()
    # Auto-detect from nvidia-smi, fall back to 9.0.
    # Without this, .cu kernels (e.g. AtomicAddKernel.cu) are compiled only
    # for sm_90 and fail at runtime on other GPUs with
    # "no kernel image is available for execution on the device".
    # See also: comms/torchcomms/triton/CMakeLists.txt for the same pattern.
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _smi_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _smi_result
    )
    if(_smi_result EQUAL 0 AND _smi_output)
        # Deduplicate compute capabilities across all GPUs
        # (e.g. "9.0\n9.0\n9.0\n9.0" -> "9.0")
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

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# Find Python and PyTorch
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Auto-detect PyTorch cmake path from installed torch package
if(NOT DEFINED CMAKE_PREFIX_PATH OR NOT CMAKE_PREFIX_PATH MATCHES "torch")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
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

# Helper to glob sources
file(GLOB TORCHCOMMS_SOURCES "comms/torchcomms/*.cpp")
file(GLOB TORCHCOMMS_DUMMY_SOURCES "comms/torchcomms/dummy/*.cpp")
file(GLOB TORCHCOMMS_PYTHON_SOURCES "comms/torchcomms/*Py.cpp")
file(GLOB HOOKS_SOURCES "comms/torchcomms/hooks/*.cpp")
file(GLOB HOOKS_PYTHON_SOURCES "comms/torchcomms/hooks/*Py.cpp")
file(GLOB UTILS_SOURCES "comms/torchcomms/utils/*.cpp")
file(GLOB UTILS_PYTHON_SOURCES "comms/torchcomms/utils/*Py.cpp")

list(REMOVE_ITEM TORCHCOMMS_SOURCES ${TORCHCOMMS_PYTHON_SOURCES})
list(REMOVE_ITEM HOOKS_SOURCES ${HOOKS_PYTHON_SOURCES})
list(REMOVE_ITEM UTILS_SOURCES ${UTILS_PYTHON_SOURCES})

list(APPEND TORCHCOMMS_SOURCES ${TORCHCOMMS_DUMMY_SOURCES})

list(APPEND TORCHCOMMS_SOURCES ${HOOKS_SOURCES})
list(APPEND TORCHCOMMS_SOURCES ${UTILS_SOURCES})
list(APPEND TORCHCOMMS_PYTHON_SOURCES ${HOOKS_PYTHON_SOURCES})

# Set include directories
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(USE_SYSTEM_LIBS $ENV{USE_SYSTEM_LIBS})

message(STATUS "Python3_ROOT_DIR : ${Python3_ROOT_DIR}")

# Find CONDA_PREFIX
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
set(TORCH_PYTHON_LIB "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")

# Core libtorchcomms
add_library(torchcomms SHARED ${TORCHCOMMS_SOURCES})
set_target_properties(torchcomms PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)

target_include_directories(torchcomms PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms PRIVATE cxx_std_20)
target_link_directories(torchcomms PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
if(USE_SYSTEM_LIBS)
    target_link_libraries(torchcomms PRIVATE
        "-lglog"
        "-lgflags"
        "-lfmt"
    )
else()
    target_include_directories(torchcomms PRIVATE
        ${ROOT}/third-party/fmt/include
    )
    target_link_libraries(torchcomms PRIVATE
        "-l:libglog.a"
        "-l:libgflags.a"
        "-l:libfmt.a"
    )
endif()

# Flight Recorder sources (compiled into _comms)
file(GLOB TORCHCOMMS_FR_SOURCES "comms/torchcomms/hooks/fr/FlightRecorder.cpp")
file(GLOB TORCHCOMMS_FR_PYTHON_SOURCES "comms/torchcomms/hooks/fr/FlightRecorderPy.cpp")

# Extension: torchcomms._comms
add_library(torchcomms_comms MODULE
    ${TORCHCOMMS_PYTHON_SOURCES}
    ${HOOKS_PYTHON_SOURCES}
    ${TORCHCOMMS_FR_SOURCES}
    ${TORCHCOMMS_FR_PYTHON_SOURCES}
)
set_target_properties(torchcomms_comms PROPERTIES
    PREFIX ""  # No 'lib' prefix for Python extension
    OUTPUT_NAME "_comms"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)
target_include_directories(torchcomms_comms PRIVATE
    ${ROOT}
    ${CONDA_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_comms PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    torchcomms
    nlohmann_json::nlohmann_json
)

if (USE_NCCL)
    include(comms/torchcomms/nccl/CMakeLists.txt)
endif()
if (USE_NCCLX)
    include(comms/torchcomms/ncclx/CMakeLists.txt)
endif()
if (USE_GLOO)
    include(comms/torchcomms/gloo/CMakeLists.txt)
endif()
if (USE_RCCL)
    include(comms/torchcomms/rccl/CMakeLists.txt)
endif()
if (USE_RCCLX)
    include(comms/torchcomms/rcclx/CMakeLists.txt)
endif()
if (USE_XCCL)
    include(comms/torchcomms/xccl/CMakeLists.txt)
endif()
if (USE_UCC)
    include(comms/torchcomms/ucc/CMakeLists.txt)
endif()
if (USE_MPI)
    include(comms/torchcomms/mpi/CMakeLists.txt)
endif()
if (USE_TRANSPORT)
    include(comms/torchcomms/transport/CMakeLists.txt)
endif()
if (USE_TRITON)
    include(comms/torchcomms/triton/CMakeLists.txt)
endif()
# Install targets to Python package structure
install(TARGETS torchcomms
    LIBRARY DESTINATION .
)
install(TARGETS torchcomms_comms
    LIBRARY DESTINATION .
)

# Build tests
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(comms/torchcomms/tests/unit/cpp)
endif()
