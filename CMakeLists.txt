# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.22)
project(torchcomms_ncclx)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(FETCHCONTENT_BASE_DIR "build/fetchcontent")

# Some fetched third-party projects still declare pre-3.5 policy versions.
# Newer CMake releases reject those unless we set a minimum policy floor.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# colltrace's HRDWRingBufferReader uses __atomic_load_n on a 128-bit value,
# which the compiler lowers to __atomic_load_16 (libatomic). On x86_64 we
# sidestep this in source via an SSE2 MOVDQA specialization (see
# HRDWRingBufferReader.h), so no libatomic dep is needed there. On other
# archs (aarch64, etc.) we fall back to linking libatomic.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    add_link_options("LINKER:--push-state,--no-as-needed" "-latomic" "LINKER:--pop-state")
endif()

# Options
option(USE_NCCL "Whether to build NCCL or not" ON)
option(USE_NCCLX "Whether to build NCCLX or not" ON)
option(USE_GLOO "Whether to build Gloo or not" ON)
option(USE_RCCL "Whether to build RCCL or not" OFF)
option(USE_RCCLX "Whether to build RCCLX or not" OFF)
option(USE_XCCL "Whether to build XCCL or not" OFF)
option(USE_TRANSPORT "Whether to build TRANSPORT or not" ON)
option(USE_TRITON "Whether to build Triton device bitcode or not" OFF)
option(BUILD_TESTS "Whether to build tests or not" OFF)
message(STATUS "  USE_NCCL : ${USE_NCCL}")
message(STATUS "  USE_NCCLX : ${USE_NCCLX}")
message(STATUS "  USE_GLOO  : ${USE_GLOO}")
message(STATUS "  USE_RCCL  : ${USE_RCCL}")
message(STATUS "  USE_RCCLX  : ${USE_RCCLX}")
message(STATUS "  USE_XCCL  : ${USE_XCCL}")
message(STATUS "  USE_TRANSPORT  : ${USE_TRANSPORT}")
message(STATUS "  USE_TRITON     : ${USE_TRITON}")
message(STATUS "  BUILD_TESTS    : ${BUILD_TESTS}")

# NEEDS_CUDA: TRUE when any enabled backend requires CUDA. GLOO and XCCL
# are the only non-CUDA backends; if no CUDA-dependent backend is enabled
# we can skip CUDA-specific setup like TORCH_CUDA_ARCH_LIST detection.
if(USE_NCCL OR USE_NCCLX OR USE_RCCL OR USE_RCCLX OR USE_TRANSPORT OR USE_TRITON)
    set(NEEDS_CUDA TRUE)
else()
    set(NEEDS_CUDA FALSE)
endif()
message(STATUS "  NEEDS_CUDA  : ${NEEDS_CUDA}")

if(NOT NEEDS_CUDA)
    message(STATUS "Skipping CUDA arch detection (no CUDA-dependent backends enabled)")
else()
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
endif()

# Set ROOT early — needed by build_ncclx.cmake and backend CMakeLists.
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(USE_SYSTEM_LIBS $ENV{USE_SYSTEM_LIBS})

# Find CONDA_PREFIX — must be set before third-party includes so find_package
# can discover libs installed by build_ncclx.sh.
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

# Third-party dependencies (find_package with FetchContent fallback)
# Allow FetchContent_Populate() which we use to patch glog before add_subdirectory.
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/comms/torchcomms/cmake/third_party ${CMAKE_MODULE_PATH})
# When USE_NCCLX=ON, build_ncclx.sh installs gflags/glog/fmt/folly/etc. into
# CONDA_PREFIX. Run it first so the find_package calls below find them.
include(build_ncclx)
include(nlohmann_json)
include(gflags)
include(glog)
include(fmt)

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
file(GLOB TORCHCOMMS_FAKE_SOURCES "comms/torchcomms/fake/*.cpp")
file(GLOB TORCHCOMMS_PYTHON_SOURCES "comms/torchcomms/*Py.cpp")
file(GLOB HOOKS_SOURCES "comms/torchcomms/hooks/*.cpp")
file(GLOB HOOKS_PYTHON_SOURCES "comms/torchcomms/hooks/*Py.cpp")
file(GLOB UTILS_SOURCES "comms/torchcomms/utils/*.cpp")
file(GLOB UTILS_PYTHON_SOURCES "comms/torchcomms/utils/*Py.cpp")

list(REMOVE_ITEM TORCHCOMMS_SOURCES ${TORCHCOMMS_PYTHON_SOURCES})
list(REMOVE_ITEM HOOKS_SOURCES ${HOOKS_PYTHON_SOURCES})
list(REMOVE_ITEM UTILS_SOURCES ${UTILS_PYTHON_SOURCES})

list(APPEND TORCHCOMMS_SOURCES ${TORCHCOMMS_FAKE_SOURCES})

list(APPEND TORCHCOMMS_SOURCES ${HOOKS_SOURCES})
list(APPEND TORCHCOMMS_SOURCES ${UTILS_SOURCES})
list(APPEND TORCHCOMMS_PYTHON_SOURCES ${HOOKS_PYTHON_SOURCES})

message(STATUS "Python3_ROOT_DIR : ${Python3_ROOT_DIR}")

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
# glog/gflags/fmt must come before TORCH_LIBRARIES so our headers are found
# before PyTorch's bundled (and potentially different-version) fmt headers.
# PRIVATE: consumer extensions that need these link them explicitly.
# Extensions that link torchcomms (gloo, nccl, ncclx) resolve glog/gflags
# symbols from libtorchcomms.so at runtime via the DT_NEEDED chain.
target_link_libraries(torchcomms PRIVATE glog::glog gflags::gflags fmt::fmt)
target_link_libraries(torchcomms PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)

# Clog Hook sources (compiled into _comms)
file(GLOB TORCHCOMMS_CLOG_SOURCES "comms/torchcomms/hooks/clog/ClogHook.cpp")
file(GLOB TORCHCOMMS_CLOG_PYTHON_SOURCES
    "comms/torchcomms/hooks/clog/ClogHookPy.cpp")

# Flight Recorder sources (compiled into _comms)
file(GLOB TORCHCOMMS_FR_SOURCES "comms/torchcomms/hooks/fr/FlightRecorder.cpp")
file(GLOB TORCHCOMMS_FR_PYTHON_SOURCES "comms/torchcomms/hooks/fr/FlightRecorderPy.cpp")

# Extension: torchcomms._comms
add_library(torchcomms_comms MODULE
    ${TORCHCOMMS_PYTHON_SOURCES}
    ${HOOKS_PYTHON_SOURCES}
    ${TORCHCOMMS_CLOG_SOURCES}
    ${TORCHCOMMS_CLOG_PYTHON_SOURCES}
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
    ${TORCH_INSTALL_PREFIX}/include
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_comms PRIVATE cxx_std_20)
target_link_directories(torchcomms_comms PRIVATE ${CONDA_LIB})
target_link_libraries(torchcomms_comms PRIVATE
    torchcomms
    fmt::fmt
    nlohmann_json::nlohmann_json
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
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
