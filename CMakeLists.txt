# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.22)
project(torchcomms_ncclx)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/fetchcontent")

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(USE_NCCL "Whether to build NCCL or not" ON)
option(USE_NCCLX "Whether to build NCCLX or not" ON)
option(USE_GLOO "Whether to build Gloo or not" ON)
option(USE_RCCL "Whether to build RCCL or not" OFF)
option(USE_RCCLX "Whether to build RCCLX or not" OFF)
option(USE_TRANSPORT "Whether to build TRANSPORT or not" ON)
message(STATUS "  USE_NCCL : ${USE_NCCL}")
message(STATUS "  USE_NCCLX : ${USE_NCCLX}")
message(STATUS "  USE_GLOO  : ${USE_GLOO}")
message(STATUS "  USE_RCCL  : ${USE_RCCL}")
message(STATUS "  USE_RCCLX  : ${USE_RCCLX}")
message(STATUS "  USE_TRANSPORT  : ${USE_TRANSPORT}")

if(NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(TORCH_CUDA_ARCH_LIST "9.0")
else()
    set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
endif()
message(STATUS "TORCH_CUDA_ARCH_LIST : ${TORCH_CUDA_ARCH_LIST}")

# Find Python and PyTorch
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)


# Helper to glob sources
file(GLOB TORCHCOMMS_SOURCES "comms/torchcomms/*.cpp")
file(GLOB TORCHCOMMS_PYTHON_SOURCES "comms/torchcomms/*Py.cpp")
list(REMOVE_ITEM TORCHCOMMS_SOURCES ${TORCHCOMMS_PYTHON_SOURCES})

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
set(CONDA_LIB "${CONDA_PREFIX}/${LIB_SUFFIX}")
set(CONDA_INCLUDE "${CONDA_PREFIX}/include")
set(TORCH_PYTHON_LIB "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")

# Build third-party dependencies and NCCLX if needed
# The cmake modules handle USE_SYSTEM_LIBS and NCCL_BUILD_SKIP_DEPS internally
if(USE_NCCLX)
    include(ThirdParty)
    include(CommsTracingService)
    include(NCCLX)
endif()

# Set FMT_INCLUDE for use by sub-CMakeLists
if(NOT USE_SYSTEM_LIBS)
    set(FMT_INCLUDE "${CONDA_INCLUDE}")
endif()

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
target_link_libraries(torchcomms PRIVATE ${TORCH_LIBRARIES})
if(USE_SYSTEM_LIBS)
    target_link_libraries(torchcomms PRIVATE
        "-lglog"
        "-lgflags"
        "-lfmt"
    )
else()
    target_include_directories(torchcomms PRIVATE
        ${FMT_INCLUDE}
    )
    target_link_libraries(torchcomms PRIVATE
        "-l:libglog.a"
        "-l:libgflags.a"
        "-l:libfmt.a"
    )
    # Add dependency on third-party build if ncclx is enabled
    if(USE_NCCLX AND TARGET third_party_deps)
        add_dependencies(torchcomms third_party_deps)
    endif()
endif()

# Extension: torchcomms._comms
add_library(torchcomms_comms MODULE ${TORCHCOMMS_PYTHON_SOURCES})
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
if (USE_TRANSPORT)
    include(comms/torchcomms/transport/CMakeLists.txt)
endif()

# Install targets to Python package structure
install(TARGETS torchcomms
    LIBRARY DESTINATION .
)
install(TARGETS torchcomms_comms
    LIBRARY DESTINATION .
)
