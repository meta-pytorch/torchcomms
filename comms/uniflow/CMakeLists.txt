# Copyright (c) Meta Platforms, Inc. and affiliates.

# Standalone project support (for OSS builds)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(uniflow LANGUAGES CXX)
  if(DEFINED ENV{ROOT})
    get_filename_component(ROOT $ENV{ROOT} ABSOLUTE)
  else()
    get_filename_component(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
  endif()
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  endif()
  set(UNIFLOW_STANDALONE ON)
else()
  set(UNIFLOW_STANDALONE OFF)
endif()

if(NOT UNIFLOW_STANDALONE)
  # Intermediate static libs (uniflow, fetched spdlog) must not land in the
  # parent's CMAKE_ARCHIVE_OUTPUT_DIRECTORY: setup.py points it at the
  # packaged torchcomms/ dir and they would ship in the wheel.
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

find_package(Threads REQUIRED)

# libibverbs/libmlx5 are dlopen'd at runtime, but rdma-core headers are
# needed at build time (mlx5dv.h includes verbs.h).
include(CheckIncludeFileCXX)
check_include_file_cxx("infiniband/mlx5dv.h" UNIFLOW_HAS_RDMA_HEADERS)
if(NOT UNIFLOW_HAS_RDMA_HEADERS)
  message(FATAL_ERROR
      "uniflow requires rdma-core headers (infiniband/verbs.h, mlx5dv.h). "
      "Install libibverbs-dev / rdma-core-devel, or build with USE_UNIFLOW=OFF.")
endif()

# Enable position-independent code for all targets (including fetched
# spdlog) so uniflow can be linked into shared libraries (e.g., the
# uniflow._core Python extension).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# fmt: supplied by the parent torchcomms build (header-only); find or fetch
# when building standalone. Must precede spdlog so a fetched spdlog uses the
# same fmt (SPDLOG_FMT_EXTERNAL) instead of its bundled copy.
if(NOT TARGET fmt::fmt)
  find_package(fmt QUIET)
  if(NOT fmt_FOUND)
    message(STATUS "fmt not found on system, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 11.2.0
    )
    FetchContent_MakeAvailable(fmt)
  endif()
endif()

# spdlog: use system install if available, otherwise fetch from GitHub.
option(UNIFLOW_USE_SYSTEM_SPDLOG "Use system-installed spdlog instead of fetching" ON)

if(UNIFLOW_USE_SYSTEM_SPDLOG)
  find_package(spdlog QUIET)
endif()

if(NOT spdlog_FOUND)
  message(STATUS "spdlog not found on system, fetching from GitHub...")
  set(SPDLOG_FMT_EXTERNAL ON)
  include(FetchContent)
  FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG v1.15.2
  )
  FetchContent_MakeAvailable(spdlog)
endif()

# Make ROOT available as a global include path so all targets
# can resolve #include "comms/uniflow/..." paths.
message(STATUS "ROOT = ${ROOT}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${ROOT})

# Add CUDA include path globally so all sub-targets can find cuda.h
# and cuda_runtime_api.h, even when compiled via distcc.
if(CUDAToolkit_FOUND)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# Add spdlog include path globally for distcc compatibility.
# Check the target, not spdlog_FOUND: fetched spdlog sets only the former.
if(TARGET spdlog::spdlog)
  get_target_property(_spdlog_inc spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
  if(_spdlog_inc)
    include_directories(${_spdlog_inc})
  endif()
endif()

if(CMAKE_HIP_COMPILER)
  set(UNIFLOW_ENABLE_TCP_TRANSPORT ON)
else()
  set(UNIFLOW_ENABLE_TCP_TRANSPORT OFF)
endif()

# Sub-components
add_subdirectory(executor)
add_subdirectory(controller)
add_subdirectory(transport)
add_subdirectory(core)
add_subdirectory(drivers)
add_subdirectory(logging)

# Sources
set(UNIFLOW_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Segment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Connection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MultiTransport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Uniflow.cpp
)

set(UNIFLOW_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Result.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Segment.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Connection.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MultiTransport.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Uniflow.h
)

# Included via add_subdirectory: build STATIC library for parent to consume
if(UNIFLOW_STANDALONE)
  set(UNIFLOW_LIB_TYPE SHARED)
else()
  set(UNIFLOW_LIB_TYPE STATIC)
endif()

add_library(uniflow ${UNIFLOW_LIB_TYPE}
    ${UNIFLOW_SOURCES}
    ${UNIFLOW_HEADERS}
    ${UNIFLOW_CORE_HEADERS}
    ${UNIFLOW_CORE_SOURCES}
    $<TARGET_OBJECTS:uniflow_controller>
    $<TARGET_OBJECTS:uniflow_transport>
    $<TARGET_OBJECTS:uniflow_nvlink_transport>
    $<TARGET_OBJECTS:uniflow_rdma_transport>
    $<TARGET_OBJECTS:uniflow_cuda>
    $<TARGET_OBJECTS:uniflow_nvml>
    $<TARGET_OBJECTS:uniflow_ibv>
    $<TARGET_OBJECTS:uniflow_sysfs>
    $<TARGET_OBJECTS:uniflow_topology_discovery>
    $<TARGET_OBJECTS:uniflow_logging>
)

target_compile_features(uniflow PUBLIC cxx_std_20)

if(UNIFLOW_ENABLE_TCP_TRANSPORT)
  target_sources(uniflow PRIVATE $<TARGET_OBJECTS:uniflow_tcp_transport>)
  target_compile_definitions(
      uniflow PUBLIC UNIFLOW_ENABLE_TCP_TRANSPORT=1)
endif()

target_include_directories(uniflow PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow PUBLIC
    Threads::Threads
    spdlog::spdlog
    fmt::fmt
    uniflow_executor
)

if(CUDAToolkit_FOUND)
    target_link_libraries(uniflow PUBLIC CUDA::cudart CUDA::cuda_driver)
endif()

# Extension: uniflow._core (built by the parent torchcomms build, which
# provides the pybind11/Python helpers; the internal Buck build has its own
# target)
if(UNIFLOW_BUILD_PYTHON_EXT AND NOT UNIFLOW_STANDALONE)
  add_library(uniflow_core MODULE
      ${CMAKE_CURRENT_SOURCE_DIR}/UniflowPy.cpp
  )
  set_target_properties(uniflow_core PROPERTIES
      PREFIX ""  # No 'lib' prefix for Python extension
      OUTPUT_NAME "_core"
      SUFFIX ".${Python3_SOABI}.so"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  target_include_directories(uniflow_core PRIVATE ${Python3_INCLUDE_DIRS})
  torchcomms_use_torch_pybind11(uniflow_core)
  target_compile_features(uniflow_core PRIVATE cxx_std_20)
  target_link_libraries(uniflow_core PRIVATE uniflow)
  # CMAKE_INSTALL_PREFIX is the torchcomms package dir (see setup.py);
  # the uniflow package is its sibling.
  install(TARGETS uniflow_core LIBRARY DESTINATION ../uniflow)
endif()

# Tests (standalone builds or when BUILD_TESTING is ON)
if(UNIFLOW_STANDALONE OR BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
  add_subdirectory(executor/tests)
  add_subdirectory(controller/tests)
  add_subdirectory(core/tests)
  add_subdirectory(logging/tests)
endif()
