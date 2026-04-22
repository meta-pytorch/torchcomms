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

find_package(Threads REQUIRED)

# spdlog: use system install if available, otherwise fetch from GitHub.
option(UNIFLOW_USE_SYSTEM_SPDLOG "Use system-installed spdlog instead of fetching" ON)

if(UNIFLOW_USE_SYSTEM_SPDLOG)
  find_package(spdlog QUIET)
endif()

if(NOT spdlog_FOUND)
  message(STATUS "spdlog not found on system, fetching from GitHub...")
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

# Enable position-independent code for all targets so uniflow can be
# linked into shared libraries (e.g., torchcomms _transport.so).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add CUDA include path globally so all sub-targets can find cuda.h
# and cuda_runtime_api.h, even when compiled via distcc.
if(CUDAToolkit_FOUND)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# Add spdlog include path globally for distcc compatibility.
if(spdlog_FOUND)
  get_target_property(_spdlog_inc spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
  if(_spdlog_inc)
    include_directories(${_spdlog_inc})
  endif()
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
    $<TARGET_OBJECTS:uniflow_controller>
    $<TARGET_OBJECTS:uniflow_transport>
    $<TARGET_OBJECTS:uniflow_nvlink_transport>
    $<TARGET_OBJECTS:uniflow_rdma_transport>
    $<TARGET_OBJECTS:uniflow_cuda>
    $<TARGET_OBJECTS:uniflow_nvml>
    $<TARGET_OBJECTS:uniflow_ibv>
    $<TARGET_OBJECTS:uniflow_sysfs>
    $<TARGET_OBJECTS:uniflow_logging>
)

target_compile_features(uniflow PUBLIC cxx_std_20)

target_include_directories(uniflow PUBLIC
    ${ROOT}
)

target_link_libraries(uniflow PUBLIC
    Threads::Threads
    spdlog::spdlog
    uniflow_executor
)

if(CUDAToolkit_FOUND)
    target_link_libraries(uniflow PUBLIC CUDA::cudart CUDA::cuda_driver)
endif()

# Tests (standalone builds or when BUILD_TESTING is ON)
if(UNIFLOW_STANDALONE OR BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
  add_subdirectory(executor/tests)
  add_subdirectory(core/tests)
  add_subdirectory(logging/tests)
endif()
