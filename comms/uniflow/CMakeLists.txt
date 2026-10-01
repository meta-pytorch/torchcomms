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

# Where translated AMD sources land; consumed by uniflow_hipify below. Set before
# including the module so nested directories all agree on one location.
set(UNIFLOW_HIPIFY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipified)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# spdlog: use system install if available, otherwise fetch from GitHub.
option(UNIFLOW_USE_SYSTEM_SPDLOG "Use system-installed spdlog instead of fetching" ON)

# Header-only escape hatch. Building the Python extension for an interpreter
# that ships its own C++ runtime -- a relocatable toolchain rather than the
# distribution's -- must not pull in the distribution's shared spdlog and fmt,
# because loading those alongside that interpreter mixes two C libraries. Point
# these at header trees instead and the extension carries no such dependency.
set(UNIFLOW_SPDLOG_INCLUDE_DIR "" CACHE PATH "Header-only spdlog include directory")
set(UNIFLOW_FMT_INCLUDE_DIR "" CACHE PATH "Header-only fmt include directory")

# They are one setting: spdlog's headers include fmt's, so a header-only spdlog
# with a packaged fmt mixes two copies.
if((UNIFLOW_SPDLOG_INCLUDE_DIR AND NOT UNIFLOW_FMT_INCLUDE_DIR) OR
   (UNIFLOW_FMT_INCLUDE_DIR AND NOT UNIFLOW_SPDLOG_INCLUDE_DIR))
  message(FATAL_ERROR
      "UNIFLOW_SPDLOG_INCLUDE_DIR and UNIFLOW_FMT_INCLUDE_DIR must be set together.")
endif()

if(UNIFLOW_SPDLOG_INCLUDE_DIR)
  # Deliberately not named fmt::fmt / spdlog::spdlog: those belong to the
  # upstream packages, and a parent project that has already imported them would
  # collide with an alias of ours.
  add_library(uniflow_fmt_headers INTERFACE)
  target_include_directories(uniflow_fmt_headers SYSTEM INTERFACE ${UNIFLOW_FMT_INCLUDE_DIR})
  target_compile_definitions(uniflow_fmt_headers INTERFACE FMT_HEADER_ONLY=1)

  # spdlog is header-only unless told to expect a compiled library.
  add_library(uniflow_spdlog_headers INTERFACE)
  target_include_directories(uniflow_spdlog_headers SYSTEM INTERFACE ${UNIFLOW_SPDLOG_INCLUDE_DIR})
  target_compile_definitions(uniflow_spdlog_headers INTERFACE SPDLOG_FMT_EXTERNAL=1)
  target_link_libraries(uniflow_spdlog_headers INTERFACE uniflow_fmt_headers)

  set(UNIFLOW_SPDLOG_TARGET uniflow_spdlog_headers)
  set(UNIFLOW_FMT_TARGET uniflow_fmt_headers)

  # Also at directory scope, not only on the interface targets above: the OBJECT
  # libraries include Logger.h without linking spdlog, so they see its headers
  # through the include path alone. Without the macro spdlog falls back to its
  # vendored fmt, which then collides with the one selected here.
  add_compile_definitions(SPDLOG_FMT_EXTERNAL FMT_HEADER_ONLY)
  link_libraries(uniflow_fmt_headers)
elseif(UNIFLOW_USE_SYSTEM_SPDLOG)
  find_package(spdlog QUIET)
endif()

if(UNIFLOW_SPDLOG_INCLUDE_DIR)
  message(STATUS "spdlog and fmt taken from header trees")
elseif(NOT spdlog_FOUND)
  message(STATUS "spdlog not found on system, fetching from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG v1.15.2
  )
  FetchContent_MakeAvailable(spdlog)
  set(UNIFLOW_SPDLOG_TARGET spdlog::spdlog)
else()
  # TcpController.cpp includes <fmt/core.h> directly, which spdlog's vendored
  # copy cannot satisfy because it lives under spdlog/fmt/bundled. A system
  # spdlog is built against the system fmt anyway, so take both from there and
  # tell spdlog to use it; the link is the symmetric half of the definition.
  find_package(fmt REQUIRED)
  set(UNIFLOW_SPDLOG_TARGET spdlog::spdlog)
  set(UNIFLOW_FMT_TARGET fmt::fmt)
  add_compile_definitions(SPDLOG_FMT_EXTERNAL)
  link_libraries(fmt::fmt)
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
include(UniflowGpuPlatform)

if(CUDAToolkit_FOUND)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# Add spdlog include path globally for distcc compatibility, and so the OBJECT
# libraries that include Logger.h without linking spdlog themselves can resolve
# it whether spdlog was found on the system or fetched.
get_target_property(_spdlog_inc ${UNIFLOW_SPDLOG_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
if(_spdlog_inc)
  include_directories(${_spdlog_inc})
endif()

# TCP is integrated into MultiTransport on AMD only; on NVIDIA the intra-node
# tier is NVLink and TCP is not part of the selection.
if(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
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

uniflow_finalize_hipify()

# Sources
set(UNIFLOW_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Segment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Connection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MultiTransport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Uniflow.cpp
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/UniflowPublicHeaders.cmake)

# Included via add_subdirectory: build STATIC library for parent to consume
if(UNIFLOW_STANDALONE)
  set(UNIFLOW_LIB_TYPE SHARED)
else()
  set(UNIFLOW_LIB_TYPE STATIC)
endif()

if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  set(UNIFLOW_INTRA_NODE_TRANSPORT_OBJECTS $<TARGET_OBJECTS:uniflow_nvlink_transport>)
else()
  set(UNIFLOW_INTRA_NODE_TRANSPORT_OBJECTS $<TARGET_OBJECTS:uniflow_p2p_transport>)
endif()

add_library(uniflow ${UNIFLOW_LIB_TYPE}
    ${UNIFLOW_SOURCES}
    ${UNIFLOW_PUBLIC_HEADERS}
    ${UNIFLOW_CORE_HEADERS}
    ${UNIFLOW_CORE_SOURCES}
    $<TARGET_OBJECTS:uniflow_controller>
    $<TARGET_OBJECTS:uniflow_transport>
    ${UNIFLOW_INTRA_NODE_TRANSPORT_OBJECTS}
    $<TARGET_OBJECTS:uniflow_rdma_transport>
    $<TARGET_OBJECTS:uniflow_cuda>
    $<TARGET_OBJECTS:uniflow_drivers>
    $<TARGET_OBJECTS:uniflow_nvml>
    $<TARGET_OBJECTS:uniflow_ibv>
    $<TARGET_OBJECTS:uniflow_sysfs>
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
    ${UNIFLOW_SPDLOG_TARGET}
    uniflow_executor
    ${UNIFLOW_GPU_LIBRARIES}
)

uniflow_target_hipified(uniflow)

target_link_libraries(uniflow PRIVATE ${CMAKE_DL_LIBS})

if(UNIFLOW_LIB_TYPE STREQUAL SHARED)
  target_link_options(uniflow PRIVATE "LINKER:--no-undefined")
endif()

# Python binding: the uniflow._core extension that the Python package imports.
# Off by default so builds that only embed the C++ library need neither Python
# nor pybind11.
option(UNIFLOW_BUILD_PYTHON "Build the uniflow._core Python extension" OFF)

if(UNIFLOW_BUILD_PYTHON)
  if(NOT UNIFLOW_LIB_TYPE STREQUAL SHARED)
    message(FATAL_ERROR
        "UNIFLOW_BUILD_PYTHON needs a shared uniflow; the extension loads it at import.")
  endif()
  add_subdirectory(python)
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
