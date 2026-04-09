# Copyright (c) Meta Platforms, Inc. and affiliates.

# Standalone project support (for OSS builds)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(uniflow LANGUAGES CXX)
  if(DEFINED ENV{ROOT})
    set(ROOT $ENV{ROOT})
  else()
    set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
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

# Standalone: build shared library (libuniflow.so)
# Included via add_subdirectory: build OBJECT library for parent to consume
if(UNIFLOW_STANDALONE)
  set(UNIFLOW_LIB_TYPE SHARED)
else()
  set(UNIFLOW_LIB_TYPE OBJECT)
endif()

add_library(uniflow ${UNIFLOW_LIB_TYPE}
    ${UNIFLOW_SOURCES}
    ${UNIFLOW_HEADERS}
    ${UNIFLOW_CORE_HEADERS}
    $<TARGET_OBJECTS:uniflow_controller>
    $<TARGET_OBJECTS:uniflow_transport>
    $<TARGET_OBJECTS:uniflow_nvlink_transport>
    $<TARGET_OBJECTS:uniflow_rdma_transport>
    $<TARGET_OBJECTS:uniflow_nvml>
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

# Tests (standalone builds or when BUILD_TESTING is ON)
if(UNIFLOW_STANDALONE OR BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
  add_subdirectory(executor/tests)
  add_subdirectory(core/tests)
  add_subdirectory(logging/tests)
endif()
