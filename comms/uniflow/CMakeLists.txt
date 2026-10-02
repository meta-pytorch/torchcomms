# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.20)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" UNIFLOW_PACKAGE_VERSION)
string(STRIP "${UNIFLOW_PACKAGE_VERSION}" UNIFLOW_PACKAGE_VERSION)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" UNIFLOW_VERSION
  "${UNIFLOW_PACKAGE_VERSION}")
if(NOT UNIFLOW_VERSION)
  message(FATAL_ERROR "Invalid UniFlow version: ${UNIFLOW_PACKAGE_VERSION}")
endif()

# Standalone project support (for OSS builds)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(uniflow VERSION "${UNIFLOW_VERSION}" LANGUAGES CXX)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  endif()
  set(UNIFLOW_STANDALONE ON)
else()
  set(UNIFLOW_STANDALONE OFF)
endif()

find_package(Threads REQUIRED)
include(GNUInstallDirs)

set(UNIFLOW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Where translated AMD sources land; consumed by uniflow_hipify below. Set before
# including the module so nested directories all agree on one location.
set(UNIFLOW_HIPIFY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipified)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(UNIFLOW_BUILD_PYTHON "Build the UniFlow Python extension" OFF)
option(UNIFLOW_BUILD_TESTS "Build UniFlow tests" OFF)
option(UNIFLOW_BUILD_BENCHMARKS "Build UniFlow benchmarks" OFF)

set(UNIFLOW_MINIMUM_CUDA_VERSION 12.8)
include(UniflowGpuPlatform)

if(UNIFLOW_GPU_PLATFORM STREQUAL NONE)
  set(_UNIFLOW_ENABLE_RDMA_DEFAULT OFF)
else()
  set(_UNIFLOW_ENABLE_RDMA_DEFAULT ON)
endif()
option(UNIFLOW_ENABLE_RDMA
  "Build the RDMA transport" ${_UNIFLOW_ENABLE_RDMA_DEFAULT})
unset(_UNIFLOW_ENABLE_RDMA_DEFAULT)

if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  set(_UNIFLOW_ENABLE_NVLINK_DEFAULT ON)
else()
  set(_UNIFLOW_ENABLE_NVLINK_DEFAULT OFF)
endif()
option(UNIFLOW_ENABLE_NVLINK
  "Build the NVLink transport" ${_UNIFLOW_ENABLE_NVLINK_DEFAULT})
unset(_UNIFLOW_ENABLE_NVLINK_DEFAULT)

if(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  set(_UNIFLOW_ENABLE_TCP_DEFAULT ON)
else()
  set(_UNIFLOW_ENABLE_TCP_DEFAULT OFF)
endif()
option(UNIFLOW_ENABLE_TCP
  "Build the TCP transport" ${_UNIFLOW_ENABLE_TCP_DEFAULT})
unset(_UNIFLOW_ENABLE_TCP_DEFAULT)
option(UNIFLOW_USE_SYSTEM_SPDLOG "Use system-installed spdlog and fmt" ON)
option(UNIFLOW_INSTALL_CPP "Install the UniFlow C++ SDK" ON)
option(UNIFLOW_BUILD_SHARED_LIBS
  "Build UniFlow shared libraries" ${UNIFLOW_STANDALONE})

if(UNIFLOW_GPU_PLATFORM STREQUAL NONE AND
   (UNIFLOW_ENABLE_RDMA OR UNIFLOW_ENABLE_NVLINK OR UNIFLOW_ENABLE_TCP))
  message(FATAL_ERROR
    "RDMA, NVLink, and TCP require a GPU platform in this release; disable "
    "them or choose UNIFLOW_GPU_PLATFORM=CUDA|HIP.")
endif()
if(UNIFLOW_GPU_PLATFORM STREQUAL HIP AND UNIFLOW_ENABLE_NVLINK)
  message(FATAL_ERROR
    "NVLink is available only with UNIFLOW_GPU_PLATFORM=CUDA.")
endif()

set(UNIFLOW_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${UNIFLOW_BUILD_INCLUDE_DIR}/comms")
file(
  CREATE_LINK
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${UNIFLOW_BUILD_INCLUDE_DIR}/comms/uniflow"
  SYMBOLIC
  RESULT _UNIFLOW_INCLUDE_LINK_RESULT
)
if(NOT _UNIFLOW_INCLUDE_LINK_RESULT STREQUAL "0")
  message(FATAL_ERROR
    "Failed to create the UniFlow build include symlink: "
    "${_UNIFLOW_INCLUDE_LINK_RESULT}")
endif()
unset(_UNIFLOW_INCLUDE_LINK_RESULT)

set(UNIFLOW_MINIMUM_SPDLOG_VERSION 1.15.2)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
elseif(UNIFLOW_USE_SYSTEM_SPDLOG)
  find_package(spdlog ${UNIFLOW_MINIMUM_SPDLOG_VERSION} QUIET)
endif()

if(UNIFLOW_SPDLOG_INCLUDE_DIR)
  message(STATUS "spdlog and fmt taken from header trees")
elseif(NOT UNIFLOW_USE_SYSTEM_SPDLOG OR NOT spdlog_FOUND)
  message(STATUS "spdlog not found on system, fetching from GitHub...")
  include(FetchContent)
  set(FMT_DOC OFF CACHE BOOL "" FORCE)
  set(FMT_INSTALL ${UNIFLOW_INSTALL_CPP} CACHE BOOL "" FORCE)
  set(FMT_TEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
      fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG 11.1.4
  )
  FetchContent_MakeAvailable(fmt)

  set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)
  set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
  set(SPDLOG_INSTALL ${UNIFLOW_INSTALL_CPP} CACHE BOOL "" FORCE)
  FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG "v${UNIFLOW_MINIMUM_SPDLOG_VERSION}"
  )
  FetchContent_MakeAvailable(spdlog)
  set(UNIFLOW_SPDLOG_TARGET spdlog::spdlog)
  set(UNIFLOW_FMT_TARGET fmt::fmt)
  if(UNIFLOW_INSTALL_CPP)
    install(FILES "${fmt_SOURCE_DIR}/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/fmt"
      COMPONENT Development)
    install(FILES "${spdlog_SOURCE_DIR}/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/spdlog"
      COMPONENT Development)
  endif()
else()
  # TcpController.cpp includes <fmt/core.h> directly, which spdlog's vendored
  # copy cannot satisfy because it lives under spdlog/fmt/bundled. A system
  # spdlog is built against the system fmt anyway, so take both from there.
  find_package(fmt REQUIRED)
  set(UNIFLOW_SPDLOG_TARGET spdlog::spdlog)
  set(UNIFLOW_FMT_TARGET fmt::fmt)
endif()

add_library(uniflow_build_config INTERFACE)
target_compile_features(uniflow_build_config INTERFACE cxx_std_20)
target_include_directories(uniflow_build_config INTERFACE
  $<BUILD_INTERFACE:${UNIFLOW_BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(uniflow_build_config INTERFACE
  UNIFLOW_CMAKE_BUILD=1
  $<$<BOOL:${UNIFLOW_ENABLE_RDMA}>:UNIFLOW_ENABLE_RDMA_TRANSPORT=1>
  $<$<AND:$<STREQUAL:${UNIFLOW_GPU_PLATFORM},CUDA>,$<BOOL:${UNIFLOW_ENABLE_NVLINK}>>:UNIFLOW_ENABLE_NVLINK_TRANSPORT=1>
  $<$<BOOL:${UNIFLOW_ENABLE_TCP}>:UNIFLOW_ENABLE_TCP_TRANSPORT=1>
)
target_link_libraries(uniflow_build_config INTERFACE
  Threads::Threads
  ${UNIFLOW_FMT_TARGET}
  ${UNIFLOW_SPDLOG_TARGET}
)

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

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/UniflowPublicHeaders.cmake)

# Select the concrete library type without changing the caller's global
# BUILD_SHARED_LIBS setting.
if(UNIFLOW_BUILD_SHARED_LIBS)
  set(UNIFLOW_LIB_TYPE SHARED)
else()
  set(UNIFLOW_LIB_TYPE STATIC)
endif()

add_library(uniflow_objects OBJECT
    ${UNIFLOW_SOURCES}
    ${UNIFLOW_PUBLIC_HEADERS}
    ${UNIFLOW_CORE_HEADERS}
    ${UNIFLOW_CORE_SOURCES}
)
target_link_libraries(uniflow_objects PRIVATE uniflow_build_config)
if(NOT UNIFLOW_GPU_PLATFORM STREQUAL NONE)
  target_link_libraries(uniflow_objects PRIVATE ${UNIFLOW_GPU_LIBRARIES})
  uniflow_target_hipified(uniflow_objects)
endif()

add_library(uniflow_implementation INTERFACE)
target_sources(uniflow_implementation INTERFACE
    $<TARGET_OBJECTS:uniflow_objects>
    $<TARGET_OBJECTS:uniflow_controller>
    $<TARGET_OBJECTS:uniflow_logging>
    $<TARGET_OBJECTS:uniflow_transport>
)
target_link_libraries(uniflow_implementation INTERFACE
    uniflow_build_config
    uniflow_controller
    uniflow_drivers
    uniflow_executor
    uniflow_logging
    uniflow_transport
)

if(UNIFLOW_ENABLE_NVLINK AND UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  target_sources(uniflow_implementation INTERFACE
    $<TARGET_OBJECTS:uniflow_nvlink_transport>)
  target_link_libraries(uniflow_implementation INTERFACE
    uniflow_nvlink_transport)
elseif(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  target_sources(uniflow_implementation INTERFACE
    $<TARGET_OBJECTS:uniflow_p2p_transport>)
  target_link_libraries(uniflow_implementation INTERFACE
    uniflow_p2p_transport)
endif()
if(UNIFLOW_ENABLE_RDMA)
  target_sources(uniflow_implementation INTERFACE
    $<TARGET_OBJECTS:uniflow_rdma_transport>)
  target_link_libraries(uniflow_implementation INTERFACE
    uniflow_rdma_transport)
endif()
if(UNIFLOW_ENABLE_TCP)
  target_sources(uniflow_implementation INTERFACE
    $<TARGET_OBJECTS:uniflow_tcp_transport>)
  target_link_libraries(uniflow_implementation INTERFACE
    uniflow_tcp_transport)
endif()

add_library(uniflow ${UNIFLOW_LIB_TYPE})
add_library(uniflow::uniflow ALIAS uniflow)

target_compile_features(uniflow PUBLIC cxx_std_20)

target_include_directories(uniflow PUBLIC
    $<BUILD_INTERFACE:${UNIFLOW_BUILD_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(uniflow PUBLIC
    Threads::Threads
    $<BUILD_INTERFACE:${UNIFLOW_FMT_TARGET}>
    $<BUILD_INTERFACE:${UNIFLOW_SPDLOG_TARGET}>
    $<INSTALL_INTERFACE:fmt::fmt>
    $<INSTALL_INTERFACE:spdlog::spdlog>
)
target_link_libraries(uniflow PRIVATE
    $<BUILD_INTERFACE:uniflow_implementation>)

if(NOT UNIFLOW_GPU_PLATFORM STREQUAL NONE)
  target_link_libraries(uniflow PUBLIC ${UNIFLOW_GPU_LIBRARIES})
  uniflow_target_hipified(uniflow)
  uniflow_finalize_hipify()
endif()

target_link_libraries(uniflow PRIVATE ${CMAKE_DL_LIBS})

if(UNIFLOW_LIB_TYPE STREQUAL SHARED)
  target_link_options(uniflow PRIVATE "LINKER:--no-undefined")
endif()

if(UNIFLOW_BUILD_PYTHON)
  if(NOT UNIFLOW_LIB_TYPE STREQUAL SHARED)
    message(FATAL_ERROR
        "UNIFLOW_BUILD_PYTHON needs a shared uniflow; the extension loads it at import.")
  endif()
  add_subdirectory(python)
endif()

if(UNIFLOW_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

if(UNIFLOW_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(UNIFLOW_INSTALL_CPP)
  include(CMakePackageConfigHelpers)

  install(TARGETS uniflow
    EXPORT uniflowTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Runtime
      NAMELINK_COMPONENT Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  foreach(_UNIFLOW_PUBLIC_HEADER IN LISTS UNIFLOW_PUBLIC_HEADERS)
    file(RELATIVE_PATH _UNIFLOW_HEADER_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}" "${_UNIFLOW_PUBLIC_HEADER}")
    get_filename_component(_UNIFLOW_HEADER_DIRECTORY
      "${_UNIFLOW_HEADER_PATH}" DIRECTORY)
    install(FILES "${_UNIFLOW_PUBLIC_HEADER}"
      DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/comms/uniflow/${_UNIFLOW_HEADER_DIRECTORY}"
      COMPONENT Development
    )
  endforeach()
  unset(_UNIFLOW_HEADER_DIRECTORY)
  unset(_UNIFLOW_HEADER_PATH)
  unset(_UNIFLOW_PUBLIC_HEADER)

  set(_UNIFLOW_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/uniflow")
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uniflowConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/uniflowConfig.cmake"
    INSTALL_DESTINATION "${_UNIFLOW_CMAKE_INSTALL_DIR}"
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/uniflowConfigVersion.cmake"
    VERSION "${UNIFLOW_VERSION}"
    COMPATIBILITY SameMajorVersion
  )
  install(EXPORT uniflowTargets
    FILE uniflowTargets.cmake
    NAMESPACE uniflow::
    DESTINATION "${_UNIFLOW_CMAKE_INSTALL_DIR}"
    COMPONENT Development
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/uniflowConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/uniflowConfigVersion.cmake"
    DESTINATION "${_UNIFLOW_CMAKE_INSTALL_DIR}"
    COMPONENT Development
  )
  unset(_UNIFLOW_CMAKE_INSTALL_DIR)
endif()
