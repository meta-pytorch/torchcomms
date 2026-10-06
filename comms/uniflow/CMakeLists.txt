# Copyright (c) Meta Platforms, Inc. and affiliates.

cmake_minimum_required(VERSION 3.20)

# Standalone project support (for OSS builds)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(uniflow LANGUAGES CXX)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  endif()
  set(UNIFLOW_STANDALONE ON)
else()
  set(UNIFLOW_STANDALONE OFF)
endif()

option(UNIFLOW_BUILD_SHARED_LIBS
  "Build UniFlow shared libraries" ${UNIFLOW_STANDALONE})
if(UNIFLOW_BUILD_SHARED_LIBS)
  set(UNIFLOW_LIB_TYPE SHARED)
else()
  set(UNIFLOW_LIB_TYPE STATIC)
endif()

find_package(Threads REQUIRED)

set(UNIFLOW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Where translated AMD sources land; consumed by uniflow_hipify below. Set before
# including the module so nested directories all agree on one location.
set(UNIFLOW_HIPIFY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipified)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

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

option(UNIFLOW_USE_SYSTEM_SPDLOG "Use system-installed spdlog and fmt" ON)
set(UNIFLOW_MINIMUM_SPDLOG_VERSION 1.15.2)

# Fetched fmt and spdlog libraries can be linked into the shared UniFlow
# library, so they must be compiled as position-independent code too.
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

set(UNIFLOW_FETCHED_FMT OFF)
set(UNIFLOW_FETCHED_SPDLOG OFF)

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
elseif(UNIFLOW_USE_SYSTEM_SPDLOG AND NOT TARGET spdlog::spdlog)
  find_package(spdlog ${UNIFLOW_MINIMUM_SPDLOG_VERSION} QUIET)
endif()

if(UNIFLOW_SPDLOG_INCLUDE_DIR)
  message(STATUS "spdlog and fmt taken from header trees")
elseif(NOT TARGET spdlog::spdlog)
  if(NOT UNIFLOW_USE_SYSTEM_SPDLOG)
    message(STATUS
      "UNIFLOW_USE_SYSTEM_SPDLOG=OFF; fetching fmt and spdlog")
  elseif(spdlog_CONSIDERED_VERSIONS)
    list(JOIN spdlog_CONSIDERED_VERSIONS ", " _UNIFLOW_SPDLOG_VERSIONS)
    message(STATUS
      "Found system spdlog version(s) ${_UNIFLOW_SPDLOG_VERSIONS}, but "
      "UniFlow requires ${UNIFLOW_MINIMUM_SPDLOG_VERSION} or newer; "
      "fetching fmt and spdlog")
    unset(_UNIFLOW_SPDLOG_VERSIONS)
  elseif(spdlog_CONSIDERED_CONFIGS)
    message(STATUS
      "Found a system spdlog package that does not report its version, but "
      "UniFlow requires ${UNIFLOW_MINIMUM_SPDLOG_VERSION} or newer; "
      "fetching fmt and spdlog")
  else()
    message(STATUS
      "No system spdlog CMake package was found; fetching fmt and spdlog")
  endif()
  include(FetchContent)
  function(_uniflow_fetch_fmt_and_spdlog)
    # Keep fetched dependencies in the same library form as UniFlow. In
    # particular, a shared UniFlow must not embed one static spdlog registry
    # while its consumers link another.
    set(BUILD_SHARED_LIBS ${UNIFLOW_BUILD_SHARED_LIBS})
    if(NOT TARGET fmt::fmt)
      FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 11.1.4
      )
      FetchContent_MakeAvailable(fmt)
      set(UNIFLOW_FETCHED_FMT ON PARENT_SCOPE)
    endif()

    set(SPDLOG_BUILD_PIC ON)
    if(SPDLOG_USE_STD_FORMAT)
      set(SPDLOG_FMT_EXTERNAL OFF)
    else()
      set(SPDLOG_FMT_EXTERNAL ON)
    endif()
    FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG "v${UNIFLOW_MINIMUM_SPDLOG_VERSION}"
    )
    FetchContent_MakeAvailable(spdlog)
    set(UNIFLOW_FETCHED_SPDLOG ON PARENT_SCOPE)

    set(fmt_SOURCE_DIR "${fmt_SOURCE_DIR}" PARENT_SCOPE)
    set(spdlog_SOURCE_DIR "${spdlog_SOURCE_DIR}" PARENT_SCOPE)
  endfunction()
  _uniflow_fetch_fmt_and_spdlog()
else()
  message(STATUS "Using existing spdlog CMake target")
endif()

if(NOT UNIFLOW_SPDLOG_INCLUDE_DIR)
  # TcpController.cpp includes fmt directly even when spdlog uses std::format.
  if(NOT TARGET fmt::fmt)
    find_package(fmt REQUIRED)
  endif()

  get_target_property(_UNIFLOW_SPDLOG_ALIASED_TARGET
      spdlog::spdlog ALIASED_TARGET)
  if(_UNIFLOW_SPDLOG_ALIASED_TARGET)
    set(_UNIFLOW_SPDLOG_REAL_TARGET
        "${_UNIFLOW_SPDLOG_ALIASED_TARGET}")
  else()
    set(_UNIFLOW_SPDLOG_REAL_TARGET spdlog::spdlog)
  endif()
  get_target_property(_UNIFLOW_SPDLOG_IMPORTED
      "${_UNIFLOW_SPDLOG_REAL_TARGET}" IMPORTED)

  if(_UNIFLOW_SPDLOG_IMPORTED)
    # Imported targets can be passed to try_compile. This lets CMake apply the
    # target's complete transitive usage requirements to the probe.
    include(CheckCXXSourceCompiles)
    file(READ
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpdlogCompatibilityCheck.cpp"
      _UNIFLOW_SPDLOG_COMPATIBILITY_SOURCE)
    set(_UNIFLOW_SAVED_REQUIRED_LIBRARIES
        "${CMAKE_REQUIRED_LIBRARIES}")
    set(CMAKE_REQUIRED_LIBRARIES spdlog::spdlog)
    unset(_UNIFLOW_SPDLOG_COMPATIBLE CACHE)
    check_cxx_source_compiles(
      "${_UNIFLOW_SPDLOG_COMPATIBILITY_SOURCE}"
      _UNIFLOW_SPDLOG_COMPATIBLE)
    set(CMAKE_REQUIRED_LIBRARIES
        "${_UNIFLOW_SAVED_REQUIRED_LIBRARIES}")
    unset(_UNIFLOW_SAVED_REQUIRED_LIBRARIES)
    unset(_UNIFLOW_SPDLOG_COMPATIBILITY_SOURCE)
    if(NOT _UNIFLOW_SPDLOG_COMPATIBLE)
      message(FATAL_ERROR
          "The selected spdlog target is not compatible with UniFlow. UniFlow "
          "requires spdlog ${UNIFLOW_MINIMUM_SPDLOG_VERSION} or newer, built "
          "with SPDLOG_FMT_EXTERNAL=ON or SPDLOG_USE_STD_FORMAT=ON.")
    endif()
    unset(_UNIFLOW_SPDLOG_COMPATIBLE CACHE)
  else()
    # try_compile cannot consume normal build targets. Compile the same check
    # as part of the build so parent-provided wrappers retain their complete
    # transitive usage requirements.
    add_library(uniflow_spdlog_compatibility_check OBJECT
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpdlogCompatibilityCheck.cpp")
    target_link_libraries(uniflow_spdlog_compatibility_check PRIVATE
      spdlog::spdlog)
  endif()
  unset(_UNIFLOW_SPDLOG_IMPORTED)
  unset(_UNIFLOW_SPDLOG_REAL_TARGET)
  unset(_UNIFLOW_SPDLOG_ALIASED_TARGET)

  set(UNIFLOW_SPDLOG_TARGET spdlog::spdlog)
  set(UNIFLOW_FMT_TARGET fmt::fmt)
endif()

add_library(uniflow_build_config INTERFACE)
target_compile_features(uniflow_build_config INTERFACE cxx_std_20)
target_include_directories(uniflow_build_config INTERFACE
  $<BUILD_INTERFACE:${UNIFLOW_BUILD_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(uniflow_build_config INTERFACE
  Threads::Threads
  ${UNIFLOW_FMT_TARGET}
  ${UNIFLOW_SPDLOG_TARGET}
)

include(UniflowGpuPlatform)

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

if(TARGET uniflow_spdlog_compatibility_check)
  add_dependencies(uniflow uniflow_spdlog_compatibility_check)
endif()

target_compile_features(uniflow PUBLIC cxx_std_20)

if(UNIFLOW_ENABLE_TCP_TRANSPORT)
  target_sources(uniflow PRIVATE $<TARGET_OBJECTS:uniflow_tcp_transport>)
  target_compile_definitions(
      uniflow PUBLIC UNIFLOW_ENABLE_TCP_TRANSPORT=1)
endif()

target_include_directories(uniflow PUBLIC
    $<BUILD_INTERFACE:${UNIFLOW_BUILD_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(uniflow PUBLIC
    Threads::Threads
    ${UNIFLOW_SPDLOG_TARGET}
    uniflow_build_config
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
  set_target_properties(uniflow PROPERTIES INSTALL_RPATH "$ORIGIN")
  if(UNIFLOW_FETCHED_SPDLOG)
    set_target_properties(spdlog PROPERTIES INSTALL_RPATH "$ORIGIN")
  endif()
  add_subdirectory(python)
endif()

option(UNIFLOW_BUILD_TESTS "Build UniFlow tests" OFF)
if(UNIFLOW_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
  add_subdirectory(executor/tests)
  add_subdirectory(controller/tests)
  add_subdirectory(core/tests)
  add_subdirectory(logging/tests)
endif()
