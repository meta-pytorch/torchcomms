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

option(UNIFLOW_BUILD_SHARED_LIBS
  "Build UniFlow shared libraries" ${UNIFLOW_STANDALONE})
if(UNIFLOW_BUILD_SHARED_LIBS)
  set(UNIFLOW_LIB_TYPE SHARED)
else()
  set(UNIFLOW_LIB_TYPE STATIC)
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

function(_uniflow_resolve_feature_option name description default_value)
  if(DEFINED ${name})
    set(requested_value "${${name}}")
  else()
    set(requested_value AUTO)
  endif()
  string(TOUPPER "${requested_value}" requested_value)
  if(NOT requested_value MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
      "Invalid ${name}=${requested_value}; expected AUTO, ON, or OFF.")
  endif()

  set(${name} "${requested_value}" CACHE STRING
    "${description}: AUTO, ON, or OFF" FORCE)
  set_property(CACHE ${name} PROPERTY STRINGS AUTO ON OFF)

  if(requested_value STREQUAL AUTO)
    set(resolved_value "${default_value}")
  else()
    set(resolved_value "${requested_value}")
  endif()
  set(${name} "${resolved_value}" PARENT_SCOPE)
  message(STATUS "${name} = ${resolved_value} (${requested_value})")
endfunction()

if(UNIFLOW_GPU_PLATFORM STREQUAL NONE)
  set(_UNIFLOW_ENABLE_RDMA_DEFAULT OFF)
else()
  set(_UNIFLOW_ENABLE_RDMA_DEFAULT ON)
endif()
_uniflow_resolve_feature_option(
  UNIFLOW_ENABLE_RDMA "Build the RDMA transport"
  ${_UNIFLOW_ENABLE_RDMA_DEFAULT})
unset(_UNIFLOW_ENABLE_RDMA_DEFAULT)

if(UNIFLOW_GPU_PLATFORM STREQUAL CUDA)
  set(_UNIFLOW_ENABLE_NVLINK_DEFAULT ON)
else()
  set(_UNIFLOW_ENABLE_NVLINK_DEFAULT OFF)
endif()
_uniflow_resolve_feature_option(
  UNIFLOW_ENABLE_NVLINK "Build the NVLink transport"
  ${_UNIFLOW_ENABLE_NVLINK_DEFAULT})
unset(_UNIFLOW_ENABLE_NVLINK_DEFAULT)

if(UNIFLOW_GPU_PLATFORM STREQUAL HIP)
  set(_UNIFLOW_ENABLE_TCP_DEFAULT ON)
else()
  set(_UNIFLOW_ENABLE_TCP_DEFAULT OFF)
endif()
_uniflow_resolve_feature_option(
  UNIFLOW_ENABLE_TCP "Build the TCP transport"
  ${_UNIFLOW_ENABLE_TCP_DEFAULT})
unset(_UNIFLOW_ENABLE_TCP_DEFAULT)
option(UNIFLOW_USE_SYSTEM_SPDLOG "Use system-installed spdlog and fmt" ON)
if(UNIFLOW_STANDALONE AND
   NOT UNIFLOW_BUILD_PYTHON AND
   NOT UNIFLOW_SPDLOG_INCLUDE_DIR AND
   NOT UNIFLOW_FMT_INCLUDE_DIR)
  set(_UNIFLOW_INSTALL_CPP_DEFAULT ON)
else()
  set(_UNIFLOW_INSTALL_CPP_DEFAULT OFF)
endif()
option(UNIFLOW_INSTALL_CPP
  "Install the UniFlow C++ SDK" ${_UNIFLOW_INSTALL_CPP_DEFAULT})
unset(_UNIFLOW_INSTALL_CPP_DEFAULT)
option(UNIFLOW_ALLOW_DEPENDENCY_OVERWRITE
  "Allow fetched fmt and spdlog to overwrite files in the install prefix" OFF)

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

set(UNIFLOW_FMT_VERSION 11.1.4)
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
if(UNIFLOW_INSTALL_CPP AND UNIFLOW_SPDLOG_INCLUDE_DIR)
  message(FATAL_ERROR
      "UNIFLOW_INSTALL_CPP=ON cannot be used with the spdlog and fmt "
      "header-tree options because those dependencies are not installed "
      "with the C++ SDK. Set UNIFLOW_INSTALL_CPP=OFF for a Python-only "
      "build, or unset UNIFLOW_SPDLOG_INCLUDE_DIR and "
      "UNIFLOW_FMT_INCLUDE_DIR to use packaged dependencies.")
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
  set(_UNIFLOW_DEFAULT_SPDLOG_FMT_EXTERNAL ON)
  if(SPDLOG_USE_STD_FORMAT)
    set(_UNIFLOW_DEFAULT_SPDLOG_FMT_EXTERNAL OFF)
  endif()
  set(_UNIFLOW_DEPENDENCY_OPTION_NAMES
    FMT_DOC
    FMT_INSTALL
    FMT_TEST
    SPDLOG_BUILD_PIC
    SPDLOG_FMT_EXTERNAL
    SPDLOG_INSTALL)
  set(_UNIFLOW_DEPENDENCY_OPTION_DEFAULTS
    OFF
    ${UNIFLOW_INSTALL_CPP}
    OFF
    ON
    ${_UNIFLOW_DEFAULT_SPDLOG_FMT_EXTERNAL}
    ${UNIFLOW_INSTALL_CPP})
  foreach(_UNIFLOW_DEPENDENCY_OPTION _UNIFLOW_DEPENDENCY_OPTION_DEFAULT
      IN ZIP_LISTS
        _UNIFLOW_DEPENDENCY_OPTION_NAMES
        _UNIFLOW_DEPENDENCY_OPTION_DEFAULTS)
    if(NOT DEFINED ${_UNIFLOW_DEPENDENCY_OPTION})
      set(${_UNIFLOW_DEPENDENCY_OPTION}
        ${_UNIFLOW_DEPENDENCY_OPTION_DEFAULT})
    endif()
  endforeach()
  unset(_UNIFLOW_DEPENDENCY_OPTION_DEFAULT)
  unset(_UNIFLOW_DEPENDENCY_OPTION)
  unset(_UNIFLOW_DEPENDENCY_OPTION_DEFAULTS)
  unset(_UNIFLOW_DEPENDENCY_OPTION_NAMES)
  unset(_UNIFLOW_DEFAULT_SPDLOG_FMT_EXTERNAL)

  if(UNIFLOW_INSTALL_CPP AND NOT FMT_INSTALL)
    message(FATAL_ERROR
      "UNIFLOW_INSTALL_CPP=ON with fetched dependencies requires "
      "FMT_INSTALL=ON so the installed SDK can find fmt.")
  endif()
  if(UNIFLOW_INSTALL_CPP AND NOT SPDLOG_INSTALL)
    message(FATAL_ERROR
      "UNIFLOW_INSTALL_CPP=ON with fetched dependencies requires "
      "SPDLOG_INSTALL=ON so the installed SDK can find spdlog.")
  endif()

  if(SPDLOG_FMT_EXTERNAL AND SPDLOG_USE_STD_FORMAT)
    message(FATAL_ERROR
      "Fetched spdlog cannot use external fmt and std::format together. "
      "Enable only one of SPDLOG_FMT_EXTERNAL or SPDLOG_USE_STD_FORMAT.")
  elseif(NOT SPDLOG_FMT_EXTERNAL AND NOT SPDLOG_USE_STD_FORMAT)
    message(FATAL_ERROR
      "UniFlow requires fetched spdlog to use external fmt or std::format. "
      "Set SPDLOG_FMT_EXTERNAL=ON or SPDLOG_USE_STD_FORMAT=ON.")
  endif()

  if(SPDLOG_USE_STD_FORMAT)
    set(UNIFLOW_SPDLOG_FORMAT_MODE STD_FORMAT)
  else()
    set(UNIFLOW_SPDLOG_FORMAT_MODE EXTERNAL_FMT)
  endif()

  if(UNIFLOW_INSTALL_CPP)
    install(SCRIPT
      "${CMAKE_CURRENT_BINARY_DIR}/CheckDependencyInstall.cmake")
  endif()

  function(_uniflow_fetch_fmt_and_spdlog)
    # Keep fetched dependencies in the same library form as UniFlow. In
    # particular, a shared UniFlow must not embed one static spdlog registry
    # while its consumers link another.
    set(BUILD_SHARED_LIBS ${UNIFLOW_BUILD_SHARED_LIBS})
    if(NOT TARGET fmt::fmt)
      FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG "${UNIFLOW_FMT_VERSION}"
      )
      FetchContent_MakeAvailable(fmt)
      set(UNIFLOW_FETCHED_FMT ON PARENT_SCOPE)
    endif()
    FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG "v${UNIFLOW_MINIMUM_SPDLOG_VERSION}"
    )
    FetchContent_MakeAvailable(spdlog)
    set(UNIFLOW_FETCHED_SPDLOG ON PARENT_SCOPE)

    if(fmt_SOURCE_DIR)
      get_directory_property(_uniflow_fmt_version
        DIRECTORY "${fmt_SOURCE_DIR}" DEFINITION FMT_VERSION)
      set(fmt_SOURCE_DIR "${fmt_SOURCE_DIR}" PARENT_SCOPE)
    else()
      set(_uniflow_fmt_version "${fmt_VERSION}")
    endif()
    get_directory_property(_uniflow_spdlog_version
      DIRECTORY "${spdlog_SOURCE_DIR}" DEFINITION SPDLOG_VERSION)
    set(spdlog_SOURCE_DIR "${spdlog_SOURCE_DIR}" PARENT_SCOPE)
    set(UNIFLOW_FMT_PACKAGE_VERSION
      "${_uniflow_fmt_version}" PARENT_SCOPE)
    set(UNIFLOW_SPDLOG_PACKAGE_VERSION
      "${_uniflow_spdlog_version}" PARENT_SCOPE)
  endfunction()
  _uniflow_fetch_fmt_and_spdlog()

  if(UNIFLOW_INSTALL_CPP)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckDependencyInstall.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/CheckDependencyInstall.cmake"
      @ONLY)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uniflowBundledDependencies.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/uniflowBundledDependencies.cmake"
      @ONLY)
  endif()

  set(UNIFLOW_SPDLOG_TARGET spdlog::spdlog)
  set(UNIFLOW_FMT_TARGET fmt::fmt)
  if(UNIFLOW_INSTALL_CPP)
    if(UNIFLOW_FETCHED_FMT)
      install(FILES "${fmt_SOURCE_DIR}/LICENSE"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/fmt")
    endif()
    install(FILES "${spdlog_SOURCE_DIR}/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/spdlog")
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/uniflowBundledDependencies.cmake"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/uniflow")
  endif()
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
    set(_UNIFLOW_SAVED_REQUIRED_DEFINITIONS
        "${CMAKE_REQUIRED_DEFINITIONS}")
    set(CMAKE_REQUIRED_LIBRARIES spdlog::spdlog)

    set(CMAKE_REQUIRED_DEFINITIONS
      -DUNIFLOW_EXPECT_SPDLOG_EXTERNAL_FMT=1)
    unset(_UNIFLOW_SPDLOG_USES_EXTERNAL_FMT CACHE)
    check_cxx_source_compiles(
      "${_UNIFLOW_SPDLOG_COMPATIBILITY_SOURCE}"
      _UNIFLOW_SPDLOG_USES_EXTERNAL_FMT)
    set(_UNIFLOW_SPDLOG_USES_EXTERNAL_FMT_RESULT
        "${_UNIFLOW_SPDLOG_USES_EXTERNAL_FMT}")
    unset(_UNIFLOW_SPDLOG_USES_EXTERNAL_FMT CACHE)

    set(CMAKE_REQUIRED_DEFINITIONS
      -DUNIFLOW_EXPECT_SPDLOG_STD_FORMAT=1)
    unset(_UNIFLOW_SPDLOG_USES_STD_FORMAT CACHE)
    check_cxx_source_compiles(
      "${_UNIFLOW_SPDLOG_COMPATIBILITY_SOURCE}"
      _UNIFLOW_SPDLOG_USES_STD_FORMAT)
    set(_UNIFLOW_SPDLOG_USES_STD_FORMAT_RESULT
        "${_UNIFLOW_SPDLOG_USES_STD_FORMAT}")
    unset(_UNIFLOW_SPDLOG_USES_STD_FORMAT CACHE)

    set(CMAKE_REQUIRED_LIBRARIES
        "${_UNIFLOW_SAVED_REQUIRED_LIBRARIES}")
    set(CMAKE_REQUIRED_DEFINITIONS
        "${_UNIFLOW_SAVED_REQUIRED_DEFINITIONS}")
    unset(_UNIFLOW_SAVED_REQUIRED_LIBRARIES)
    unset(_UNIFLOW_SAVED_REQUIRED_DEFINITIONS)
    unset(_UNIFLOW_SPDLOG_COMPATIBILITY_SOURCE)

    if(_UNIFLOW_SPDLOG_USES_EXTERNAL_FMT_RESULT)
      set(UNIFLOW_SPDLOG_FORMAT_MODE EXTERNAL_FMT)
    elseif(_UNIFLOW_SPDLOG_USES_STD_FORMAT_RESULT)
      set(UNIFLOW_SPDLOG_FORMAT_MODE STD_FORMAT)
    else()
      message(FATAL_ERROR
          "The selected spdlog target is not compatible with UniFlow. UniFlow "
          "requires spdlog ${UNIFLOW_MINIMUM_SPDLOG_VERSION} or newer, built "
          "with SPDLOG_FMT_EXTERNAL=ON or SPDLOG_USE_STD_FORMAT=ON.")
    endif()
    unset(_UNIFLOW_SPDLOG_USES_EXTERNAL_FMT_RESULT)
    unset(_UNIFLOW_SPDLOG_USES_STD_FORMAT_RESULT)
  else()
    # try_compile cannot consume normal build targets. Compile the same check
    # as part of the build so parent-provided wrappers retain their complete
    # transitive usage requirements.
    add_library(uniflow_spdlog_compatibility_check OBJECT
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpdlogCompatibilityCheck.cpp")
    target_link_libraries(uniflow_spdlog_compatibility_check PRIVATE
      spdlog::spdlog)
    if(UNIFLOW_FETCHED_SPDLOG)
      target_compile_definitions(uniflow_spdlog_compatibility_check PRIVATE
        "UNIFLOW_EXPECT_SPDLOG_${UNIFLOW_SPDLOG_FORMAT_MODE}=1")
    elseif(UNIFLOW_INSTALL_CPP)
      message(FATAL_ERROR
        "Installing the C++ SDK with a parent-provided spdlog build target "
        "is not supported because that target cannot be included in "
        "UniFlow's CMake export. Use an installed spdlog package, allow "
        "UniFlow to fetch spdlog, or set UNIFLOW_INSTALL_CPP=OFF.")
    endif()
  endif()
  unset(_UNIFLOW_SPDLOG_IMPORTED)
  unset(_UNIFLOW_SPDLOG_REAL_TARGET)
  unset(_UNIFLOW_SPDLOG_ALIASED_TARGET)

  if(NOT UNIFLOW_FMT_PACKAGE_VERSION)
    set(UNIFLOW_FMT_PACKAGE_VERSION "${fmt_VERSION}")
  endif()
  if(NOT UNIFLOW_SPDLOG_PACKAGE_VERSION)
    set(UNIFLOW_SPDLOG_PACKAGE_VERSION "${spdlog_VERSION}")
  endif()

  set(UNIFLOW_SPDLOG_TARGET spdlog::spdlog)
  set(UNIFLOW_FMT_TARGET fmt::fmt)
endif()

if(UNIFLOW_INSTALL_CPP AND
   (NOT UNIFLOW_FMT_PACKAGE_VERSION OR
    NOT UNIFLOW_SPDLOG_PACKAGE_VERSION OR
    NOT UNIFLOW_SPDLOG_FORMAT_MODE))
  message(FATAL_ERROR
    "The C++ SDK requires versioned fmt and spdlog packages with a known "
    "formatting mode.")
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
  $<$<STREQUAL:${UNIFLOW_GPU_PLATFORM},HIP>:UNIFLOW_ENABLE_P2P_TRANSPORT=1>
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

if(TARGET uniflow_spdlog_compatibility_check)
  add_dependencies(uniflow uniflow_spdlog_compatibility_check)
endif()

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
  set_target_properties(uniflow PROPERTIES INSTALL_RPATH "$ORIGIN")
  if(UNIFLOW_FETCHED_SPDLOG)
    set_target_properties(spdlog PROPERTIES INSTALL_RPATH "$ORIGIN")
  endif()
  add_subdirectory(python)
endif()

if(UNIFLOW_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests)
endif()

if(UNIFLOW_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(UNIFLOW_INSTALL_CPP)
  include(CMakePackageConfigHelpers)

  install(TARGETS uniflow
    EXPORT uniflowTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  set(UNIFLOW_INSTALLED_PUBLIC_HEADERS)
  foreach(_UNIFLOW_PUBLIC_HEADER IN LISTS UNIFLOW_PUBLIC_HEADERS)
    file(RELATIVE_PATH _UNIFLOW_HEADER_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}" "${_UNIFLOW_PUBLIC_HEADER}")
    list(APPEND UNIFLOW_INSTALLED_PUBLIC_HEADERS
      "${_UNIFLOW_HEADER_PATH}")
    get_filename_component(_UNIFLOW_HEADER_DIRECTORY
      "${_UNIFLOW_HEADER_PATH}" DIRECTORY)
    install(FILES "${_UNIFLOW_PUBLIC_HEADER}"
      DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/comms/uniflow/${_UNIFLOW_HEADER_DIRECTORY}"
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
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/uniflowConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/uniflowConfigVersion.cmake"
    DESTINATION "${_UNIFLOW_CMAKE_INSTALL_DIR}"
  )

  if(UNIFLOW_BUILD_TESTS)
    add_test(NAME uniflow_installed_consumer
      COMMAND ${CMAKE_COMMAND}
        "-DUNIFLOW_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DUNIFLOW_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/consumer"
        "-DUNIFLOW_CONSUMER_GENERATOR=${CMAKE_GENERATOR}"
        "-DUNIFLOW_CONSUMER_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DUNIFLOW_CONSUMER_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DUNIFLOW_CONSUMER_CXX_COMPILER_TARGET=${CMAKE_CXX_COMPILER_TARGET}"
        "-DUNIFLOW_CONSUMER_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}"
        "-DUNIFLOW_CONSUMER_SYSROOT=${CMAKE_SYSROOT}"
        "-DUNIFLOW_CONSUMER_SYSROOT_COMPILE=${CMAKE_SYSROOT_COMPILE}"
        "-DUNIFLOW_CONSUMER_SYSROOT_LINK=${CMAKE_SYSROOT_LINK}"
        "-DUNIFLOW_CONSUMER_CROSSCOMPILING=${CMAKE_CROSSCOMPILING}"
        "-DUNIFLOW_CONSUMER_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
        "-DUNIFLOW_CONSUMER_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}"
        "-DUNIFLOW_CONSUMER_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DUNIFLOW_CONSUMER_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DUNIFLOW_CONSUMER_CONFIG=$<CONFIG>"
        "-DUNIFLOW_CONSUMER_CUDA_ROOT=${CUDAToolkit_ROOT}"
        "-DUNIFLOW_CONSUMER_FMT_DIR=${fmt_DIR}"
        "-DUNIFLOW_CONSUMER_HIP_DIR=${hip_DIR}"
        "-DUNIFLOW_CONSUMER_SPDLOG_DIR=${spdlog_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/RunInstalledConsumer.cmake")
    set_tests_properties(uniflow_installed_consumer PROPERTIES LABELS unit)
  endif()
  unset(_UNIFLOW_CMAKE_INSTALL_DIR)
endif()

if(NOT UNIFLOW_GPU_PLATFORM STREQUAL NONE)
  uniflow_finalize_hipify()
endif()
