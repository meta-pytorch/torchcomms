cmake_minimum_required(VERSION 3.5)
option(GOTCHA_ENABLE_TESTS "Enable internal tests" Off)
option(GOTCHA_ENABLE_WARNING_ERROR "Enable throw error for build warning" Off)
project(gotcha)
include(CheckCXXCompilerFlag)
include(cmake/gotcha.cmake)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

set(LIBTOOL_INTERFACE 2)
set(LIBTOOL_REVISION 4)
set(LIBTOOL_AGE 2)

set(GOTCHA_VERSION "(1, 0, 8)")


set(DEFAULT_SYMBOL_VISIBILITY hidden)

if(GOTCHA_ENABLE_TESTS)
  enable_testing()
  set(DEFAULT_SYMBOL_VISIBILITY default)
  add_definitions("-DGOTCHA_ENABLE_TESTING")
  if(CMAKE_BUILD_TYPE STREQUAL "PROFILE")
    set(CMAKE_C_FLAGS_PROFILE --coverage)
    set(CMAKE_CXX_FLAGS_PROFILE --coverage)
  endif()
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(src)
if(GOTCHA_ENABLE_TESTS)
add_subdirectory(test)
setup_coverage_target()
endif()

include(CMakePackageConfigHelpers)

# Configure gotcha-config-version.cmake
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gotcha-config-version.cmake"
  VERSION ${LIBTOOL_INTERFACE}.${LIBTOOL_REVISION}.${LIBTOOL_AGE}
  COMPATIBILITY SameMajorVersion)

set(gotcha_INSTALL_INCLUDE_DIR include/)
set(gotcha_INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR})

# Configure gotcha-config.cmake
configure_package_config_file(
  "cmake/config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gotcha-config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gotcha"
  PATH_VARS gotcha_INSTALL_INCLUDE_DIR PATH_VARS gotcha_INSTALL_LIBRARY_DIR)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/gotcha-config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/gotcha-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/gotcha")

# Suppress CMP0177 warning (CMake 3.28+)
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()

# Write the configure file
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gotcha_config.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/gotcha/gotcha_config.h" @ONLY)
install(
                FILES "${CMAKE_CURRENT_BINARY_DIR}/include/gotcha/gotcha_config.h"
                DESTINATION "${gotcha_INSTALL_INCLUDE_DIR}/gotcha")
