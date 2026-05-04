cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)

set(LIBTOOL_INTERFACE 3)
set(LIBTOOL_REVISION 0)
set(LIBTOOL_AGE 0)

set(GOTCHA_VERSION "1.1.0")

project(gotcha VERSION ${GOTCHA_VERSION})

cmake_policy(SET CMP0077 NEW) # option() honors normal variables
option(GOTCHA_ENABLE_TESTS "Enable internal tests" Off)
option(GOTCHA_ENABLE_WARNING_ERROR "Enable throw error for build warning" Off)

include(CheckCXXCompilerFlag)
include(cmake/gotcha.cmake)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

set(GOTCHA_VERSION "(${PROJECT_VERSION_MAJOR}, ${PROJECT_VERSION_MINOR}, ${PROJECT_VERSION_PATCH})")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(GOTCHA_ENABLE_TESTS)
    enable_testing()
    set(CMAKE_C_VISIBILITY_PRESET "default")
    set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
    add_compile_definitions("GOTCHA_ENABLE_TESTING")

    if(CMAKE_BUILD_TYPE STREQUAL "PROFILE")
        set(CMAKE_C_FLAGS_PROFILE --coverage)
        set(CMAKE_CXX_FLAGS_PROFILE --coverage)
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(src)

if(GOTCHA_ENABLE_TESTS)
    add_subdirectory(test)
    setup_coverage_target()
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Configure gotcha-config-version.cmake
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gotcha-config-version.cmake"
    VERSION ${LIBTOOL_INTERFACE}.${LIBTOOL_REVISION}.${LIBTOOL_AGE}
    COMPATIBILITY SameMajorVersion)

set(gotcha_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
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

# Write the configure file
configure_file("${PROJECT_SOURCE_DIR}/cmake/gotcha_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/gotcha/gotcha_config.h" @ONLY)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/gotcha/gotcha_config.h"
    DESTINATION "${gotcha_INSTALL_INCLUDE_DIR}/gotcha")
