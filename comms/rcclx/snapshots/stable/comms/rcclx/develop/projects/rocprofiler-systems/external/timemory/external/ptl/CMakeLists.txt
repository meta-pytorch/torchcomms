################################################################################
#   cmake settings
#
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

# Add allocation export symbol for the PLT module
add_compile_definitions(PTL_ALLOC_EXPORT)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)

# Check if project is being used directly or via add_subdirectory
set(PTL_MASTER_PROJECT ON)
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(PTL_MASTER_PROJECT OFF)
endif()

################################################################################
#   version
#
file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" VERSION_STRING "${VERSION_STRING}")
string(REGEX REPLACE "[A-Za-z].*" "" VERSION_STRING "${VERSION_STRING}")
set(PTL_VERSION "${VERSION_STRING}")


################################################################################
#   project
#
project(PTL LANGUAGES C CXX VERSION ${PTL_VERSION})
if(NOT PTL_MASTER_PROJECT)
    unset(${PROJECT_NAME}_C_FLAGS CACHE)
    unset(${PROJECT_NAME}_CXX_FLAGS CACHE)
endif()
set(CMAKE_DIRECTORY_LABELS "PTL")

################################################################################
#
#
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

foreach(_TYPE MAJOR MINOR PATCH)
    set(PTL_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}}
        CACHE STRING "PTL ${_TYPE} version" FORCE)
    mark_as_advanced(PTL_VERSION_${_TYPE})
endforeach()
set(PTL_VERSION_STRING "${PTL_VERSION_MAJOR}.${PTL_VERSION_MINOR}.${PTL_VERSION_PATCH}"
    CACHE STRING "PTL version string")
set(PTL_VERSION_STRING "${PTL_VERSION_MAJOR}.${PTL_VERSION_MINOR}.${PTL_VERSION_PATCH}")
math(EXPR PTL_VERSION_CODE
    "10000 * ${PTL_VERSION_MAJOR} + 100 * ${PTL_VERSION_MINOR} + ${PTL_VERSION_PATCH}")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)
mark_as_advanced(PTL_VERSION)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type" FORCE)
endif()

################################################################################

# include(MacroUtilities)
include(Options)
include(Compilers)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)
include(ClangFormat)

################################################################################
#   PTL source
################################################################################

add_subdirectory(source)

################################################################################
#   Installation and info
################################################################################

include(ConfigurePackage)

if(PTL_BUILD_EXAMPLES)
    if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/examples)
        set(PTL_DIR ${CMAKE_BINARY_DIR})
        add_subdirectory(examples)
    else()
        set(PTL_BUILD_EXAMPLES OFF CACHE BOOL
	    "PTL examples directory doesn't exist" FORCE)
    endif()
endif()

if(PTL_MASTER_PROJECT)
    ptl_print_features()
else()
    set(${PROJECT_NAME}_C_FLAGS ${${PROJECT_NAME}_C_FLAGS} CACHE STRING
        "C compiler flags for ${PROJECT_NAME}")
    set(${PROJECT_NAME}_CXX_FLAGS ${${PROJECT_NAME}_CXX_FLAGS} CACHE STRING
        "CXX compiler flags for ${PROJECT_NAME}")
endif()

