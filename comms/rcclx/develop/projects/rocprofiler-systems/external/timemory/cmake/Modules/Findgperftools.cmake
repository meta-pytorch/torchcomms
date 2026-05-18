# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)
include(CMakeParseArguments)
include(TimemoryFindUtilities)

# ----------------------------------------------------------------------------------------#
# Useful config variables:
#
# gperftools_PREFER_STATIC gperftools_PERFER_SHARED gperftools_INTERFACE_LIBRARY
# gperftools_INTERFACE_COMPILE_DEFINITIONS
#
# ----------------------------------------------------------------------------------------#

set(gperftools_PREFER_SHARED
    ON
    CACHE BOOL "Prefer goerftools shared libraries")
mark_as_advanced(gperftools_PREFER_SHARED)

# ----------------------------------------------------------------------------------------#
# make invocable more than once
#
foreach(
    _VAR
    gperftools_INCLUDE_DIRS gperftools_LIBRARIES gperftools_STATIC_LIBRARIES
    gperftools_SHARED_LIBRARIES _gperftools_MISSING_LIBRARIES
    _gperftools_MISSING_COMPONENTS _gperftools_LIBRARY_BASE)
    unset(${_VAR})
endforeach()

# ----------------------------------------------------------------------------------------#

function(FIND_SHARED_LIBRARY)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .so .dylib .dll)
    find_library(${ARGN})
endfunction()

# ----------------------------------------------------------------------------------------#

if(CMAKE_VERSION VERSION_GREATER 2.8.7)
    set(gperftools_CHECK_COMPONENTS FALSE)
else(CMAKE_VERSION VERSION_GREATER 2.8.7)
    set(gperftools_CHECK_COMPONENTS TRUE)
endif()

# ----------------------------------------------------------------------------------------#

# Component options
set(gperftools_COMPONENT_OPTIONS
    profiler tcmalloc tcmalloc_and_profiler tcmalloc_debug tcmalloc_minimal
    tcmalloc_minimal_debug
    CACHE STRING "gperftools possible components")

# ----------------------------------------------------------------------------------------#

set(_gperftools_POSSIBLE_LIB_SUFFIXES lib lib64 lib32)

# ----------------------------------------------------------------------------------------#

find_path(
    gperftools_ROOT_DIR
    NAMES include/gperftools/tcmalloc.h include/gperftools/profiler.h
          include/google/tcmalloc.h include/google/profiler.h
    DOC "Google perftools root directory")

# ----------------------------------------------------------------------------------------#

find_path(
    gperftools_INCLUDE_DIR
    NAMES gperftools/tcmalloc.h gperftools/profiler.h google/tcmalloc.h google/profiler.h
    HINTS ${gperftools_ROOT_DIR}
    PATH_SUFFIXES include
    DOC "Google perftools profiler include directory")

# ----------------------------------------------------------------------------------------#

if(gperftools_INCLUDE_DIR)
    set(gperftools_INCLUDE_DIRS ${gperftools_INCLUDE_DIR})
endif()

# ----------------------------------------------------------------------------------------#
# Find components

foreach(_gperftools_COMPONENT ${gperftools_FIND_COMPONENTS})
    if(NOT "${_gperftools_COMPONENT}" IN_LIST gperftools_COMPONENT_OPTIONS)
        message(WARNING "${_gperftools_COMPONENT} is not listed as a real component")
    endif()

    string(TOLOWER ${_gperftools_COMPONENT} _gperftools_COMPONENT_LOWER)
    set(_gperftools_LIBRARY_BASE gperftools_${_gperftools_COMPONENT_LOWER}_LIBRARY)

    set(_gperftools_LIBRARY_NAME ${_gperftools_COMPONENT})

    find_shared_library(
        ${_gperftools_LIBRARY_BASE}_SHARED
        NAMES
        ${_gperftools_LIBRARY_NAME}
        HINTS
        ${gperftools_ROOT_DIR}
        PATH_SUFFIXES
        ${_gperftools_POSSIBLE_LIB_SUFFIXES}
        DOC
        "gperftools ${_gperftools_COMPONENT} library (shared)")

    timemory_find_static_library(
        ${_gperftools_LIBRARY_BASE}_STATIC
        NAMES ${_gperftools_LIBRARY_NAME}
        HINTS ${gperftools_ROOT_DIR}
        PATH_SUFFIXES ${_gperftools_POSSIBLE_LIB_SUFFIXES}
        DOC "gperftools ${_gperftools_COMPONENT} library (static)")

    # handle preference settings first
    if(gperftools_PREFER_SHARED)
        set(${_gperftools_LIBRARY_BASE} ${${_gperftools_LIBRARY_BASE}_SHARED})
    elseif(gperftools_PREFER_STATIC)
        set(${_gperftools_LIBRARY_BASE} ${${_gperftools_LIBRARY_BASE}_STATIC})
    endif()

    # set depending on try BUILD_{SHARED,STATIC}_LIBS
    if("${${_gperftools_LIBRARY_BASE}}" STREQUAL "")
        if(BUILD_SHARED_LIBS)
            set(${_gperftools_LIBRARY_BASE} ${${_gperftools_LIBRARY_BASE}_SHARED})
        elseif(BUILD_STATIC_LIBS)
            set(${_gperftools_LIBRARY_BASE} ${${_gperftools_LIBRARY_BASE}_STATIC})
        endif()
    endif()

    # if no preference and neither BUILD_SHARED_LIBS or BUILD_STATIC_LIBS, use first found
    if(NOT _gperftools_LIBRARY_BASE)
        if(${_gperftools_LIBRARY_BASE}_STATIC)
            set(${_gperftools_LIBRARY_BASE} ${${_gperftools_LIBRARY_BASE}_STATIC})
        elseif(${_gperftools_LIBRARY_BASE}_SHARED)
            set(${_gperftools_LIBRARY_BASE} ${${_gperftools_LIBRARY_BASE}_SHARED})
        endif()
    endif()

    mark_as_advanced(${_gperftools_LIBRARY_BASE}_SHARED
                     ${_gperftools_LIBRARY_BASE}_STATIC)

    set(gperftools_${_gperftools_COMPONENT_LOWER}_FOUND TRUE)

    if(NOT ${_gperftools_LIBRARY_BASE})
        # Component missing: record it for a later report
        list(APPEND _gperftools_MISSING_COMPONENTS ${_gperftools_COMPONENT})
        set(gperftools_${_gperftools_COMPONENT_LOWER}_FOUND FALSE)
    endif()

    set(gperftools_${_gperftools_COMPONENT}_FOUND
        ${gperftools_${_gperftools_COMPONENT_LOWER}_FOUND})

    if(${_gperftools_LIBRARY_BASE})
        # setup the gperftools_<COMPONENT>_LIBRARIES variable
        set(gperftools_${_gperftools_COMPONENT_LOWER}_LIBRARIES
            ${${_gperftools_LIBRARY_BASE}})
        list(APPEND gperftools_LIBRARIES ${${_gperftools_LIBRARY_BASE}})
    else(${_gperftools_LIBRARY_BASE})
        list(APPEND _gperftools_MISSING_LIBRARIES ${_gperftools_LIBRARY_BASE})
    endif()

    foreach(_TYPE STATIC SHARED)
        if(${_gperftools_LIBRARY_BASE}_${_TYPE})
            # setup the gperftools_<COMPONENT>_${_TYPE}_LIBRARIES variable
            set(gperftools_${_gperftools_COMPONENT_LOWER}_${_TYPE}_LIBRARIES
                ${${_gperftools_LIBRARY_BASE}_${_TYPE}})
            list(APPEND gperftools_${_TYPE}_LIBRARIES
                 ${${_gperftools_LIBRARY_BASE}_${_TYPE}})
        endif()
    endforeach()

endforeach()

# ----------------------------------------------------------------------------------------#
# handle missing components
#
if(DEFINED _gperftools_MISSING_COMPONENTS AND _gperftools_CHECK_COMPONENTS)
    if(NOT gperftools_FIND_QUIETLY)
        message(STATUS "One or more gperftools components were not found:")
        # Display missing components indented, each on a separate line
        foreach(_gperftools_MISSING_COMPONENT ${_gperftools_MISSING_COMPONENTS})
            message(STATUS "    ${_gperftools_MISSING_COMPONENT}")
            #
            # NOTE: let 'find_package_handle_standard_args' handle this
            #
            # if(gperftools_FIND_REQUIRED_${_gperftools_MISSING_COMPONENT})
            # message(FATAL_ERROR "Missing required component
            # ${_gperftools_MISSING_COMPONENT}") endif()
        endforeach()
    endif()
endif()

# ----------------------------------------------------------------------------------------#

mark_as_advanced(gperftools_INCLUDE_DIR gperftools_COMPONENTS gperftools_ROOT_DIR)

find_package_handle_standard_args(
    gperftools DEFAULT_MSG gperftools_ROOT_DIR gperftools_INCLUDE_DIRS
    ${_gperftools_MISSING_LIBRARIES})

# ----------------------------------------------------------------------------------------#
# generate an interface library if desired
#
if(gperftools_INTERFACE_LIBRARY)
    # add the target if it doesn't already exist
    if(NOT TARGET ${gperftools_INTERFACE_LIBRARY})
        add_library(${gperftools_INTERFACE_LIBRARY} INTERFACE)
    endif()
    # add the libraries is found
    if(gperftools_LIBRARIES)
        target_link_libraries(${gperftools_INTERFACE_LIBRARY}
                              INTERFACE ${gperftools_LIBRARIES})
    endif()
    # add the include directory
    target_include_directories(${gperftools_INTERFACE_LIBRARY} SYSTEM
                               INTERFACE ${gperftools_INCLUDE_DIRS})
    # if compile definitions
    if(gperftools_INTERFACE_COMPILE_DEFINITIONS)
        target_compile_definitions(${gperftools_INTERFACE_LIBRARY}
                                   INTERFACE ${gperftools_INTERFACE_COMPILE_DEFINITIONS})
    endif()
endif()
