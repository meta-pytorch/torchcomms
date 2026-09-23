# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

#------------------------------------------------------------------------------#

IF (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (GPerfTools_CHECK_COMPONENTS FALSE)
ELSE (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (GPerfTools_CHECK_COMPONENTS TRUE)
ENDIF (CMAKE_VERSION VERSION_GREATER 2.8.7)

#------------------------------------------------------------------------------#

# Component options
set(_GPerfTools_COMPONENT_OPTIONS
    profiler
    tcmalloc
    tcmalloc_and_profiler
    tcmalloc_debug
    tcmalloc_minimal
    tcmalloc_minimal_debug
)

#------------------------------------------------------------------------------#

IF("${GPerfTools_FIND_COMPONENTS}" STREQUAL "")
    IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        LIST(APPEND GPerfTools_FIND_COMPONENTS profiler tcmalloc_debug)
    ELSE()
        LIST(APPEND GPerfTools_FIND_COMPONENTS tcmalloc_and_profiler)
    ENDIF()
ENDIF("${GPerfTools_FIND_COMPONENTS}" STREQUAL "")

#------------------------------------------------------------------------------#

set(_GPerfTools_POSSIBLE_LIB_SUFFIXES lib lib64 lib32)

#------------------------------------------------------------------------------#

find_path(GPerfTools_ROOT_DIR
    NAMES
        include/gperftools/tcmalloc.h
        include/google/tcmalloc.h
        include/gperftools/profiler.h
        include/google/profiler.h
    DOC "Google perftools root directory")

#------------------------------------------------------------------------------#

find_path(GPerfTools_INCLUDE_DIR
    NAMES
        gperftools/tcmalloc.h
        google/tcmalloc.h
        gperftools/profiler.h
        google/profiler.h
    HINTS
        ${GPerfTools_ROOT_DIR}
    PATH_SUFFIXES
        include
    DOC "Google perftools profiler include directory")

#------------------------------------------------------------------------------#

set(GPerfTools_INCLUDE_DIRS ${GPerfTools_INCLUDE_DIR})

#------------------------------------------------------------------------------#
# Find components

FOREACH (_GPerfTools_COMPONENT ${GPerfTools_FIND_COMPONENTS})
    IF(NOT "${_GPerfTools_COMPONENT_OPTIONS}" MATCHES "${_GPerfTools_COMPONENT}")
        MESSAGE(WARNING "${_GPerfTools_COMPONENT} is not listed as a real component")
    ENDIF()

    STRING (TOUPPER ${_GPerfTools_COMPONENT} _GPerfTools_COMPONENT_UPPER)
    SET (_GPerfTools_LIBRARY_BASE GPerfTools_${_GPerfTools_COMPONENT_UPPER}_LIBRARY)

    SET (_GPerfTools_LIBRARY_NAME ${_GPerfTools_COMPONENT})

    FIND_LIBRARY (${_GPerfTools_LIBRARY_BASE}
        NAMES ${_GPerfTools_LIBRARY_NAME}
        HINTS ${GPerfTools_ROOT_DIR}
        PATH_SUFFIXES ${_GPerfTools_POSSIBLE_LIB_SUFFIXES}
        DOC "MKL ${_GPerfTools_COMPONENT} library")

    MARK_AS_ADVANCED (${_GPerfTools_LIBRARY_BASE})

    SET (GPerfTools_${_GPerfTools_COMPONENT_UPPER}_FOUND TRUE)

    IF (NOT ${_GPerfTools_LIBRARY_BASE})
        # Component missing: record it for a later report
        LIST (APPEND _GPerfTools_MISSING_COMPONENTS ${_GPerfTools_COMPONENT})
        SET (GPerfTools_${_GPerfTools_COMPONENT_UPPER}_FOUND FALSE)
    ENDIF (NOT ${_GPerfTools_LIBRARY_BASE})

    SET (GPerfTools_${_GPerfTools_COMPONENT}_FOUND
        ${GPerfTools_${_GPerfTools_COMPONENT_UPPER}_FOUND})

    IF (${_GPerfTools_LIBRARY_BASE})
        # setup the GPerfTools_<COMPONENT>_LIBRARIES variable
        SET (GPerfTools_${_GPerfTools_COMPONENT_UPPER}_LIBRARIES
            ${${_GPerfTools_LIBRARY_BASE}})
        LIST (APPEND GPerfTools_LIBRARIES ${${_GPerfTools_LIBRARY_BASE}})
    ELSE (${_GPerfTools_LIBRARY_BASE})
        LIST (APPEND _GPerfTools_MISSING_LIBRARIES ${_GPerfTools_LIBRARY_BASE})
    ENDIF (${_GPerfTools_LIBRARY_BASE})

ENDFOREACH (_GPerfTools_COMPONENT ${GPerfTools_FIND_COMPONENTS})


#----- Missing components
IF (DEFINED _GPerfTools_MISSING_COMPONENTS AND _GPerfTools_CHECK_COMPONENTS)
    IF (NOT GPerfTools_FIND_QUIETLY)
        MESSAGE (STATUS "One or more MKL components were not found:")
        # Display missing components indented, each on a separate line
        FOREACH (_GPerfTools_MISSING_COMPONENT ${_GPerfTools_MISSING_COMPONENTS})
            MESSAGE (STATUS "  " ${_GPerfTools_MISSING_COMPONENT})
        ENDFOREACH (_GPerfTools_MISSING_COMPONENT ${_GPerfTools_MISSING_COMPONENTS})
    ENDIF (NOT GPerfTools_FIND_QUIETLY)
ENDIF (DEFINED _GPerfTools_MISSING_COMPONENTS AND _GPerfTools_CHECK_COMPONENTS)

#------------------------------------------------------------------------------#

mark_as_advanced(GPerfTools_INCLUDE_DIR)
find_package_handle_standard_args(GPerfTools DEFAULT_MSG
    GPerfTools_ROOT_DIR
    GPerfTools_INCLUDE_DIRS ${_GPerfTools_MISSING_LIBRARIES})
