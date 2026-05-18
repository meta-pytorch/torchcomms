# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

#------------------------------------------------------------------------------#

set(_ITTNOTIFY_PATH_HINTS)

#------------------------------------------------------------------------------#
# function to check for hints in environment
function(_ITTNOTIFY_CHECK_VAR)
    foreach(_VAR ${ARGN})
        if(NOT "$ENV{${_VAR}}" STREQUAL "")
            list(APPEND _HINTS "$ENV{${_VAR}}")
        endif()
        if(NOT "${${_VAR}}" STREQUAL "")
            list(APPEND _HINTS "${${_VAR}}")
        endif()
    endforeach()
    set(_ITTNOTIFY_PATH_HINTS ${_ITTNOTIFY_PATH_HINTS} ${_HINTS} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------#

_ittnotify_check_var(VTUNE_AMPLIFIER_DIR VTUNE_AMPLIFIER_XE_DIR)
_ittnotify_check_var(VTUNE_AMPLIFIER_2019_DIR VTUNE_AMPLIFIER_XE_2019_DIR)
_ittnotify_check_var(VTUNE_AMPLIFIER_2018_DIR VTUNE_AMPLIFIER_XE_2018_DIR)
_ittnotify_check_var(VTUNE_AMPLIFIER_2017_DIR VTUNE_AMPLIFIER_XE_2017_DIR)

#------------------------------------------------------------------------------#

find_path(ITTNOTIFY_ROOT_DIR
    NAMES include/ittnotify.h
    HINTS ${_ITTNOTIFY_PATH_HINTS}
    PATHS ${_ITTNOTIFY_PATH_HINTS}
)

#------------------------------------------------------------------------------#

find_path(ITTNOTIFY_INCLUDE_DIR
    NAMES ittnotify.h
    PATH_SUFFIXES include
    HINTS ${ITTNOTIFY_ROOT_DIR} ${_ITTNOTIFY_PATH_HINTS}
    PATHS ${ITTNOTIFY_ROOT_DIR} ${_ITTNOTIFY_PATH_HINTS}
)

#------------------------------------------------------------------------------#

find_library(ITTNOTIFY_LIBRARY
    NAMES ittnotify
    PATH_SUFFIXES lib lib64 lib32
    HINTS ${ITTNOTIFY_ROOT_DIR} ${_ITTNOTIFY_PATH_HINTS}
    PATHS ${ITTNOTIFY_ROOT_DIR} ${_ITTNOTIFY_PATH_HINTS}
)

#------------------------------------------------------------------------------#

if(ITTNOTIFY_INCLUDE_DIR)
    set(ITTNOTIFY_INCLUDE_DIRS ${ITTNOTIFY_INCLUDE_DIR})
endif()

#------------------------------------------------------------------------------#

if(ITTNOTIFY_LIBRARY)
    set(ITTNOTIFY_LIBRARIES ${ITTNOTIFY_LIBRARY})
endif()

#------------------------------------------------------------------------------#

mark_as_advanced(ITTNOTIFY_ROOT_DIR ITTNOTIFY_INCLUDE_DIR ITTNOTIFY_LIBRARY)
find_package_handle_standard_args(ittnotify DEFAULT_MSG
    ITTNOTIFY_ROOT_DIR ITTNOTIFY_INCLUDE_DIR ITTNOTIFY_LIBRARY)

#------------------------------------------------------------------------------#

unset(_ITTNOTIFY_PATH_HINTS)

#------------------------------------------------------------------------------#
