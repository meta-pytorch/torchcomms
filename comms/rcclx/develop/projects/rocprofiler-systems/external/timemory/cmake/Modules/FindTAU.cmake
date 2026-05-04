# Try to find the TAU library and headers Usage of this module is as follows
#
# find_package( TAU ) if(TAU_FOUND) include_directories(${TAU_INCLUDE_DIRS})
# add_executable(foo foo.cc) target_link_libraries(foo ${TAU_LIBRARIES}) endif()
#
# You can provide a minimum version number that should be used. If you provide this
# version number and specify the REQUIRED attribute, this module will fail if it can't
# find a TAU of the specified version or higher. If you further specify the EXACT
# attribute, then this module will fail if it can't find a TAU with a version eaxctly as
# specified.
#
# ===========================================================================
# Variables used by this module which can be used to change the default behaviour, and
# hence need to be set before calling find_package:
#
# TAU_ROOT_DIR The preferred installation prefix for searching for TAU Set this if the
# module has problems finding the proper TAU installation.
#
# If you don't supply TAU_ROOT_DIR, the module will search on the standard system paths.
#
# ============================================================================
# Variables set by this module:
#
# TAU_FOUND           System has TAU.
#
# TAU_INCLUDE_DIRS    TAU include directories: not cached.
#
# TAU_LIBRARIES       Link to these to use the TAU library: not cached.
#
# ===========================================================================
# If TAU is installed in a non-standard way, e.g. a non GNU-style install of
# <prefix>/{lib,include}, then this module may fail to locate the headers and libraries as
# needed. In this case, the following cached variables can be editted to point to the
# correct locations.
#
# TAU_INCLUDE_DIR    The path to the TAU include directory: cached
#
# TAU_LIBRARIES        The path to the TAU library: cached
#
# You should not need to set these in the vast majority of cases
#

# ----------------------------------------------------------------------------------------#

find_program(
    TAU_CXX_COMPILER
    NAMES tau_cxx.sh
    PATH_SUFFIXES bin)

# ----------------------------------------------------------------------------------------#

function(TAU_CXX_DEFINE_SHOW_VARIABLES)
    set(_TAU_CXX_EXECUTE_OPTIONS)
    if(NOT CMAKE_VERSION VERSION_LESS 3.18)
        list(APPEND _TAU_CXX_EXECUTE_OPTIONS ECHO_ERROR_VARIABLE)
    endif()
    if(NOT CMAKE_VERSION VERSION_LESS 3.19)
        list(APPEND _TAU_CXX_EXECUTE_OPTIONS COMMAND_ERROR_IS_FATAL ANY)
    endif()
    foreach(_OPT SHOW_COMPILER SHOW SHOW_INCLUDES SHOW_LIBS SHOW_SHARED_LIBS)
        string(TOLOWER "${_OPT}" _TAU_CXX_${_OPT}_OPT)
        string(REPLACE "_" "" _TAU_CXX_${_OPT}_OPT "${_TAU_CXX_${_OPT}_OPT}")
        set(TAU_CXX_${_OPT}_OPT
            "${_TAU_CXX_${_OPT}_OPT}"
            CACHE STRING "tau_cxx.sh -tau:\${TAU_CXX_${_OPT}_OPT}")
        mark_as_advanced(TAU_CXX_${_OPT}_OPT)
        if(NOT TAU_CXX_${_OPT})
            execute_process(
                COMMAND ${TAU_CXX_COMPILER} -tau:${TAU_CXX_${_OPT}_OPT}
                OUTPUT_VARIABLE _OUT
                RESULT_VARIABLE _RET
                ERROR_VARIABLE _ERR
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE
                COMMAND_ECHO STDOUT ${_TAU_CXX_EXECUTE_OPTIONS})
            if(NOT _RET EQUAL 0)
                message(
                    SEND_ERROR
                        "`${TAU_CXX_COMPILER} -tau:${TAU_CXX_${_OPT}_OPT}` exited with error code: ${_RET}"
                    )
                message(SEND_ERROR "tau_cxx.sh output :: ${_OUT}")
                if(NOT "ECHO_ERROR_VARIABLE" IN_LIST _TAU_CXX_EXECUTE_OPTIONS)
                    message(SEND_ERROR "tau_cxx.sh error  :: ${_ERR}")
                endif()
                continue()
            endif()
            if("${_OPT}" STREQUAL "SHOW")
                string(REPLACE "${TAU_CXX_SHOW_COMPILER} " "" _OUT "${_OUT}")
            endif()
            if("${TAU_CXX_${_OPT}}" STREQUAL "")
                unset(TAU_CXX_${_OPT} CACHE)
            endif()
            set(TAU_CXX_${_OPT}
                "${_OUT}"
                CACHE STRING "output from `tau_cxx.sh -tau:${TAU_CXX_${_OPT}_OPT}`")
            mark_as_advanced(TAU_CXX_${_OPT})
        endif()
    endforeach()
endfunction()

function(_SET_TAU_PROCESS_VARIABLE _VAR)
    string(REGEX REPLACE "[ \t]$" "" _VAL "${ARGN}")
    set(TAU_CXX_${_VAR}
        "${_VAL}"
        CACHE STRING "Regex for matching to flag")
    mark_as_advanced(TAU_CXX_${_VAR})
endfunction()

# variables for matching the prefix of an option
_set_tau_process_variable(INCLUDE_DIRS_MATCH "-I|${CMAKE_INCLUDE_SYSTEM_FLAG_CXX}")
_set_tau_process_variable(DEFINITIONS_MATCH "-D|/D")
_set_tau_process_variable(LIBRARY_DIRS_MATCH "-L")
_set_tau_process_variable(LIBRARIES_MATCH "-l(TAU|tau)")
_set_tau_process_variable(LINK_OPTIONS_MATCH "-Wl")

# variables for replacing the prefix of an option
_set_tau_process_variable(INCLUDE_DIRS_REPLACE "${TAU_CXX_INCLUDE_DIRS_MATCH}")
_set_tau_process_variable(DEFINITIONS_REPLACE "${TAU_CXX_DEFINITIONS_MATCH}")
_set_tau_process_variable(LIBRARY_DIRS_REPLACE "${TAU_CXX_LIBRARY_DIRS_MATCH}")
_set_tau_process_variable(LIBRARIES_REPLACE "-l")
_set_tau_process_variable(LINK_OPTIONS_REPLACE)

function(TAU_PROCESS_CXX_OUTPUT _VAR_PREFIX _STR)
    set(_VARS)
    foreach(_VAR INCLUDE_DIRS DEFINITIONS LIBRARY_DIRS LIBRARIES LINK_OPTIONS)
        if("${_VAR}" IN_LIST ARGN AND NOT "${_VAR_PREFIX}_${_VAR}")
            list(APPEND _VARS "${_VAR}")
        endif()
    endforeach()

    string(REPLACE " " ";" _STR "${_STR}")

    foreach(_VAR ${_VARS})
        set(_${_VAR})
        set(_MATCH "${TAU_CXX_${_VAR}_MATCH}")
        set(_REPLACE "${TAU_CXX_${_VAR}_REPLACE}")
        set(_ADD_NEXT OFF)
        foreach(_VAL ${_STR})
            if(_ADD_NEXT)
                set(_ADD_NEXT OFF)
                list(APPEND _${_VAR} "${_VAL}")
                continue()
            elseif(NOT "${_VAL}" MATCHES "^(${_MATCH})")
                continue()
            elseif("${_VAL}" MATCHES "^(${_MATCH})$")
                set(_ADD_NEXT ON)
            else()
                if(_REPLACE)
                    string(REGEX REPLACE "^(${_REPLACE})(.*)" "\\2" _VAL "${_VAL}")
                endif()
                if(_VAL)
                    list(APPEND _${_VAR} "${_VAL}")
                endif()
            endif()
        endforeach()
        if(_${_VAR})
            set(${_VAR_PREFIX}_${_VAR}
                "${_${_VAR}}"
                CACHE STRING "${_VAR} for TAU")
        endif()
    endforeach()
endfunction()

if(TAU_CXX_COMPILER)
    tau_cxx_define_show_variables()

    tau_process_cxx_output(TAU "${TAU_CXX_SHOW}" DEFINITIONS)
    tau_process_cxx_output(TAU "${TAU_CXX_SHOW_INCLUDES}" INCLUDE_DIRS)

    if(BUILD_SHARED_LIBS
       OR "shared" IN_LIST TAU_FIND_COMPONENTS
       OR (BUILD_STATIC_LIBS AND CMAKE_POSITION_INDEPENDENT_CODE))
        tau_process_cxx_output(TAU "${TAU_CXX_SHOW_SHARED_LIBS}" LIBRARY_DIRS LIBRARIES
                               LINK_OPTIONS)
        if(TAU_LIBRARIES)
            set(TAU_shared_FOUND ON)
        endif()
    elseif(BUILD_STATIC_LIBS OR "static" IN_LIST TAU_FIND_COMPONENTS)
        tau_process_cxx_output(TAU "${TAU_CXX_SHOW_LIBS}" LIBRARY_DIRS LIBRARIES
                               LINK_OPTIONS)
        if(TAU_LIBRARIES)
            set(TAU_static_FOUND ON)
        endif()
    else()
        tau_process_cxx_output(TAU "${TAU_CXX_SHOW_SHARED_LIBS}" LIBRARY_DIRS LIBRARIES
                               LINK_OPTIONS)
    endif()
endif()

# ----------------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set TAU_FOUND to TRUE if all listed
# variables are TRUE
find_package_handle_standard_args(
    TAU
    REQUIRED_VARS TAU_INCLUDE_DIRS TAU_LIBRARIES
    HANDLE_COMPONENTS)

# ----------------------------------------------------------------------------------------#

if(TAU_FOUND)
    add_library(TAU::TAU INTERFACE IMPORTED)
    target_link_libraries(TAU::TAU INTERFACE ${TAU_LIBRARIES})
    target_compile_definitions(TAU::TAU INTERFACE ${TAU_DEFINITIONS})
    target_include_directories(TAU::TAU SYSTEM INTERFACE ${TAU_INCLUDE_DIRS})
    target_link_directories(TAU::TAU INTERFACE ${TAU_LIBRARY_DIRS})
    target_link_options(TAU::TAU INTERFACE ${TAU_LINK_OPTIONS})
endif()

mark_as_advanced(TAU_INCLUDE_DIRS TAU_DEFINITIONS TAU_LIBRARY_DIRS TAU_LIBRARIES
                 TAU_LINK_OPTIONS)
