#[=======================================================================[.rst:
UPCXXConfig
-------

Configuration for UPC++.

UPC++ is a C++ library that supports Partitioned Global Address Space
(PGAS) programming, and is designed to interoperate smoothly and
efficiently with MPI, OpenMP, CUDA and AMTs. It leverages GASNet-EX to
deliver low-overhead, fine-grained communication, including Remote Memory
Access (RMA) and Remote Procedure Call (RPC).

This module checks if either the upcxx-meta utility can be found in the path
or in the bin sub-directory located inside the path pointed by the
``UPCXX_INSTALL`` environment variable if it is defined.

#]=======================================================================]

# Try to find the UPCXX library, headers, compile options, etc. Basic usage of this module
# is as follows:
#
# find_package(UPCXX) if(UPCXX_FOUND) add_executable(foo foo.cc) target_link_libraries(foo
# UPCXX::upcxx) endif()
#
# You can provide a minimum version number that should be used. If you provide this
# version number and specify the REQUIRED attribute, this module will fail if it can't
# find a UPCXX of the specified version or higher. If you further specify the EXACT
# attribute, then this module will fail if it can't find a UPCXX with a version eaxctly as
# specified.
#
# ===========================================================================
# Variables used by this module which can be used to change the default behaviour, and
# hence need to be set before calling find_package:
#
# ENV UPCXX_INSTALL   Used to find path to upcxx-meta
#
# UPCXX_ROOT_DIR      Used to find path to upcxx-meta if UPCXX_INSTALL not provided
#
# UPCXX_TARGET_NAME   Default is "upcxx" to create UPCXX::upcxx interface library. If
# multiple configurations are needed. Pair changing this variable with the COMPONENTS
# field in find_package, e.g.
#
# set(UPCXX_TARGET_NAME upcxx-seq-debug) find_package(UPCXX COMPONENTS par debug)
#
# set(UPCXX_TARGET_NAME upcxx-par-opt) find_package(UPCXX COMPONENTS seq O3)
#
# target_link_libraries(foo-debug UPCXX::upcxx-seq-debug) target_link_libraries(foo-opt
# UPCXX::upcxx-par-opt)
#
# ============================================================================
# Variables set by this module:
#
# UPCXX_FOUND             System has UPCXX.
#
# UPCXX_INCLUDE_DIRS      UPCXX include directories: not cached.
#
# UPCXX_LIBRARIES         Link to these to use the UPCXX library: not cached.
#
# UPCXX_OPTIONS           Compiler flags for C++
#
# UPCXX_LINK_OPTIONS      Linker flags
#
# UPCXX_DEFINITIONS       Pre-processor definitions
#
# UPCXX_CXX_STANDARD      C++ language standard
#
# UPCXX_CXX_STD_FLAG      C++ language standard flag used by UPC++ compiler
#

cmake_minimum_required(VERSION 3.6)
cmake_policy(PUSH)
# ensure that IN_LIST operator is not ignored:
# https://cmake.org/cmake/help/v3.3/policy/CMP0057.html
cmake_policy(SET CMP0057 NEW)

# these are the variables set in the script unset them in case multiple configurations are
# generated
set(UPCXX_VARIABLES
    UPCXX_FOUND
    UPCXX_META_EXECUTABLE
    UPCXX_INCLUDE_DIRS
    UPCXX_LIBRARIES
    UPCXX_DEFINITIONS
    UPCXX_CXX_STANDARD
    UPCXX_OPTIONS
    UPCXX_LINK_OPTIONS
    UPCXX_COMPATIBLE_COMPILER)

foreach(_VAR ${UPCXX_VARIABLES})
    unset(${_VAR})
endforeach()

# option for verbosity
option(UPCXX_VERBOSE "Verbose UPC++ detection" OFF)
mark_as_advanced(UPCXX_VERBOSE)
function(UPCXX_VERB MESSAGE)
    if(UPCXX_VERBOSE OR DEFINED ENV{UPCXX_VERBOSE})
        message(STATUS "${MESSAGE}")
    endif()
endfunction()

# default to no preference
set(UPCXX_THREADMODE
    ""
    CACHE STRING "UPC++ thread-mode")
set(UPCXX_CODEMODE
    ""
    CACHE STRING "UPC++ code mode")
set(UPCXX_NETWORK
    ""
    CACHE STRING "UPC++ networking mode")

# these appear to have defaults but thread-mode does not
mark_as_advanced(UPCXX_THREADMODE UPCXX_CODEMODE UPCXX_NETWORK)

# for IN_LIST checks
set(UPCXX_THREADMODE_STRINGS "seq" "par")
set(UPCXX_CODEMODE_STRINGS "O3" "debug")
set(UPCXX_NETWORK_STRINGS "ibv" "aries" "smp" "udp" "mpi")

# set options in GUI
set_property(CACHE UPCXX_THREADMODE PROPERTY STRINGS "${UPCXX_THREADMODE_STRINGS}")
set_property(CACHE UPCXX_CODEMODE PROPERTY STRINGS "${UPCXX_CODEMODE_STRINGS}")
set_property(CACHE UPCXX_NETWORK PROPERTY STRINGS "${UPCXX_NETWORK_STRINGS}")

# macro for checking if value is in list of possible options
macro(UPCXX_CHECK_PROPERTY_STRING _TYPE)
    if(NOT "${UPCXX_${_TYPE}}" IN_LIST UPCXX_${_TYPE}_STRINGS)
        message(
            FATAL_ERROR
                "Invalid value for UPCXX_${_TYPE}: '${UPCXX_${_TYPE}}'. Valid options: '${UPCXX_${_TYPE}_STRINGS}'"
            )
    endif()
endmacro()

# ----------------------------------------------------------------------------------------#
#
# UPCXX_THREADMODE
#
# ----------------------------------------------------------------------------------------#

if((UPCXX_THREADMODE STREQUAL "" OR NOT DEFINED UPCXX_THREADMODE)
   AND "$ENV{UPCXX_THREADMODE}" STREQUAL "")
    # if find_package(Threads) set Threads_FOUND previously, this suggests threading is
    # used by the application and would be a good default but otherwise, leave blank
    if(Threads_FOUND)
        set(UPCXX_THREADMODE "par")
    else()
        set(UPCXX_THREADMODE "seq")
    endif()
elseif(UPCXX_THREADMODE)
    upcxx_check_property_string(THREADMODE)
endif()

# ----------------------------------------------------------------------------------------#
#
# UPCXX_CODEMODE
#
# ----------------------------------------------------------------------------------------#

if((UPCXX_CODEMODE STREQUAL "" OR NOT DEFINED UPCXX_CODEMODE) AND "$ENV{UPCXX_CODEMODE}"
                                                                  STREQUAL "")
    # use CMAKE_BUILD_TYPE to provide a default choice
    if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
        set(UPCXX_CODEMODE "debug")
    elseif("${CMAKE_BUILD_TYPE}" MATCHES "Release|RelWithDebInfo|MinSizeRel")
        set(UPCXX_CODEMODE "O3")
    endif()
elseif(UPCXX_CODEMODE)
    upcxx_check_property_string(CODEMODE)
endif()

# ----------------------------------------------------------------------------------------#
#
# UPCXX_NETWORK
#
# ----------------------------------------------------------------------------------------#

if((UPCXX_NETWORK STREQUAL "" OR NOT DEFINED UPCXX_NETWORK) AND "$ENV{UPCXX_NETWORK}"
                                                                STREQUAL "")
    # no defaults provided -- let upcxx-meta choose
elseif(UPCXX_NETWORK)
    upcxx_check_property_string(NETWORK)
endif()

# ----------------------------------------------------------------------------------------#
#
# UPCXX_FIND_COMPONENTS, e.g:     find_package(UPCXX COMPONENTS udp seq debug)
#
# ----------------------------------------------------------------------------------------#

# for the find_package_handle_standard_args
set(UPCXX_FOUND_COMPONENTS)

# loop over all components requested, e.g. find_package(UPCXX COMPONENTS udp seq debug)
foreach(_COMPONENT ${UPCXX_FIND_COMPONENTS})
    foreach(_VAR THREADMODE CODEMODE NETWORK)
        if("${_COMPONENT}" IN_LIST UPCXX_${_VAR}_STRINGS)
            # remove the item from the missing list
            list(REMOVE_ITEM UPCXX_FIND_COMPONENTS "${_COMPONENT}")
            list(APPEND UPCXX_FOUND_COMPONENTS "${_COMPONENT}")
            # set the variable
            set(UPCXX_${_VAR} "${_COMPONENT}")
        endif()
    endforeach()
endforeach()

if(UPCXX_FOUND_COMPONENTS AND NOT UPCXX_FIND_QUIETLY)
    message(STATUS "UPCXX found components: ${UPCXX_FOUND_COMPONENTS}")
endif()

# ----------------------------------------------------------------------------------------#
#
# Propagate to environment
#
# ----------------------------------------------------------------------------------------#

foreach(_VAR THREADMODE CODEMODE NETWORK)
    if(UPCXX_${_VAR})
        # set the variable in the environment for upcxx-meta
        set(ENV{UPCXX_${_VAR}} "${UPCXX_${_VAR}}")
    endif()
endforeach()

# Set up some auxillary vars if hints have been set
if(DEFINED ENV{UPCXX_INSTALL})
    find_program(
        UPCXX_META_EXECUTABLE upcxx-meta
        HINTS "$ENV{UPCXX_INSTALL}/bin"
        NO_DEFAULT_PATH)
else()
    find_program(
        UPCXX_META_EXECUTABLE upcxx-meta
        HINTS UPCXX_ROOT_DIR ENV UPCXX_ROOT_DIR
        PATH_SUFFIXES bin)
endif()

if(NOT EXISTS "${UPCXX_META_EXECUTABLE}" AND NOT UPCXX_FIND_QUIETLY)
    message(
        WARNING
            "Failed to find UPC++ command interface 'upcxx-meta'. Please set UPCXX_INSTALL=/path/to/upcxx or add /path/to/upcxx/bin to $PATH"
        )
endif()

if(UPCXX_META_EXECUTABLE)
    execute_process(
        COMMAND ${UPCXX_META_EXECUTABLE} CXXFLAGS
        OUTPUT_VARIABLE UPCXX_CXXFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND ${UPCXX_META_EXECUTABLE} CPPFLAGS
        OUTPUT_VARIABLE UPCXX_CPPFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND ${UPCXX_META_EXECUTABLE} LIBS
        OUTPUT_VARIABLE UPCXX_LIBFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND ${UPCXX_META_EXECUTABLE} LDFLAGS
        OUTPUT_VARIABLE UPCXX_LDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND ${UPCXX_META_EXECUTABLE} CXX
        OUTPUT_VARIABLE UPCXX_CXX_COMPILER
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND ${UPCXX_META_EXECUTABLE} GASNET_CONDUIT
        OUTPUT_VARIABLE UPCXX_NETWORK
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    list(APPEND UPCXX_LIBRARIES ${UPCXX_LIBFLAGS})

    # move any embedded options from UPCXX_CXX_COMPILER to UPCXX_CXXFLAGS
    if(UPCXX_CXX_COMPILER MATCHES "[ \t\n]+(.+)$")
        upcxx_verb("embedded CXX options: ${CMAKE_MATCH_0}")
        string(REGEX REPLACE "[ \t\n]+.+$" "" UPCXX_CXX_COMPILER ${UPCXX_CXX_COMPILER})
        set(UPCXX_CXXFLAGS "${CMAKE_MATCH_0} ${UPCXX_CXXFLAGS}")
        string(STRIP ${UPCXX_CXXFLAGS} UPCXX_CXXFLAGS)
    endif()
    upcxx_verb("UPCXX_CXX_COMPILER=${UPCXX_CXX_COMPILER}")
    upcxx_verb("CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
    # get absolute path, resolving symbolic links, of UPCXX_CXX_COMPILER
    get_filename_component(ABS_UPCXX_CXX_PATH "${UPCXX_CXX_COMPILER}" REALPATH CACHE)
    if(NOT EXISTS "${ABS_UPCXX_CXX_PATH}" AND NOT UPCXX_FIND_QUIETLY)
        message(
            WARNING
                "CANNOT FIND ABSOLUTE PATH TO UPCXX_CXX_COMPILER (${UPCXX_CXX_COMPILER})")
        set(ABS_UPCXX_CXX_PATH "${UPCXX_CXX_COMPILER}")
    endif()

    # get absolute path, resolving symbolic links, of CMAKE_CXX_COMPILER
    get_filename_component(ABS_CMAKE_CXX_PATH ${CMAKE_CXX_COMPILER} REALPATH CACHE)
    upcxx_verb("ABS_UPCXX_CXX_PATH=${ABS_UPCXX_CXX_PATH}")
    upcxx_verb("ABS_CMAKE_CXX_PATH=${ABS_CMAKE_CXX_PATH}")

    set(UPCXX_COMPATIBLE_COMPILER FALSE)
    # compare the files -- generally more reliable than string parse + compare
    execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files ${ABS_UPCXX_CXX_PATH}
                            ${ABS_CMAKE_CXX_PATH} RESULT_VARIABLE COMPILER_FILE_DIFF)

    if(("${ABS_UPCXX_CXX_PATH}" STREQUAL "${ABS_CMAKE_CXX_PATH}") OR (COMPILER_FILE_DIFF
                                                                      EQUAL 0))
        set(UPCXX_COMPATIBLE_COMPILER TRUE)
    else()
        get_filename_component(UPCXX_CXX_NAME ${ABS_UPCXX_CXX_PATH} NAME)
        get_filename_component(CMAKE_CXX_NAME ${ABS_CMAKE_CXX_PATH} NAME)
        upcxx_verb("compiler names: ${UPCXX_CXX_NAME} vs ${CMAKE_CXX_NAME}")
        if("${UPCXX_CXX_NAME}" STREQUAL "${CMAKE_CXX_NAME}")
            # compare the versions
            execute_process(COMMAND ${UPCXX_CXX_COMPILER} --version
                            OUTPUT_VARIABLE UPCXX_CXX_COMPILER_VERSION)
            string(REPLACE "\n" " " UPCXX_CXX_COMPILER_VERSION
                           ${UPCXX_CXX_COMPILER_VERSION})
            execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
                            OUTPUT_VARIABLE LOC_CMAKE_CXX_COMPILER_VERSION)
            string(REPLACE "\n" " " LOC_CMAKE_CXX_COMPILER_VERSION
                           ${LOC_CMAKE_CXX_COMPILER_VERSION})
            # message(STATUS "${UPCXX_CXX_COMPILER_VERSION} vs
            # ${LOC_CMAKE_CXX_COMPILER_VERSION}")
            if("${UPCXX_CXX_COMPILER_VERSION}" STREQUAL
               "${LOC_CMAKE_CXX_COMPILER_VERSION}")
                set(UPCXX_COMPATIBLE_COMPILER TRUE)
            endif()
        endif()
    endif()

    if(NOT UPCXX_COMPATIBLE_COMPILER AND NOT UPCXX_FIND_QUIETLY)
        message(
            WARNING
                "Compiler compatibility check failed!\nUPCXX compiler provided by upcxx-meta CXX:\n    ${UPCXX_CXX_COMPILER} ->\n    ${ABS_UPCXX_CXX_PATH}\nis different from CMAKE_CXX_COMPILER:\n    ${CMAKE_CXX_COMPILER} ->\n    ${ABS_CMAKE_CXX_PATH}\n\nPlease either pass cmake: -DCMAKE_CXX_COMPILER=${UPCXX_CXX_COMPILER}\nor re-install UPC++ with: CXX=${CMAKE_CXX_COMPILER}\n"
            )
    endif()

    unset(ABS_UPCXX_CXX_PATH)
    unset(ABS_CMAKE_CXX_PATH)
    unset(UPCXX_CXX_NAME)
    unset(CMAKE_CXX_NAME)
    unset(UPCXX_CXX_COMPILER_VERSION)
    unset(COMPILER_FILE_DIFF)

    # now separate include dirs from flags
    if(UPCXX_CPPFLAGS)
        string(REGEX REPLACE "[ ]+" ";" UPCXX_CPPFLAGS ${UPCXX_CPPFLAGS})
        foreach(option ${UPCXX_CPPFLAGS})
            string(STRIP ${option} option)
            string(REGEX MATCH "^-I" UPCXX_INCLUDE ${option})
            if(UPCXX_INCLUDE)
                string(REGEX REPLACE "^-I" "" option ${option})
                list(APPEND UPCXX_INCLUDE_DIRS ${option})
            else()
                string(REGEX MATCH "^-D" UPCXX_DEFINE ${option})
                if(UPCXX_DEFINE)
                    string(REGEX REPLACE "^-D" "" option ${option})
                    list(APPEND UPCXX_DEFINITIONS ${option})
                else()
                    list(APPEND UPCXX_OPTIONS ${option})
                endif()
            endif()
        endforeach()
    endif()
    if(UPCXX_LDFLAGS)
        string(REGEX REPLACE "[ ]+" ";" UPCXX_LDFLAGS ${UPCXX_LDFLAGS})
        foreach(option ${UPCXX_LDFLAGS})
            string(STRIP ${option} option)
            if(option MATCHES "^-O" AND CMAKE_BUILD_TYPE)
                # filter -O options when CMake is handling that
            else()
                list(APPEND UPCXX_LINK_OPTIONS ${option})
            endif()
        endforeach()
    endif()

    # extract the required cxx standard from the flags
    if(UPCXX_CXXFLAGS)
        string(REGEX REPLACE "[ ]+" ";" UPCXX_CXXFLAGS ${UPCXX_CXXFLAGS})
        foreach(option ${UPCXX_CXXFLAGS})
            if(option MATCHES "^-+std=(c|gnu)\\+\\+([0-9]+)")
                set(UPCXX_CXX_STD_FLAG ${option})
                set(UPCXX_CXX_STANDARD ${CMAKE_MATCH_2})
            elseif(option MATCHES "^-O" AND CMAKE_BUILD_TYPE)
                # filter -O options when CMake is handling that
            else()
                list(APPEND UPCXX_OPTIONS ${option})
            endif()
        endforeach()
    endif()

    unset(UPCXX_CXXFLAGS)
    unset(UPCXX_LIBFLAGS)
    unset(UPCXX_CPPFLAGS)
    unset(UPCXX_LDFLAGS)
    unset(UPCXX_INCLUDE)
    unset(UPCXX_DEFINE)
endif()

foreach(dir ${UPCXX_INCLUDE_DIRS})
    if(EXISTS ${dir}/upcxx/upcxx.hpp)
        set(version_pattern "#[\t ]*define[\t ]+UPCXX_VERSION[\t ]+([0-9]+)")
        # message(STATUS "checking ${dir}/upcxx/upcxx.hpp for ${version_pattern}" )
        file(STRINGS ${dir}/upcxx/upcxx.hpp upcxx_version REGEX ${version_pattern})
        # message(STATUS "upcxx_version ${upcxx_version}" )

        if(${upcxx_version} MATCHES ${version_pattern})
            set(UPCXX_VERSION_STRING ${CMAKE_MATCH_1})
        endif()

        unset(upcxx_version)
        unset(version_pattern)
    endif()
endforeach()

# CMake bug #15826: CMake's ill-advised deduplication mis-feature breaks certain types of
# compiler arguments, see: https://gitlab.kitware.com/cmake/cmake/issues/15826 Here we
# workaround the problem as best we can to prevent compile failures
function(UPCXX_FIX_FRAGILE_OPTS var)
    set(fragile_option_pat ";(-+param);([^;]+);")
    set(temp ";${${var}};")
    while(temp MATCHES "${fragile_option_pat}")
        # should NOT need a loop here, but regex replace is buggy, at least in cmake 3.6
        if(CMAKE_VERSION VERSION_LESS 3.12.0)
            # no known workaround, must strip these options
            string(REGEX REPLACE "${fragile_option_pat}" ";" temp "${temp}")
        else()
            # use the SHELL: prefix introduced in cmake 3.12
            string(REGEX REPLACE "${fragile_option_pat}" ";SHELL:\\1 \\2;" temp "${temp}")
        endif()
    endwhile()
    list(FILTER temp EXCLUDE REGEX "^$") # strip surrounding empties
    set("${var}"
        "${temp}"
        PARENT_SCOPE)
endfunction()
upcxx_fix_fragile_opts(UPCXX_OPTIONS)
upcxx_fix_fragile_opts(UPCXX_LINK_OPTIONS)

# Determine if we've found UPCXX
mark_as_advanced(
    UPCXX_FOUND
    UPCXX_META_EXECUTABLE
    UPCXX_INCLUDE_DIRS
    UPCXX_LIBRARIES
    UPCXX_DEFINITIONS
    UPCXX_CXX_STANDARD
    UPCXX_OPTIONS
    UPCXX_LINK_OPTIONS
    UPCXX_COMPATIBLE_COMPILER)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    UPCXX
    REQUIRED_VARS UPCXX_META_EXECUTABLE UPCXX_LIBRARIES UPCXX_INCLUDE_DIRS
                  UPCXX_DEFINITIONS UPCXX_LINK_OPTIONS UPCXX_COMPATIBLE_COMPILER
    VERSION_VAR UPCXX_VERSION_STRING
    HANDLE_COMPONENTS)

if(NOT UPCXX_FIND_QUIETLY)
    if(NOT "${UPCXX_CXX_STANDARD}" STREQUAL "${CMAKE_CXX_STANDARD}")
        message(STATUS "UPC++ requires the c++${UPCXX_CXX_STANDARD} standard.")
    endif()
    message(STATUS "UPCXX_NETWORK=${UPCXX_NETWORK}")
    message(STATUS "UPCXX_THREADMODE=$ENV{UPCXX_THREADMODE}")
    message(STATUS "UPCXX_CODEMODE=$ENV{UPCXX_CODEMODE}")
endif()

# ----------------------------------------------------------------------------------------#
#
# UPCXX_TARGET_NAME <-- set this variable if multiple configuration are needed
#
# ----------------------------------------------------------------------------------------#

if(NOT UPCXX_TARGET_NAME)
    set(UPCXX_TARGET_NAME upcxx)
endif()

# Export a UPCXX::upcxx target for modern cmake projects
if(UPCXX_FOUND AND NOT TARGET UPCXX::${UPCXX_TARGET_NAME})
    add_library(UPCXX::${UPCXX_TARGET_NAME} INTERFACE IMPORTED)
    # Handle various CMake version dependencies
    if(NOT CMAKE_VERSION VERSION_LESS 3.8.0)
        set_property(TARGET UPCXX::${UPCXX_TARGET_NAME}
                     PROPERTY INTERFACE_COMPILE_FEATURES "cxx_std_${UPCXX_CXX_STANDARD}")
    elseif(DEFINED UPCXX_CXX_STD_FLAG)
        upcxx_verb("UPCXX_CXX_STD_FLAG=${UPCXX_CXX_STD_FLAG}")
        list(APPEND UPCXX_OPTIONS ${UPCXX_CXX_STD_FLAG})
    endif()
    if(CMAKE_VERSION VERSION_LESS 3.13.0)
        list(APPEND UPCXX_OPTIONS "$<LINK_ONLY:${UPCXX_LINK_OPTIONS}>")
    else()
        set_property(TARGET UPCXX::${UPCXX_TARGET_NAME} PROPERTY INTERFACE_LINK_OPTIONS
                                                                 "${UPCXX_LINK_OPTIONS}")
    endif()
    set_target_properties(
        UPCXX::${UPCXX_TARGET_NAME}
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${UPCXX_INCLUDE_DIRS}"
                   INTERFACE_LINK_LIBRARIES "${UPCXX_LIBRARIES}"
                   INTERFACE_COMPILE_DEFINITIONS "${UPCXX_DEFINITIONS}"
                   # ensures that UPCXX_OPTIONS is only added for CXX compilation:
                   INTERFACE_COMPILE_OPTIONS
                   "$<$<COMPILE_LANGUAGE:CXX>:${UPCXX_OPTIONS}>")
    # set(UPCXX_LIBRARIES UPCXX::${UPCXX_TARGET_NAME})
    upcxx_verb("UPCXX_INCLUDE_DIRS: ${UPCXX_INCLUDE_DIRS}")
    upcxx_verb("UPCXX_DEFINITIONS:  ${UPCXX_DEFINITIONS}")
    upcxx_verb("UPCXX_OPTIONS:      ${UPCXX_OPTIONS}")
    upcxx_verb("UPCXX_LINK_OPTIONS: ${UPCXX_LINK_OPTIONS}")
    upcxx_verb("UPCXX_LIBRARIES:    ${UPCXX_LIBRARIES}")
endif()

# pop the policy scope
cmake_policy(POP)
