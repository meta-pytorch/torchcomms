set_property(DIRECTORY PROPERTY PROCESSED)

# ----------------------------------------------------------------------------------------#

function(TIMEMORY_MAKEFILE_TARGET _VAR _TARG)
    string(REPLACE "-" "_" _FILE_TARG "${_TARG}")
    string(REGEX REPLACE ".*::" "" _FILE_TARG "${_FILE_TARG}")
    set(${_VAR}
        ${_FILE_TARG}
        PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------------------#

function(TIMEMORY_MAKEFILE_TARGET_NAME _VAR _TARG)
    string(REPLACE "-" "_" _FILE_TARG "${_TARG}")
    string(REGEX REPLACE ".*::" "" _FILE_TARG "${_FILE_TARG}")
    string(TOUPPER "${_FILE_TARG}" _NAME)
    set(${_VAR}
        ${_NAME}
        PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------------------#

macro(TIMEMORY_MAKEFILE_REMOVE_COMPILE_LANGUAGE)
    foreach(_ARG ${ARGN})
        if(NOT "${${_ARG}}" MATCHES "COMPILE_LANGUAGE")
            continue()
        endif()
        string(REPLACE "\$<\$<COMPILE_LANGUAGE:" "<<COMPILE_LANGUAGE:" ${_ARG}
                       "${${_ARG}}")
        string(REGEX REPLACE "<<COMPILE_LANGUAGE:(C|CXX|CUDA)>:" "" ${_ARG} "${${_ARG}}")
        string(REGEX REPLACE ">$" "" ${_ARG} "${${_ARG}}")
    endforeach()
endmacro()

# ----------------------------------------------------------------------------------------#

macro(TIMEMORY_MAKEFILE_REMOVE_COMPILER_ID)
    foreach(_ARG ${ARGN})
        if(NOT "${${_ARG}}" MATCHES "COMPILER_ID:")
            continue()
        else()
            set(${_ARG})
        endif()
    endforeach()
endmacro()

# ----------------------------------------------------------------------------------------#

function(TIMEMORY_MAKEFILE_TARGET_EXTRACT _TARG _TYPE)

    # don't process alias libraries
    string(REGEX REPLACE "^timemory::" "" _BASE_TARG "${_TARG}")
    if(TARGET "${_BASE_TARG}")
        set(_TARG ${_BASE_TARG})
    endif()

    get_property(
        GLOBAL_PROCESSED
        DIRECTORY
        PROPERTY GLOBAL_PROCESSED)
    get_property(
        PROCESSED
        DIRECTORY
        PROPERTY PROCESSED)
    # first
    set(_GLOBAL_FIRST OFF)
    set(_FIRST OFF)
    if("${PROCESSED}" STREQUAL "")
        set(_FIRST ON)
    endif()
    if("${_TARG}" IN_LIST PROCESSED)
        return()
    endif()
    if(NOT "${_TARG}" IN_LIST GLOBAL_PROCESSED)
        set(_GLOBAL_FIRST ON)
    endif()
    set_property(
        DIRECTORY
        APPEND
        PROPERTY GLOBAL_PROCESSED ${_TARG})
    set_property(
        DIRECTORY
        APPEND
        PROPERTY PROCESSED ${_TARG})

    # if(_GLOBAL_FIRST) message(STATUS "|_ Processing ${_TARG}...") endif()

    timemory_makefile_target(_FILE_TARG ${_TARG})
    timemory_makefile_target_name(_NAME ${_TARG})

    set(VARS C_OPTS CXX_OPTS CUDA_OPTS EXTRA_OPTS INCLUDE DEFINE LINKED DEPEND)
    foreach(_VAR ${VARS})
        set(${_VAR})
    endforeach()

    get_target_property(INCLUDE_DIRS ${_TARG} ${_TYPE}_INCLUDE_DIRECTORIES)
    get_target_property(COMPILE_OPTS ${_TARG} ${_TYPE}_COMPILE_OPTIONS)
    get_target_property(COMPILE_DEFS ${_TARG} ${_TYPE}_COMPILE_DEFINITIONS)
    get_target_property(LIBRARY_LINK ${_TARG} ${_TYPE}_LINK_LIBRARIES)
    get_target_property(LIBRARY_LDIR ${_TARG} ${_TYPE}_LINK_DIRECTORIES)

    if(INCLUDE_DIRS)
        foreach(_DIR ${INCLUDE_DIRS})
            if(_DIR MATCHES ".*<BUILD_INTERFACE:")
                continue()
            endif()
            if(_DIR MATCHES ".*<INSTALL_INTERFACE:")
                string(REPLACE "\$" "" _DIR "${_DIR}")
                string(REPLACE "<INSTALL_INTERFACE:" "" _DIR "${_DIR}")
                string(REPLACE ">" "" _DIR "${_DIR}")
                if(NOT EXISTS "${_DIR}")
                    set(_DIR "${CMAKE_INSTALL_PREFIX}/${_DIR}")
                endif()
            endif()
            list(APPEND INCLUDE "-I${_DIR}")
        endforeach()

        string(REPLACE ";" " " INCLUDE "${INCLUDE}")
        # message(STATUS "[${_FILE_TARG}]> Include: ${INCLUDE}")
    endif()

    if(COMPILE_OPTS)
        foreach(_OPT ${COMPILE_OPTS})
            foreach(_LANG C CXX CUDA)
                timemory_makefile_remove_compiler_id(_OPT)
                if(NOT _OPT)
                    continue()
                endif()

                if(_OPT MATCHES ".*COMPILE_LANGUAGE:${_LANG}>.*")

                    timemory_makefile_remove_compile_language(_OPT)
                    if(_OPT)
                        list(APPEND ${_LANG}_OPTS ${_OPT})
                    endif()
                else()

                    list(APPEND EXTRA_OPTS ${_OPT})

                endif()
            endforeach()
        endforeach()

        foreach(_TYPE C CXX CUDA EXTRA)
            string(REPLACE ";" " " ${_TYPE}_OPTS "${${_TYPE}_OPTS}")
        endforeach()

        # message(STATUS "[${_FILE_TARG}]> Compile C     : ${C_OPTS}") message(STATUS
        # "[${_FILE_TARG}]> Compile CXX   : ${CXX_OPTS}") message(STATUS "[${_FILE_TARG}]>
        # Compile CUDA  : ${CUDA_OPTS}") message(STATUS "[${_FILE_TARG}]> Compile EXTRA :
        # ${EXTRA_OPTS}")
    endif()

    if(COMPILE_DEFS)
        foreach(_DEF ${COMPILE_DEFS})
            # skip configuration generator expressions
            if("${_DEF}" MATCHES "<CONFIG:"
               OR "${_DEF}" MATCHES "<BUILD_INTERFACE:"
               OR "${_DEF}" MATCHES "TIMEMORY_CMAKE")
                continue()
            endif()
            timemory_makefile_remove_compile_language(_DEF)
            if(_DEF)
                list(APPEND DEFINE "-D${_DEF}")
            endif()
        endforeach()
        string(REPLACE ";" " " DEFINE "${DEFINE}")
        # message(STATUS "[${_FILE_TARG}]> Defines: ${DEFINE}")
    endif()

    if(LIBRARY_LINK)
        foreach(_LINK ${LIBRARY_LDIR})
            if(EXISTS ${_LINK})
                list(APPEND LINKED "-L${_LDIR}")
            endif()
        endforeach()
    endif()

    if(LIBRARY_LINK)
        foreach(_LINK ${LIBRARY_LINK})
            string(REPLACE "\$" "" _LINK "${_LINK}")
            string(REGEX REPLACE "<LINK_ONLY:([A-Za-z0-9-=].+)>" "\\1" _LINK "${_LINK}")
            if(TARGET ${_LINK})
                timemory_makefile_target_extract(${_LINK} INTERFACE)
                timemory_makefile_target_name(_DEP ${_LINK})
                list(APPEND DEPEND ${_DEP})
                get_target_property(_LINK_TARGET_TYPE ${_LINK} TYPE)
                if("${_LINK_TARGET_TYPE}" MATCHES "SHARED_LIBRARY"
                   OR "${_LINK_TARGET_TYPE}" MATCHES "STATIC_LIBRARY")
                    get_target_property(_LINK_OUTPUT_NAME ${_LINK} OUTPUT_NAME)
                    if(_LINK_OUTPUT_NAME)
                        list(APPEND LINKED "-l${_LINK_OUTPUT_NAME}")
                    else()
                        list(APPEND LINKED "-l${_LINK}")
                    endif()
                endif()
            else()
                if(EXISTS "${_LINK}")
                    get_filename_component(_LDIR "${_LINK}" PATH)
                    get_filename_component(_LNAME "${_LINK}" NAME_WE)
                    string(REGEX REPLACE "^lib" "" _LNAME "${_LNAME}")
                    list(APPEND LINKED "-L${_LDIR}")
                    list(APPEND LINKED "-l${_LNAME}")
                else()
                    if(NOT "${_LINK}" MATCHES "::@")
                        if(NOT "${_LINK}" MATCHES "^-")
                            list(APPEND LINKED "-l${_LINK}")
                        else()
                            list(APPEND LINKED "${_LINK}")
                        endif()
                    endif()
                endif()
            endif()
        endforeach()
        string(REPLACE ";" " " LINKED "${LINKED}")
        # message(STATUS "[${_FILE_TARG}]> Library: ${LINKED}")
    endif()

    # if called recursively, set variables in parent scope and return
    if(NOT _FIRST)
        foreach(_VAR ${VARS})
            set(${_NAME}_${_VAR}
                "${${_VAR}}"
                PARENT_SCOPE)
        endforeach()
        if(NOT _GLOBAL_FIRST)
            return()
        endif()
    endif()

    # add variables from depends
    foreach(_DEP ${DEPEND})
        foreach(_VAR ${VARS})
            if(${_DEP}_${_VAR})
                set(${_VAR} "${${_VAR}} ${${_DEP}_${_VAR}}")
            endif()
        endforeach()
    endforeach()

    # tweak spacing
    foreach(_VAR ${VARS})
        if(${_VAR})
            string(REPLACE "  " " " ${_VAR} "${${_VAR}}")
            string(REGEX REPLACE "^ " "" ${_VAR} "${${_VAR}}")
        endif()
    endforeach()

    file(
        APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
        "#
#   TARGET: ${_NAME}
#

")

    if(INCLUDE)
        file(APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
             "${_NAME}_INCLUDE = ${INCLUDE}\n")
    endif()

    if(LINKED)
        file(APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
             "${_NAME}_LIBS = ${LINKED}\n")
    endif()

    if(DEFINE)
        file(APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
             "${_NAME}_DEFS = ${DEFINE}\n")
    endif()

    if(C_OPTS)
        file(APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
             "${_NAME}_CFLAGS = ${C_OPTS}\n")
    endif()

    if(CXX_OPTS)
        file(APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
             "${_NAME}_CPPFLAGS = ${CXX_OPTS}\n")
    endif()

    if(CUDA_OPTS)
        file(APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
             "${_NAME}_CUFLAGS = ${CUDA_OPTS}\n")
    endif()

    if(DEPEND)
        list(REMOVE_DUPLICATES DEPEND)
        string(REPLACE ";" " " DEPEND "${DEPEND}")
        file(APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
             "${_NAME}_DEPENDS = ${DEPEND}\n")
    endif()

    if(INCLUDE
       OR LINKED
       OR DEFINE
       OR C_OPTS
       OR CXX_OPTS
       OR CUDA_OPTS
       OR DEPEND)
        file(APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc "\n")
    endif()

endfunction()

# ----------------------------------------------------------------------------------------#

message(STATUS "Generating Makefile.timemory.inc...")

set(TIMEMORY_MAKEFILE_LIBRARIES
    ${TIMEMORY_MAKEFILE_INTERFACE_LIBRARIES} ${TIMEMORY_MAKEFILE_COMPILED_LIBRARIES}
    ${TIMEMORY_TOOL_LIBRARIES})

list(REMOVE_DUPLICATES TIMEMORY_MAKEFILE_LIBRARIES)
list(SORT TIMEMORY_MAKEFILE_LIBRARIES)

set(MAKEFILE_LIBS)
foreach(_LIB ${TIMEMORY_MAKEFILE_LIBRARIES})
    timemory_makefile_target_name(_LIB ${_LIB})
    list(APPEND MAKEFILE_LIBS ${_LIB})
endforeach()
string(REPLACE ";" " " MAKEFILE_LIBS "${MAKEFILE_LIBS}")

file(
    WRITE ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
    "
#
# This file was auto-generated by the timemory CMake build system
#

TIMEMORY_MAKEFILE_LIBRARIES = ${MAKEFILE_LIBS}

")

# ----------------------------------------------------------------------------------------#

list(SORT TIMEMORY_MAKEFILE_LIBRARIES)

foreach(_TARG ${TIMEMORY_MAKEFILE_LIBRARIES})
    set_property(DIRECTORY PROPERTY PROCESSED "")
    timemory_makefile_target_extract(${_TARG} INTERFACE)
endforeach()

file(APPEND ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc "\n")

# ----------------------------------------------------------------------------------------#

message(STATUS "Generated outputs/Makefile.timemory.inc")

# ----------------------------------------------------------------------------------------#

if(TIMEMORY_INSTALL_CONFIG)
    install(
        FILES ${PROJECT_BINARY_DIR}/outputs/Makefile.timemory.inc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
        OPTIONAL)
endif()
