# only include this once
include_guard(GLOBAL)

# ----------------------------------------------------------------------------------------#
# handle ccache
# ----------------------------------------------------------------------------------------#

option(TIMEMORY_CCACHE_BUILD "Enable ccache build" OFF)
mark_as_advanced(TIMEMORY_CCACHE_BUILD)
if(TIMEMORY_CCACHE_BUILD)
    find_program(TIMEMORY_CCACHE_EXE ccache PATH_SUFFIXES bin)
    if(TIMEMORY_CCACHE_EXE)
        if(NOT EXISTS "${TIMEMORY_CCACHE_EXE}")
            message(
                WARNING
                    "TIMEMORY_CCACHE_BUILD is ON but TIMEMORY_CCACHE_EXE (${TIMEMORY_CCACHE_EXE}) does not exist!"
                )
        else()
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${TIMEMORY_CCACHE_EXE}")
        endif()
    endif()
endif()

# ----------------------------------------------------------------------------------------#
# options
# ----------------------------------------------------------------------------------------#

# make sure testing enabled
if(TIMEMORY_BUILD_MINIMAL_TESTING)
    set(TIMEMORY_BUILD_TESTING ON)
endif()
# this gets annoying
if(TIMEMORY_BUILD_GOOGLE_TEST OR TIMEMORY_BUILD_TESTING)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS
        ON
        CACHE
            BOOL
            "Suppress Warnings that are meant for the author of the CMakeLists.txt files")
endif()
# override any cache settings
if(TIMEMORY_BUILD_TESTING)
    set(TIMEMORY_BUILD_GOOGLE_TEST ON)
    set(TIMEMORY_BUILD_EXAMPLES ON)
    if(TIMEMORY_BUILD_MINIMAL_TESTING)
        set(TIMEMORY_BUILD_EXAMPLES OFF)
    endif()
endif()
#
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------------------------------------------------------------#
# versioning
# ----------------------------------------------------------------------------------------#

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3" VERSION_STRING
                     "${FULL_VERSION_STRING}")
set(TIMEMORY_VERSION "${VERSION_STRING}")
if(NOT "${TIMEMORY_VERSION}" STREQUAL "${FULL_VERSION_STRING}")
    message(STATUS "[timemory] version ${TIMEMORY_VERSION} (${FULL_VERSION_STRING})")
else()
    message(STATUS "[timemory] version ${TIMEMORY_VERSION}")
endif()
set(TIMEMORY_VERSION_STRING "${FULL_VERSION_STRING}")
string(REPLACE "." ";" VERSION_LIST "${VERSION_STRING}")
list(GET VERSION_LIST 0 TIMEMORY_VERSION_MAJOR)
list(GET VERSION_LIST 1 TIMEMORY_VERSION_MINOR)
list(GET VERSION_LIST 2 TIMEMORY_VERSION_PATCH)
set(TIMEMORY_VERSION
    "${TIMEMORY_VERSION_MAJOR}.${TIMEMORY_VERSION_MINOR}.${TIMEMORY_VERSION_PATCH}")

math(
    EXPR
    TIMEMORY_VERSION_CODE
    "${TIMEMORY_VERSION_MAJOR} * 10000 + ${TIMEMORY_VERSION_MINOR} * 100 + ${TIMEMORY_VERSION_PATCH}"
    )

# ----------------------------------------------------------------------------------------#
# setup.py
# ----------------------------------------------------------------------------------------#

if(SKBUILD AND TIMEMORY_USE_PYTHON)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()

# ----------------------------------------------------------------------------------------#
# Git info
# ----------------------------------------------------------------------------------------#

set(TIMEMORY_GIT_DESCRIBE "unknown")
set(TIMEMORY_GIT_REVISION "unknown")

# the docs/.gitinfo only exists in releases
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs/.gitinfo")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/docs/.gitinfo" _GIT_INFO)
    string(REGEX REPLACE "[\n\r\t ]" ";" _GIT_INFO "${_GIT_INFO}")
    string(REGEX REPLACE ";$" "" _GIT_INFO "${_GIT_INFO}")
    list(LENGTH _GIT_INFO _GIT_INFO_LEN)
    if(_GIT_INFO_LEN GREATER 1)
        list(GET _GIT_INFO 0 TIMEMORY_GIT_REVISION)
        list(GET _GIT_INFO 1 TIMEMORY_GIT_DESCRIBE)
    endif()
endif()

find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE _GIT_DESCRIBE
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_GIT_DESCRIBE)
        set(TIMEMORY_GIT_DESCRIBE "${_GIT_DESCRIBE}")
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE _GIT_REVISION
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_GIT_REVISION)
        set(TIMEMORY_GIT_REVISION "${_GIT_REVISION}")
    endif()
endif()

if(NOT "${TIMEMORY_GIT_REVISION}" STREQUAL "unknown")
    message(STATUS "[timemory] git revision: ${TIMEMORY_GIT_REVISION}")
endif()

if(NOT "${TIMEMORY_GIT_DESCRIBE}" STREQUAL "unknown")
    message(STATUS "[timemory] git describe: ${TIMEMORY_GIT_DESCRIBE}")
endif()
