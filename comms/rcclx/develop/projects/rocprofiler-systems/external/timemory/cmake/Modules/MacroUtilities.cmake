# include guard
include_guard(DIRECTORY)

# MacroUtilities - useful macros and functions for generic tasks
#

cmake_policy(PUSH)
cmake_policy(SET CMP0054 NEW)

include(CMakeDependentOption)
include(CMakeParseArguments)

# Make this file generic by not explicitly defining the name of the project
string(TOUPPER "${PROJECT_NAME}" PROJECT_NAME_UC)
string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LC)

unset(${PROJECT_NAME_UC}_COMPILED_LIBRARIES CACHE)
unset(${PROJECT_NAME_UC}_INTERFACE_LIBRARIES CACHE)

# -----------------------------------------------------------------------
# timemory_set_if_empty: same as set(...) except only performed if the variable is
# undefined or empty
#
macro(TIMEMORY_SET_IF_EMPTY _VARIABLE _VALUE)
    if(NOT DEFINED ${_VARIABLE} OR "${${_VARIABLE}}" STREQUAL "")
        set(${_VARIABLE} "${_VALUE}" ${ARGN})
    endif()
endmacro()

# -----------------------------------------------------------------------
# message which handles TIMEMORY_QUIET_CONFIG settings
# -----------------------------------------------------------------------
#
function(TIMEMORY_MESSAGE TYPE)
    if("${TYPE}" MATCHES "ERROR|WARNING" OR NOT TIMEMORY_QUIET_CONFIG)
        message(${TYPE} "[timemory] ${ARGN}")
    endif()
endfunction()

# -----------------------------------------------------------------------
# Save a set of variables with the given prefix
# -----------------------------------------------------------------------
#
macro(TIMEMORY_SAVE_VARIABLES _PREFIX)
    # parse args
    cmake_parse_arguments(
        SAVE
        "" # options
        "CONDITION" # single value args
        "VARIABLES" # multiple value args
        ${ARGN})
    if(NOT SAVE_VARIABLES)
        message(FATAL_ERROR "Error! timemory_save_variables used incorrectly")
    endif()
    if(DEFINED SAVE_CONDITION AND NOT "${SAVE_CONDITION}" STREQUAL "")
        if(${SAVE_CONDITION})
            foreach(_VAR ${SAVE_VARIABLES})
                if(DEFINED ${_VAR})
                    set(${_PREFIX}_${_VAR} "${${_VAR}}")
                endif()
            endforeach()
        endif()
    else()
        foreach(_VAR ${SAVE_VARIABLES})
            if(DEFINED ${_VAR})
                set(${_PREFIX}_${_VAR} "${${_VAR}}")
            endif()
        endforeach()
    endif()
    unset(SAVE_CONDITION)
    unset(SAVE_VARIABLES)
endmacro()

# -----------------------------------------------------------------------
# Restore a set of variables with the given prefix
# -----------------------------------------------------------------------
#
macro(TIMEMORY_RESTORE_VARIABLES _PREFIX)
    # parse args
    cmake_parse_arguments(
        RESTORE
        "" # options
        "CONDITION" # single value args
        "VARIABLES" # multiple value args
        ${ARGN})
    if(NOT RESTORE_VARIABLES)
        message(FATAL_ERROR "Error! timemory_restore_variables used incorrectly")
    endif()
    if(DEFINED RESTORE_CONDITION AND NOT "${RESTORE_CONDITION}" STREQUAL "")
        if(${RESTORE_CONDITION})
            foreach(_VAR ${RESTORE_VARIABLES})
                if(DEFINED ${_PREFIX}_${_VAR})
                    set(${_VAR} ${${_PREFIX}_${_VAR}})
                    unset(${_PREFIX}_${_VAR})
                endif()
            endforeach()
        endif()
    else()
        foreach(_VAR ${RESTORE_VARIABLES})
            if(DEFINED ${_PREFIX}_${_VAR})
                set(${_VAR} ${${_PREFIX}_${_VAR}})
                unset(${_PREFIX}_${_VAR})
            endif()
        endforeach()
    endif()
    unset(RESTORE_CONDITION)
    unset(RESTORE_VARIABLES)
endmacro()

# -----------------------------------------------------------------------
# CACHED LIST
# -----------------------------------------------------------------------
#
function(TIMEMORY_CACHE_LIST _OP _LIST)
    set(_TMP_CACHE_LIST ${${_LIST}})
    # apply operation on list
    list(${_OP} _TMP_CACHE_LIST ${ARGN})
    # replace list
    set(${_LIST}
        "${_TMP_CACHE_LIST}"
        CACHE INTERNAL "" FORCE)
endfunction()

# -----------------------------------------------------------------------
# function - capitalize - make a string capitalized (first letter is capital) usage:
function(TIMEMORY_CAPITALIZE str var)
    # make string lower
    string(TOLOWER "${str}" str)
    string(SUBSTRING "${str}" 0 1 _first)
    string(TOUPPER "${_first}" _first)
    string(SUBSTRING "${str}" 1 -1 _remainder)
    string(CONCAT str "${_first}" "${_remainder}")
    set(${var}
        "${str}"
        PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------
# function timemory_add_enabled_interface(<NAME>) Mark an interface library as enabled
#
function(TIMEMORY_ADD_ENABLED_INTERFACE _var)
    set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_ENABLED_INTERFACES ${_var})
endfunction()

# -----------------------------------------------------------------------
# function timemory_add_disabled_interface(<NAME>) Mark an interface as disabled
#
function(TIMEMORY_ADD_DISABLED_INTERFACE _var)
    get_property(_DISABLED GLOBAL PROPERTY ${PROJECT_NAME}_DISABLED_INTERFACES)
    if(NOT ${_var} IN_LIST _DISABLED)
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_DISABLED_INTERFACES ${_var})
    endif()
endfunction()

# ------------------------------------------------------------------------------#
# function for creating a library target
#
function(TIMEMORY_CREATE_EXECUTABLE)
    # for include dirs, compile flags, definitions, etc. --> use INTERFACE libs and add
    # them to "LINK_LIBRARIES" list of arguments taking multiple values
    set(multival_args HEADERS SOURCES PROPERTIES LINK_LIBRARIES INSTALL_DESTINATION)

    # parse args
    cmake_parse_arguments(
        EXE
        "INSTALL;EXCLUDE_FROM_ALL" # options
        "TARGET_NAME;" # single value args
        "${multival_args}" # multiple value args
        ${ARGN})

    set(_EXCLUDE)
    if(EXE_EXCLUDE_FROM_ALL OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
        set(_EXCLUDE EXCLUDE_FROM_ALL)
    endif()
    # create library
    add_executable(${EXE_TARGET_NAME} ${_EXCLUDE} ${EXE_SOURCES} ${EXE_HEADERS})

    # link library
    target_link_libraries(${EXE_TARGET_NAME} ${EXE_LINK_LIBRARIES})

    # target properties
    if(NOT "${EXE_PROPERTIES}" STREQUAL "")
        set_target_properties(${EXE_TARGET_NAME} PROPERTIES ${EXE_PROPERTIES})
    endif()

    if(EXE_INSTALL AND NOT EXE_INSTALL_DESTINATION)
        set(EXE_INSTALL_DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    # Install the exe
    if(EXE_INSTALL_DESTINATION)
        install(
            TARGETS ${EXE_TARGET_NAME}
            DESTINATION ${EXE_INSTALL_DESTINATION}
            OPTIONAL)
    endif()
endfunction()

# ------------------------------------------------------------------------------#
# function timemory_add_timemory_test_target()
#
# Creates a target which runs ctest but depends on all the tests being built.
#
function(TIMEMORY_ADD_TIMEMORY_TEST_TARGET)
    if(NOT TARGET timemory-test)
        add_custom_target(
            timemory-test
            COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target test
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            COMMENT "Running tests...")
    endif()
endfunction()

# ------------------------------------------------------------------------------#
# function timemory_add_google_test()
#
# Adds a unit test and links against googletest. Additional arguments are linked against
# the test.
#
function(TIMEMORY_ADD_GOOGLE_TEST TEST_NAME)
    if(NOT TIMEMORY_BUILD_GOOGLE_TEST)
        return()
    endif()
    set(_OPTS)
    if(NOT TIMEMORY_BUILD_TESTING)
        set(_OPTS EXCLUDE_FROM_ALL)
    endif()
    timemory_add_timemory_test_target()

    include(GoogleTest)

    # list of arguments taking zero values
    set(option_args DISCOVER_TESTS ADD_TESTS MPI RUN_SERIAL)
    # list of arguments taking single value
    set(single_args NPROCS TIMEOUT TARGET)
    # list of arguments taking multiple values
    set(multiv_args
        SOURCES
        DEPENDS
        PROPERTIES
        DEFINITIONS
        LINK_LIBRARIES
        COMMAND
        OPTIONS
        ENVIRONMENT
        LABELS)
    # parse args
    cmake_parse_arguments(TEST "${option_args}" "${single_args}" "${multiv_args}" ${ARGN})

    if(NOT TARGET google-test-debug-options)
        add_library(google-test-debug-options INTERFACE)
        target_compile_definitions(google-test-debug-options
                                   INTERFACE $<$<CONFIG:Debug>:DEBUG> TIMEMORY_TESTING)
    endif()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR TIMEMORY_USE_COVERAGE)
        # create a pre-processor which relaxes pass/fail when converage is enabled
        target_compile_definitions(google-test-debug-options
                                   INTERFACE TIMEMORY_RELAXED_TESTING)
    endif()
    list(APPEND TEST_LINK_LIBRARIES google-test-debug-options)

    if(TEST_RUN_SERIAL)
        list(APPEND TEST_PROPERTIES RUN_SERIAL ON)
    endif()

    if(NOT TEST_TARGET)
        set(TEST_TARGET ${TEST_NAME})
    endif()

    if(TEST_SOURCES AND NOT TARGET ${TEST_TARGET})
        timemory_create_executable(
            ${_OPTS}
            TARGET_NAME ${TEST_TARGET} OUTPUT_NAME ${TEST_TARGET}
            SOURCES ${TEST_SOURCES}
            LINK_LIBRARIES timemory-google-test ${TEST_LINK_LIBRARIES}
            PROPERTIES "${TEST_PROPERTIES}")

        target_compile_definitions(${TEST_TARGET} PUBLIC ${TEST_DEFINITIONS})

        # always add as a dependency if target is built
        add_dependencies(timemory-test ${TEST_TARGET})

        if(WIN32)
            set_target_properties(
                ${TEST_TARGET}
                PROPERTIES FOLDER tests
                           RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/outputs/runtime
                           LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/outputs/runtime
                           ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/outputs/archive
                           PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/outputs/runtime)
        endif()
    endif()

    set(TEST_LAUNCHER)
    if(TIMEMORY_USE_MPI
       AND TEST_MPI
       AND MPIEXEC_EXECUTABLE)
        if(NOT TEST_NPROCS)
            set(TEST_NPROCS 2)
        endif()
        set(TEST_LAUNCHER ${MPIEXEC_EXECUTABLE} -n ${TEST_NPROCS})
        list(APPEND TEST_LABELS "MPI")
        if(NOT TEST_RUN_SERIAL)
            list(APPEND TEST_PROPERTIES RUN_SERIAL ON)
        endif()
    endif()

    if("${TEST_COMMAND}" STREQUAL "")
        set(TEST_COMMAND ${TEST_LAUNCHER} $<TARGET_FILE:${TEST_TARGET}>)
    elseif(TEST_LAUNCHER)
        set(TEST_COMMAND ${TEST_LAUNCHER} ${TEST_COMMAND})
    endif()

    if(NOT DEFINED CTEST_TEST_TIMEOUT OR "${CTEST_TEST_TIMEOUT}" STREQUAL "")
        set(CTEST_TEST_TIMEOUT 180)
    endif()

    if(NOT TEST_TIMEOUT)
        set(TEST_TIMEOUT ${CTEST_TEST_TIMEOUT})
    endif()

    set(WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR})
    if(WIN32)
        set(WORKING_DIR $<TARGET_FILE_DIR:${TEST_TARGET}>)
    endif()

    if(NOT TEST_ENVIRONMENT MATCHES "TIMEMORY_OUTPUT_PREFIX")
        list(INSERT TEST_ENVIRONMENT 0 "TIMEMORY_OUTPUT_PREFIX=%m-")
    endif()

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR TIMEMORY_USE_COVERAGE)
        list(INSERT TEST_ENVIRONMENT 0 "TIMEMORY_VERBOSE=6")
    endif()

    list(APPEND TEST_LABELS "minimal")

    if(TEST_DISCOVER_TESTS)
        string(REPLACE ";" "\\\\\\\\\\\\\\\\\\;" TEST_ENVIRONMENT "${TEST_ENVIRONMENT}")
        gtest_discover_tests(
            ${TEST_TARGET}
            TEST_LIST ${TEST_NAME}_TESTS ${TEST_OPTIONS} DISCOVERY_TIMEOUT 15
            WORKING_DIRECTORY ${WORKING_DIR}
            PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}"
            TIMEOUT ${TEST_TIMEOUT} LABELS "${TEST_LABELS}" ${TEST_PROPERTIES})
    elseif(TEST_ADD_TESTS)
        gtest_add_tests(
            TARGET ${TEST_TARGET}
            TEST_LIST ${TEST_NAME}_TESTS ${TEST_OPTIONS}
            WORKING_DIRECTORY ${WORKING_DIR})
        set_tests_properties(
            ${${TEST_NAME}_TESTS}
            PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}" TIMEOUT ${TEST_TIMEOUT} LABELS
                       "${TEST_LABELS}" ${TEST_PROPERTIES})
    else()
        add_test(
            NAME ${TEST_NAME}
            COMMAND ${TEST_COMMAND}
            WORKING_DIRECTORY ${WORKING_DIR} ${TEST_OPTIONS})
        set_tests_properties(
            ${TEST_NAME}
            PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}" TIMEOUT ${TEST_TIMEOUT} LABELS
                       "${TEST_LABELS}" ${TEST_PROPERTIES})
    endif()

    if(TEST_DEPENDS)
        set_property(
            TEST ${TEST_NAME}
            APPEND
            PROPERTY DEPENDS ${TEST_DEPENDS})
    endif()

endfunction()

# ----------------------------------------------------------------------------------------#
# function TIMEMORY_CHECKOUT_GIT_SUBMODULE()
#
# Run "git submodule update" if a file in a submodule does not exist
#
# ARGS: RECURSIVE (option) -- add "--recursive" flag RELATIVE_PATH (one value) --
# typically the relative path to submodule from PROJECT_SOURCE_DIR WORKING_DIRECTORY (one
# value) -- (default: PROJECT_SOURCE_DIR) TEST_FILE (one value) -- file to check for
# (default: CMakeLists.txt) ADDITIONAL_CMDS (many value) -- any addition commands to pass
#
function(TIMEMORY_CHECKOUT_GIT_SUBMODULE)
    # parse args
    cmake_parse_arguments(
        CHECKOUT "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE;REPO_URL;REPO_BRANCH"
        "ADDITIONAL_CMDS" ${ARGN})

    if(NOT CHECKOUT_WORKING_DIRECTORY)
        set(CHECKOUT_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()

    if(NOT CHECKOUT_TEST_FILE)
        set(CHECKOUT_TEST_FILE "CMakeLists.txt")
    endif()

    # default assumption
    if(NOT CHECKOUT_REPO_BRANCH)
        set(CHECKOUT_REPO_BRANCH "master")
    endif()

    find_package(Git)
    set(_DIR "${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}")
    # ensure the (possibly empty) directory exists
    if(NOT EXISTS "${_DIR}")
        if(NOT CHECKOUT_REPO_URL)
            message(FATAL_ERROR "submodule directory does not exist")
        endif()
    endif()

    # if this file exists --> project has been checked out if not exists --> not been
    # checked out
    set(_TEST_FILE "${_DIR}/${CHECKOUT_TEST_FILE}")
    # assuming a .gitmodules file exists
    set(_SUBMODULE "${PROJECT_SOURCE_DIR}/.gitmodules")

    set(_TEST_FILE_EXISTS OFF)
    if(EXISTS "${_TEST_FILE}" AND NOT IS_DIRECTORY "${_TEST_FILE}")
        set(_TEST_FILE_EXISTS ON)
    endif()

    if(_TEST_FILE_EXISTS)
        return()
    endif()

    find_package(Git REQUIRED)

    set(_SUBMODULE_EXISTS OFF)
    if(EXISTS "${_SUBMODULE}" AND NOT IS_DIRECTORY "${_SUBMODULE}")
        set(_SUBMODULE_EXISTS ON)
    endif()

    set(_HAS_REPO_URL OFF)
    if(NOT "${CHECKOUT_REPO_URL}" STREQUAL "")
        set(_HAS_REPO_URL ON)
    endif()

    # if the module has not been checked out
    if(NOT _TEST_FILE_EXISTS AND _SUBMODULE_EXISTS)
        # perform the checkout
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                    ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # check the return code
        if(RET GREATER 0)
            set(_CMD "${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}")
            message(STATUS "function(TIMEMORY_CHECKOUT_GIT_SUBMODULE) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT _TEST_FILE_EXISTS AND _HAS_REPO_URL)
        message(
            STATUS "Checking out '${CHECKOUT_REPO_URL}' @ '${CHECKOUT_REPO_BRANCH}'...")

        # remove the existing directory
        if(EXISTS "${_DIR}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${_DIR})
        endif()

        # perform the checkout
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # perform the submodule update
        if(CHECKOUT_RECURSIVE
           AND EXISTS "${_DIR}"
           AND IS_DIRECTORY "${_DIR}")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                WORKING_DIRECTORY ${_DIR}
                RESULT_VARIABLE RET)
        endif()

        # check the return code
        if(RET GREATER 0)
            set(_CMD
                "${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}"
                )
            message(STATUS "function(TIMEMORY_CHECKOUT_GIT_SUBMODULE) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT EXISTS "${_TEST_FILE}" OR NOT _TEST_FILE_EXISTS)
        message(
            FATAL_ERROR
                "Error checking out submodule: '${CHECKOUT_RELATIVE_PATH}' to '${_DIR}'")
    endif()

endfunction()

# ----------------------------------------------------------------------------------------#
# try to find a package quietly
#
function(TIMEMORY_TEST_FIND_PACKAGE PACKAGE_NAME OUTPUT_VAR)
    cmake_parse_arguments(PACKAGE "" "" "UNSET" ${ARGN})
    find_package(${PACKAGE_NAME} QUIET ${PACKAGE_UNPARSED_ARGUMENTS})
    if(NOT ${PACKAGE_NAME}_FOUND)
        set(${OUTPUT_VAR}
            OFF
            PARENT_SCOPE)
    else()
        set(${OUTPUT_VAR}
            ON
            PARENT_SCOPE)
    endif()
    foreach(_ARG ${PACKAGE_UNSET} FIND_PACKAGE_MESSAGE_DETAILS_${PACKAGE_NAME})
        unset(${_ARG} CACHE)
    endforeach()
endfunction()

# ----------------------------------------------------------------------------------------#
# macro to add an interface lib
#
macro(TIMEMORY_ADD_INTERFACE_LIBRARY _TARGET)
    add_library(${_TARGET} INTERFACE)
    add_library(${PROJECT_NAME}::${_TARGET} ALIAS ${_TARGET})
    timemory_cache_list(APPEND ${PROJECT_NAME_UC}_INTERFACE_LIBRARIES ${_TARGET})
    install(
        TARGETS ${_TARGET}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT ${PROJECT_NAME}-library-depends
        OPTIONAL)
    timemory_add_enabled_interface(${_TARGET})
    if(NOT "${ARGN}" STREQUAL "")
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_INTERFACE_DOC
                                            "${PROJECT_NAME}::${_TARGET}` | ${ARGN} |")
    endif()
endmacro()

# ----------------------------------------------------------------------------------------#
#
# handle empty interface
#
# ----------------------------------------------------------------------------------------#

function(TIMEMORY_INFORM_EMPTY_INTERFACE _TARGET _PACKAGE)
    if(NOT TARGET ${_TARGET})
        timemory_message(
            AUTHOR_WARNING
            "A non-existant target was passed to INFORM_EMPTY_INTERFACE: ${_TARGET}")
    endif()
    if(NOT ${_TARGET} IN_LIST TIMEMORY_EMPTY_INTERFACE_LIBRARIES)
        timemory_message(
            STATUS
            "[interface] ${_PACKAGE} not found/enabled. '${_TARGET}' interface will not provide ${_PACKAGE}..."
            )
        set(TIMEMORY_EMPTY_INTERFACE_LIBRARIES
            ${TIMEMORY_EMPTY_INTERFACE_LIBRARIES} ${_TARGET}
            PARENT_SCOPE)
    endif()
    timemory_add_disabled_interface(${_TARGET})
endfunction()

# ----------------------------------------------------------------------------------------#
#
# Add to RPATH
#
# ----------------------------------------------------------------------------------------#

function(TIMEMORY_ADD_RPATH)
    if(NOT CMAKE_INSTALL_RPATH_USE_LINK_PATH)
        return()
    endif()
    set(_DIRS)
    foreach(_ARG ${ARGN})
        if(EXISTS "${_ARG}" AND IS_DIRECTORY "${_ARG}")
            list(APPEND _DIRS "${_ARG}")
        else()
            get_filename_component(_DIR "${_ARG}" DIRECTORY)
            if(EXISTS "${_DIR}" AND IS_DIRECTORY "${_DIR}")
                list(APPEND _DIRS "${_DIR}")
            endif()
        endif()
    endforeach()
    if(_DIRS)
        list(REMOVE_DUPLICATES _DIRS)
        set(_RPATH_DIRS)
        foreach(_DIR ${_DIRS})
            string(REPLACE "${CMAKE_BINARY_DIR}" "build_directory" _DIR "${_DIR}")
            list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${_DIR}" _IS_SYSTEM_DIR)
            if(NOT "${_DIR}" MATCHES "^build_directory"
               AND EXISTS "${_DIR}"
               AND IS_DIRECTORY "${_DIR}"
               AND NOT _IS_SYSTEM_DIR)
                list(APPEND _RPATH_DIRS ${_DIR})
            endif()
        endforeach()
        if(_RPATH_DIRS)
            string(REPLACE ";" ":" _RPATH_DIRS "${_RPATH_DIRS}")
            set(CMAKE_INSTALL_RPATH
                "${CMAKE_INSTALL_RPATH}:${_RPATH_DIRS}"
                PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# macro to build a library of type: shared, static, object
#
function(TIMEMORY_BUILD_LIBRARY)

    # options
    set(_options PIC NO_CACHE_LIST EXCLUDE_FROM_ALL)
    # single-value
    set(_onevalue TYPE OUTPUT_NAME TARGET_NAME OUTPUT_DIR LANGUAGE LINKER_LANGUAGE)
    # multi-value
    set(_multival
        SOURCES
        LINK_LIBRARIES
        COMPILE_DEFINITIONS
        INCLUDE_DIRECTORIES
        C_COMPILE_OPTIONS
        CXX_COMPILE_OPTIONS
        CUDA_COMPILE_OPTIONS
        LINK_OPTIONS
        EXTRA_PROPERTIES)

    cmake_parse_arguments(LIBRARY "${_options}" "${_onevalue}" "${_multival}" ${ARGN})

    if("${LIBRARY_LANGUAGE}" STREQUAL "")
        set(LIBRARY_LANGUAGE CXX)
    endif()

    if("${LIBRARY_LINKER_LANGUAGE}" STREQUAL "")
        set(LIBRARY_LINKER_LANGUAGE CXX)
    endif()

    if("${LIBRARY_OUTPUT_DIR}" STREQUAL "")
        set(LIBRARY_OUTPUT_DIR ${PROJECT_BINARY_DIR})
    endif()

    set(_EXCLUDE)
    if(LIBRARY_EXCLUDE_FROM_ALL OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
        set(_EXCLUDE EXCLUDE_FROM_ALL)
    endif()

    # handle PIC not specified but global PIC specified
    if(NOT LIBRARY_PIC AND CMAKE_POSITION_INDEPENDENT_CODE)
        set(LIBRARY_PIC ON)
    endif()

    if(NOT "${LIBRARY_TYPE}" STREQUAL "OBJECT")
        if(NOT WIN32 AND NOT XCODE)
            list(APPEND LIBRARY_EXTRA_PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION
                 ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
        endif()

        if(NOT WIN32)
            set(LIB_PREFIX)
            list(APPEND LIBRARY_EXTRA_PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                 ${LIBRARY_OUTPUT_DIR} ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIR}
                 RUNTIME_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIR})
        else()
            set(LIB_PREFIX lib)
        endif()
    endif()

    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    set(_CUDA OFF)
    if(CMAKE_CUDA_COMPILER AND "CUDA" IN_LIST LANGUAGES)
        set(_CUDA ON)
    endif()

    set(_ENABLE_CUSTOM_COMMAND OFF)

    # add the library or sources
    if(NOT TARGET ${LIBRARY_TARGET_NAME})
        set(_ENABLE_CUSTOM_COMMAND ON)
        add_library(${LIBRARY_TARGET_NAME} ${LIBRARY_TYPE} ${_EXCLUDE} ${LIBRARY_SOURCES})
        add_library(${PROJECT_NAME}::${LIBRARY_TARGET_NAME} ALIAS ${LIBRARY_TARGET_NAME})
        if(TARGET libunwind-external)
            add_dependencies(${LIBRARY_TARGET_NAME} libunwind-external)
        endif()
    else()
        target_sources(${LIBRARY_TARGET_NAME} PRIVATE ${LIBRARY_SOURCES})
    endif()

    # append include directories
    target_include_directories(${LIBRARY_TARGET_NAME}
                               PUBLIC ${LIBRARY_INCLUDE_DIRECTORIES})

    # compile definitions
    timemory_target_compile_definitions(${LIBRARY_TARGET_NAME} PUBLIC
                                        ${LIBRARY_COMPILE_DEFINITIONS})

    # compile flags
    target_compile_options(
        ${LIBRARY_TARGET_NAME}
        PRIVATE $<$<COMPILE_LANGUAGE:C>:${LIBRARY_C_COMPILE_OPTIONS}>
                $<$<COMPILE_LANGUAGE:CXX>:${LIBRARY_CXX_COMPILE_OPTIONS}>)

    # windows docs.microsoft.com/en-us/cpp/build/reference/md-mt-ld-use-run-time-library
    # if(MSVC AND NOT "${LIBRARY_TYPE}" STREQUAL "OBJECT") set(_MSVC_FLAGS /MD)
    # if("${LIBRARY_TYPE}" STREQUAL "STATIC") set(_MSVC_FLAGS /MT) endif()
    # target_compile_options(${LIBRARY_TARGET_NAME} PUBLIC
    # $<$<COMPILE_LANGUAGE:C>:${_MSVC_FLAGS}>
    # $<$<COMPILE_LANGUAGE:CXX>:${_MSVC_FLAGS}>)
    # endif()

    # cuda flags
    if(_CUDA)
        target_compile_options(
            ${LIBRARY_TARGET_NAME}
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${LIBRARY_CUDA_COMPILE_OPTIONS}>)
    endif()

    # link libraries
    target_link_libraries(
        ${LIBRARY_TARGET_NAME}
        PUBLIC ${LIBRARY_LINK_LIBRARIES}
        PRIVATE ${_ANALYSIS_TOOLS} ${_ARCH_LIBRARY})

    # other properties
    if(NOT "${LIBRARY_TYPE}" STREQUAL "OBJECT")
        # link options
        if(NOT CMAKE_VERSION VERSION_LESS 3.13)
            target_link_options(${LIBRARY_TARGET_NAME} PUBLIC ${LIBRARY_LINK_OPTIONS})
        elseif(NOT "${LIBRARY_LINK_OPTIONS}" STREQUAL "")
            list(APPEND LIBRARY_EXTRA_PROPERTIES LINK_OPTIONS ${LIBRARY_LINK_OPTIONS})
        endif()
        #
        set_target_properties(
            ${LIBRARY_TARGET_NAME}
            PROPERTIES OUTPUT_NAME ${LIB_PREFIX}${LIBRARY_OUTPUT_NAME}
                       LANGUAGE ${LIBRARY_LANGUAGE}
                       LINKER_LANGUAGE ${LIBRARY_LINKER_LANGUAGE}
                       POSITION_INDEPENDENT_CODE ${LIBRARY_PIC}
                       ${LIBRARY_EXTRA_PROPERTIES})
    else()
        set_target_properties(
            ${LIBRARY_TARGET_NAME}
            PROPERTIES LANGUAGE ${LIBRARY_LANGUAGE}
                       POSITION_INDEPENDENT_CODE ${LIBRARY_PIC}
                       ${LIBRARY_EXTRA_PROPERTIES})
    endif()

    set(COMPILED_TYPES "SHARED" "STATIC" "MODULE")
    if(NOT LIBRARY_NO_CACHE_LIST)
        # add to cached list of compiled libraries
        if("${LIBRARY_TYPE}" IN_LIST COMPILED_TYPES)
            timemory_cache_list(APPEND ${PROJECT_NAME_UC}_COMPILED_LIBRARIES
                                ${LIBRARY_TARGET_NAME})
        endif()
    endif()
    unset(COMPILED_TYPES)

    set_property(GLOBAL APPEND PROPERTY TIMEMORY_${LIBRARY_TYPE}_LIBRARIES
                                        ${LIBRARY_TARGET_NAME})

    if("${LIBRARY_TYPE}" STREQUAL "SHARED" AND _ENABLE_CUSTOM_COMMAND)
        set(_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
        set(_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})

        get_target_property(_OUTNAME ${LIBRARY_TARGET_NAME} OUTPUT_NAME)
        get_target_property(_VERSION ${LIBRARY_TARGET_NAME} VERSION)
        get_target_property(_SOVERSION ${LIBRARY_TARGET_NAME} SOVERSION)

        set(_FILENAME ${_PREFIX}${_OUTNAME}${_SUFFIX})

        get_target_property(_OUTDIR ${LIBRARY_TARGET_NAME} LIBRARY_OUTPUT_DIRECTORY)
        if(NOT IS_ABSOLUTE "${_OUTDIR}")
            set(_OUTDIR "${PROJECT_BINARY_DIR}/${_OUTDIR}")
        endif()
        file(RELATIVE_PATH BINARY_RELPATH "${PROJECT_BINARY_DIR}/timemory/libs"
             "${_OUTDIR}")

        string(REGEX REPLACE "/$" "" BINARY_RELPATH "${BINARY_RELPATH}")

        # build tree
        if(WIN32)
            # copy if on windows
            add_custom_command(
                TARGET ${LIBRARY_TARGET_NAME}
                POST_BUILD
                COMMAND
                    ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:${LIBRARY_TARGET_NAME}>
                    ${PROJECT_BINARY_DIR}/timemory/libs
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        else()
            # symlink if not windows
            add_custom_command(
                TARGET ${LIBRARY_TARGET_NAME}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E create_symlink ${BINARY_RELPATH}/${_FILENAME}
                        ${PROJECT_BINARY_DIR}/timemory/libs/${_FILENAME}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()

    endif()

endfunction()

# ----------------------------------------------------------------------------------------#
# finds dependencies
#
function(TIMEMORY_GET_INTERNAL_DEPENDS VAR LINK)
    # set the depends before creating the library so it does not link to itself
    set(DEPENDS)
    foreach(DEP ${ARGN})
        #
        if(TARGET ${DEP}-object)
            list(APPEND DEPENDS $<TARGET_OBJECTS:${DEP}-object>)
        elseif(TARGET ${DEP}-${LINK})
            list(APPEND DEPENDS ${DEP}-${LINK})
        elseif(TARGET ${DEP})
            list(APPEND DEPENDS ${DEP})
        endif()
        #
        if(TARGET ${DEP}-component-object)
            list(APPEND DEPENDS $<TARGET_OBJECTS:${DEP}-component-object>)
        elseif(TARGET ${DEP}-component-${LINK})
            list(APPEND DEPENDS ${DEP}-component-${LINK})
        endif()
        #
        if(TARGET timemory-${DEP}-object)
            list(APPEND DEPENDS $<TARGET_OBJECTS:timemory-${DEP}-object>)
        elseif(TARGET timemory-${DEP}-${LINK})
            list(APPEND DEPENDS timemory-${DEP}-${LINK})
        endif()
        #
        if(TARGET timemory-${DEP}-component-object)
            list(APPEND DEPENDS $<TARGET_OBJECTS:timemory-${DEP}-component-object>)
        elseif(TARGET timemory-${DEP}-component-${LINK})
            list(APPEND DEPENDS timemory-${DEP}-component-${LINK})
        endif()
    endforeach()
    set(${VAR}
        "${DEPENDS}"
        PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------------------#
# finds dependencies
#
function(TIMEMORY_GET_PROPERTY_DEPENDS VAR LINK)
    # set the depends before creating the library so it does not link to itself
    set(DEPENDS)
    foreach(DEP ${ARGN})
        get_property(TMP GLOBAL PROPERTY TIMEMORY_${LINK}_${DEP}_LIBRARIES)
        # message(STATUS "TIMEMORY_${LINK}_${DEP}_LIBRARIES :: ${TMP}")
        foreach(_ENTRY ${TMP})
            if(NOT TARGET ${_ENTRY})
                continue()
            endif()
            if("${LINK}" STREQUAL "OBJECT")
                list(APPEND DEPENDS $<TARGET_OBJECTS:${_ENTRY}>)
            else()
                list(APPEND DEPENDS ${_ENTRY})
            endif()
        endforeach()
    endforeach()
    set(${VAR}
        "${DEPENDS}"
        PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------------------#
# require variable
#
function(TIMEMORY_CHECK_REQUIRED VAR)
    if(NOT DEFINED ${VAR} OR "${${VAR}}" STREQUAL "")
        message(FATAL_ERROR "Variable '${VAR}' must be defined and not empty")
    endif()
endfunction()

# -----------------------------------------------------------------------
# C/C++ development headers
#
function(TIMEMORY_INSTALL_HEADER_FILES)
    if(NOT TIMEMORY_INSTALL_HEADERS)
        return()
    endif()
    foreach(_header ${ARGN})
        if("${_header}" MATCHES "TARGET_OBJECTS")
            continue()
        endif()
        if(NOT EXISTS "${_header}")
            timemory_message(
                AUTHOR_WARNING
                "Skipping install of non-existant timemory header: '${_header}'")
        endif()
        file(RELATIVE_PATH _relative ${PROJECT_SOURCE_DIR}/source ${_header})
        get_filename_component(_destpath ${_relative} DIRECTORY)
        install(
            FILES ${_header}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${_destpath}
            OPTIONAL)
    endforeach()
endfunction()

# -----------------------------------------------------------------------
# Library installation
#
function(TIMEMORY_INSTALL_LIBRARIES)
    cmake_parse_arguments(LIB "LINK_VERSION;LINK_SOVERSION" "DESTINATION;EXPORT"
                          "TARGETS" ${ARGN})

    if(NOT LIB_DESTINATION)
        set(LIB_DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()

    if(NOT LIB_EXPORT)
        set(LIB_EXPORT ${PROJECT_NAME}-library-depends)
    endif()

    set(_ECHO)
    if(NOT CMAKE_VERSION VERSION_LESS 3.15)
        set(_ECHO COMMAND_ECHO STDOUT)
    endif()

    get_property(SHARED_LIBS GLOBAL PROPERTY TIMEMORY_SHARED_LIBRARIES)

    if(TIMEMORY_USE_PYTHON)
        set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}/libs)
        if(NOT IS_ABSOLUTE "${_PYLIB}")
            set(_PYLIB ${CMAKE_INSTALL_PREFIX}/${_PYLIB})
        endif()
    endif()

    list(REMOVE_DUPLICATES LIB_TARGETS)

    if(WIN32)
        # use the defaults for destination folders on windows; this follows conventional
        # windows locations by putting DLLs into bin and LIBs into lib
        set(_dst)
    else()
        # set destination to be the one specified by input argument
        set(_dst DESTINATION ${LIB_DESTINATION})
    endif()
    foreach(_LIB ${LIB_TARGETS})
        # get the list of previously exported libraries
        get_property(_PREVIOUS_EXPORTS GLOBAL PROPERTY TIMEMORY_EXPORTED_LIBRARIES)

        # skip exporting again if already exported
        if("${_LIB}" IN_LIST _PREVIOUS_EXPORTS)
            continue()
        endif()

        # add to list of export targets
        set_property(GLOBAL APPEND PROPERTY TIMEMORY_EXPORTED_LIBRARIES ${_LIB})

        get_target_property(_LIB_TYPE ${_LIB} TYPE)
        if(NOT "${_LIB_TYPE}" STREQUAL "OBJECT_LIBRARY"
           AND (TIMEMORY_INSTALL_LIBRARIES OR "${_LIB_TYPE}" MATCHES
                                              "(MODULE|SHARED|INTERFACE)_LIBRARY"))
            install(
                TARGETS ${_LIB} ${_dst}
                EXPORT ${PROJECT_NAME}-library-depends
                OPTIONAL)

            if(WIN32 AND "${_LIB}" IN_LIST SHARED_LIBS)
                # for windows install pdb files too
                install(
                    FILES $<TARGET_PDB_FILE:${_LIB}>
                    DESTINATION ${CMAKE_INSTALL_BINDIR}
                    OPTIONAL)
            endif()
        endif()

        if(TIMEMORY_USE_PYTHON AND ${_LIB} IN_LIST SHARED_LIBS)
            set(_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
            set(_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})

            get_target_property(_OUTNAME ${_LIB} OUTPUT_NAME)
            get_target_property(_VERSION ${_LIB} VERSION)
            get_target_property(_SOVERSION ${_LIB} SOVERSION)

            set(_FILENAME ${_PREFIX}${_OUTNAME}${_SUFFIX})
            set(_VERSION_FILENAME)
            set(_SOVERSION_FILENAME)
            set(_FILE_TYPES _FILENAME)
            #
            if(_VERSION AND LIB_LINK_VERSION)
                if(APPLE)
                    set(_VERSION_FILENAME ${_PREFIX}${_OUTNAME}.${_VERSION}${_SUFFIX})
                else()
                    set(_VERSION_FILENAME ${_PREFIX}${_OUTNAME}${_SUFFIX}.${_VERSION})
                endif()
                list(APPEND _FILE_TYPES _VERSION_FILENAME)
            endif()
            #
            if(_SOVERSION AND LIB_LINK_SOVERSION)
                if(APPLE)
                    set(_SOVERSION_FILENAME ${_PREFIX}${_OUTNAME}.${_SOVERSION}${_SUFFIX})
                else()
                    set(_SOVERSION_FILENAME ${_PREFIX}${_OUTNAME}${_SUFFIX}.${_SOVERSION})
                endif()
                list(APPEND _FILE_TYPES _SOVERSION_FILENAME)
            endif()

            file(RELATIVE_PATH INSTALL_RELPATH "${_PYLIB}"
                 "${CMAKE_INSTALL_PREFIX}/${LIB_DESTINATION}")

            get_target_property(_OUTDIR ${_LIB} LIBRARY_OUTPUT_DIRECTORY)
            if(NOT IS_ABSOLUTE "${_OUTDIR}")
                set(_OUTDIR "${PROJECT_BINARY_DIR}/${_OUTDIR}")
            endif()

            string(REGEX REPLACE "/$" "" INSTALL_RELPATH "${INSTALL_RELPATH}")

            # install tree
            foreach(_FNAME ${_FILE_TYPES})
                if(NOT ${_FNAME})
                    continue()
                endif()
                if(WIN32)
                    # install a second time if windows
                    install(TARGETS ${_LIB}
                            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}/libs)
                else()
                    # symlink if not windows
                    install(
                        CODE "
                        EXECUTE_PROCESS(
                            COMMAND ${CMAKE_COMMAND} -E create_symlink
                            ${INSTALL_RELPATH}/${${_FNAME}} ${_PYLIB}/${${_FNAME}}
                            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                            ${_ECHO})
                        "
                        OPTIONAL)
                endif()
            endforeach()
        endif()
    endforeach()
endfunction()

# -----------------------------------------------------------------------
# Add pre-compiled headers
#
function(TIMEMORY_TARGET_PRECOMPILE_HEADERS _TARG)
    cmake_parse_arguments(HEAD "INSTALL_INTERFACE" "" "FILES" ${ARGN})

    if(TIMEMORY_PRECOMPILE_HEADERS)
        set(_BINARY_IFACE)
        set(_INSTALL_IFACE)
        foreach(_HEADER ${HEAD_FILES})
            string(REPLACE "${PROJECT_SOURCE_DIR}/" "" _HEADER "${_HEADER}")
            list(APPEND _BINARY_IFACE "${_HEADER}")
            string(REPLACE "external/cereal/include/" "" _HEADER "${_HEADER}")
            string(REPLACE "source/" "" _HEADER "${_HEADER}")
            list(APPEND _INSTALL_IFACE "<${_HEADER}>")
        endforeach()
        target_precompile_headers(${_TARG} INTERFACE $<BUILD_INTERFACE:${_BINARY_IFACE}>)
        if(HEAD_INSTALL_INTERFACE)
            target_precompile_headers(${_TARG} INTERFACE
                                      $<BUILD_INTERFACE:${_PRECOMPILE_HEADERS}>)
        endif()
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function to build a library of type: shared, static, object
#
function(TIMEMORY_BUILD_INTERMEDIATE_LIBRARY)

    # options
    set(_options USE_INTERFACE USE_CATEGORY INSTALL_SOURCE FORCE_OBJECT FORCE_SHARED
                 FORCE_STATIC)
    # single-value
    set(_onevalue NAME TARGET CATEGORY FOLDER VISIBILITY)
    # multi-value
    set(_multival HEADERS SOURCES DEPENDS INCLUDES PROPERTY_DEPENDS PUBLIC_LINK
                  PRIVATE_LINK)

    cmake_parse_arguments(COMP "${_options}" "${_onevalue}" "${_multival}" ${ARGN})

    timemory_check_required(COMP_NAME)
    timemory_check_required(COMP_TARGET)
    timemory_check_required(COMP_CATEGORY)
    timemory_check_required(COMP_FOLDER)

    if(NOT COMP_VISIBILITY)
        if(TIMEMORY_BUILD_HIDDEN_VISIBILITY)
            set(COMP_VISIBILITY hidden)
        else()
            set(COMP_VISIBILITY default)
        endif()
    endif()

    set(VIS_OPTS "default" "hidden" "protected" "internal")
    if(NOT "${COMP_VISIBILITY}" IN_LIST VIS_OPTS)
        message(FATAL_ERROR "${COMP_TARGET} available visibility options: ${VIS_OPTS}")
    endif()

    if(TIMEMORY_USE_VISIBILITY)
        set(CMAKE_C_VISIBILITY_PRESET "${COMP_VISIBILITY}")
        set(CMAKE_CXX_VISIBILITY_PRESET "${COMP_VISIBILITY}")
    endif()

    string(TOUPPER "${COMP_NAME}" UPP_COMP)
    string(REPLACE "-" "_" UPP_COMP "${UPP_COMP}")
    string(TOLOWER "${COMP_CATEGORY}" LC_CATEGORY)

    set(_LIB_TYPES)
    set(_LIB_DEFAULT_TYPE)

    # if(TIMEMORY_BUILD_LTO OR COMP_FORCE_OBJECT)
    if(COMP_FORCE_OBJECT)
        list(APPEND _LIB_TYPES object)
        set(object_OPTIONS PIC TYPE OBJECT)
    endif()

    if(_BUILD_STATIC_CXX OR COMP_FORCE_STATIC)
        list(APPEND _LIB_TYPES static)
        set(static_OPTIONS TYPE STATIC)
        set(_LIB_DEFAULT_TYPE static)
    endif()

    if(_BUILD_SHARED_CXX OR COMP_FORCE_SHARED)
        list(APPEND _LIB_TYPES shared)
        set(shared_OPTIONS PIC TYPE SHARED)
        set(_LIB_DEFAULT_TYPE shared)
    endif()

    set(_SOURCES ${COMP_SOURCES} ${COMP_HEADERS})

    if(COMP_INSTALL_SOURCE)
        set(_SOURCES)
        foreach(_SOURCE ${COMP_SOURCES})
            get_filename_component(_SOURCE_NAME "${_SOURCE}" NAME)
            get_filename_component(_SOURCE_DIR "${_SOURCE}" DIRECTORY)
            get_filename_component(_SOURCE_DIR "${_SOURCE_DIR}" NAME)
            if(NOT "${_SOURCE_NAME}" STREQUAL "extern.cpp" AND NOT "${_SOURCE_DIR}"
                                                               STREQUAL "extern")
                list(APPEND _SOURCES ${_SOURCE})
            else()
                message(WARNING "Not installing ${_SOURCE} source")
            endif()
        endforeach()
        timemory_install_header_files(${_SOURCES})
    endif()

    foreach(LINK ${_LIB_TYPES})

        string(TOUPPER "${LINK}" UPP_LINK)
        set(TARGET_NAME timemory-${COMP_TARGET}-${LINK})

        if(NOT "${LINK}" STREQUAL OBJECT AND TARGET timemory-${COMP_TARGET}-object)
            set(_SOURCES $<TARGET_OBJECTS:timemory-${COMP_TARGET}-object>)
        endif()

        # set the depends before creating the library so it does not link to itself
        timemory_get_internal_depends(_DEPENDS ${LINK} ${COMP_DEPENDS} timemory-core)
        timemory_get_property_depends(_PROPERTY_OBJS OBJECT ${COMP_PROPERTY_DEPENDS})
        timemory_get_property_depends(_PROPERTY_LINK ${UPP_LINK} ${COMP_PROPERTY_DEPENDS})

        # must reset this variable
        set(DEPENDS)
        foreach(_DEP ${_DEPENDS} ${_PROPERTY_OBJS} ${_PROPERTY_LINK})
            if("${_DEP}" MATCHES ".*TARGET_OBJECTS:.*")
                list(APPEND _SOURCES ${_DEP})
            else()
                list(APPEND DEPENDS ${_DEP})
            endif()
        endforeach()

        if(DEPENDS)
            list(REMOVE_DUPLICATES DEPENDS)
        endif()

        set_property(GLOBAL APPEND PROPERTY TIMEMORY_HEADERS ${COMP_HEADERS})
        set_property(GLOBAL APPEND PROPERTY TIMEMORY_SOURCES ${COMP_SOURCES})
        set_property(
            GLOBAL APPEND PROPERTY TIMEMORY_${UPP_LINK}_${COMP_CATEGORY}_LIBRARIES
                                   timemory-${COMP_TARGET}-${LINK})

        # message(STATUS "Building ${TARGET_NAME}") message(STATUS "[-------]
        # ${TARGET_NAME} :: ${DEPENDS}")

        timemory_build_library(
            NO_CACHE_LIST ${${LINK}_OPTIONS}
            TARGET_NAME ${TARGET_NAME}
            OUTPUT_NAME timemory-${COMP_TARGET}
            LANGUAGE CXX
            LINKER_LANGUAGE ${_LINKER_LANGUAGE}
            OUTPUT_DIR ${PROJECT_BINARY_DIR}/${COMP_FOLDER}
            SOURCES ${_SOURCES}
            CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS})

        target_include_directories(${TARGET_NAME} PUBLIC ${COMP_INCLUDES})

        list(REMOVE_ITEM DEPENDS "${TARGET_NAME}")
        set(_DEPS ${DEPENDS})
        set(DEPENDS)
        foreach(_DEP ${_DEPS})
            if("${_DEP}" MATCHES "timemory::")
                list(APPEND DEPENDS ${_DEP})
            else()
                list(APPEND DEPENDS "timemory::${_DEP}")
            endif()
        endforeach()
        if(NOT "${DEPENDS}" STREQUAL "")
            list(REMOVE_DUPLICATES DEPENDS)
        endif()

        target_link_libraries(
            ${TARGET_NAME}
            PUBLIC timemory-external-${LINK} timemory-headers timemory-vector
                   timemory-dmp ${DEPENDS} ${COMP_PUBLIC_LINK})

        target_link_libraries(
            ${TARGET_NAME} PRIVATE timemory-compile-options timemory-develop-options
                                   ${COMP_PRIVATE_LINK})

        timemory_target_compile_definitions(
            ${TARGET_NAME} PRIVATE TIMEMORY_SOURCE TIMEMORY_${COMP_CATEGORY}_SOURCE
            TIMEMORY_${UPP_COMP}_SOURCE)

        timemory_target_compile_definitions(${TARGET_NAME} INTERFACE
                                            TIMEMORY_USE_${UPP_COMP}_EXTERN)

        if(NOT "${UPP_COMP}" STREQUAL "CORE")
            timemory_target_compile_definitions(${TARGET_NAME} PUBLIC
                                                TIMEMORY_USE_CORE_EXTERN)
        endif()

        if("${COMP_CATEGORY}" STREQUAL "COMPONENT" OR COMP_USE_CATEGORY)
            timemory_target_compile_definitions(${TARGET_NAME} INTERFACE
                                                TIMEMORY_USE_${COMP_CATEGORY}_EXTERN)
        endif()

        if("${LINK}" STREQUAL "OBJECT")
            if(NOT TARGET timemory-${LC_CATEGORY}-${LINK})
                timemory_add_interface_library(timemory-${LC_CATEGORY}-${LINK})
            endif()
            if(NOT "timemory-${LC_CATEGORY}-${LINK}" STREQUAL "${TARGET_NAME}")
                target_sources(timemory-${LC_CATEGORY}-${LINK}
                               INTERFACE $<TARGET_OBJECTS:${TARGET_NAME}>)
            endif()
        else()
            if(NOT TARGET timemory-${LC_CATEGORY}-${LINK})
                timemory_add_interface_library(timemory-${LC_CATEGORY}-${LINK})
            endif()

            if(NOT "timemory-${LC_CATEGORY}-${LINK}" STREQUAL "${TARGET_NAME}")
                target_link_libraries(timemory-${LC_CATEGORY}-${LINK}
                                      INTERFACE ${TARGET_NAME})
            endif()

            timemory_install_libraries(
                TARGETS ${TARGET_NAME}
                DESTINATION ${CMAKE_INSTALL_LIBDIR}
                EXPORT ${PROJECT_NAME}-library-depends)

            set_property(GLOBAL APPEND PROPERTY TIMEMORY_INTERMEDIATE_TARGETS
                                                ${TARGET_NAME})
            set_property(GLOBAL APPEND PROPERTY TIMEMORY_INTERMEDIATE_${UPP_LINK}_TARGETS
                                                ${TARGET_NAME})
        endif()

    endforeach()

    timemory_install_header_files(${COMP_HEADERS})
    if(COMP_INSTALL_SOURCE)
        timemory_install_header_files(${COMP_SOURCES})
    endif()

    if(NOT TARGET timemory-${COMP_TARGET})
        timemory_add_interface_library(timemory-${COMP_TARGET})
        target_link_libraries(timemory-${COMP_TARGET}
                              INTERFACE timemory-${COMP_TARGET}-${_LIB_DEFAULT_TYPE})
    endif()

    if(NOT TARGET timemory-${LC_CATEGORY})
        timemory_add_interface_library(timemory-${LC_CATEGORY})
        target_link_libraries(timemory-${LC_CATEGORY}
                              INTERFACE timemory-${LC_CATEGORY}-${_LIB_DEFAULT_TYPE})
    endif()

    if(WIN32
       AND TARGET timemory-${COMP_TARGET}-shared
       AND TARGET timemory-${COMP_TARGET}-static)
        add_dependencies(timemory-${COMP_TARGET}-shared timemory-${COMP_TARGET}-static)
    endif()

endfunction()

# ----------------------------------------------------------------------------------------#
# function to add a definition to defines.h
#
function(TIMEMORY_ADD_CMAKE_DEFINES _VAR)
    # parse args
    cmake_parse_arguments(DEF "VALUE;QUOTE;DEFAULT" "" "" ${ARGN})
    if(DEF_VALUE)
        if(DEF_QUOTE)
            set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES
                                                "${_VAR} \"@${_VAR}@\"")
        else()
            set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES
                                                "${_VAR} @${_VAR}@")
        endif()
        if(DEF_DEFAULT)
            if(DEF_QUOTE)
                set_property(
                    GLOBAL APPEND
                    PROPERTY
                        ${PROJECT_NAME}_DEFAULT_CMAKE_DEFINES
                        "#if !defined(${_VAR})\n#cmakedefine ${_VAR} \"@${_VAR}@\"\n#endif\n"
                    )
            else()
                set_property(
                    GLOBAL APPEND
                    PROPERTY
                        ${PROJECT_NAME}_DEFAULT_CMAKE_DEFINES
                        "#if !defined(${_VAR})\n#cmakedefine ${_VAR} @${_VAR}@\n#endif\n")
            endif()
        endif()
    else()
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES "${_VAR}")
    endif()
endfunction()

# -----------------------------------------------------------------------
# function timemory_add_feature(<NAME> <DOCSTRING>) Add a project feature, whose
# activation is specified by the existence of the variable <NAME>, to the list of
# enabled/disabled features, plus a docstring describing the feature
#
function(TIMEMORY_ADD_FEATURE _var _description)
    set(EXTRA_DESC "")
    foreach(currentArg ${ARGN})
        if(NOT "${currentArg}" STREQUAL "${_var}"
           AND NOT "${currentArg}" STREQUAL "${_description}"
           AND NOT "${currentArg}" STREQUAL "CMAKE_DEFINE"
           AND NOT "${currentArg}" STREQUAL "DOC")
            set(EXTRA_DESC "${EXTA_DESC}${currentArg}")
        endif()
    endforeach()

    set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_FEATURES ${_var})
    set_property(GLOBAL PROPERTY ${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")

    if("CMAKE_DEFINE" IN_LIST ARGN)
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES
                                            "${_var} @${_var}@")
        if(TIMEMORY_BUILD_DOCS)
            set_property(
                GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_OPTIONS_DOC
                                       "${_var}` | ${_description}${EXTRA_DESC} |")
        endif()
    elseif("DOC" IN_LIST ARGN AND TIMEMORY_BUILD_DOCS)
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_OPTIONS_DOC
                                            "${_var}` | ${_description}${EXTRA_DESC} |")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function timemory_add_option(<OPTION_NAME> <DOCSRING> <DEFAULT_SETTING> [NO_FEATURE])
# Add an option and add as a feature if NO_FEATURE is not provided
#
function(TIMEMORY_ADD_OPTION _NAME _MESSAGE _DEFAULT)
    option(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    if("NO_FEATURE" IN_LIST ARGN)
        mark_as_advanced(${_NAME})
    else()
        timemory_add_feature(${_NAME} "${_MESSAGE}")
        if(TIMEMORY_BUILD_DOCS)
            set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_OPTIONS_DOC
                                                "${_NAME}` | ${_MESSAGE} |")
        endif()
    endif()
    if("ADVANCED" IN_LIST ARGN)
        mark_as_advanced(${_NAME})
    endif()
    if("CMAKE_DEFINE" IN_LIST ARGN)
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES ${_NAME})
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function timemory_print_enabled_features() Print enabled  features plus their
# docstrings.
#
function(TIMEMORY_PRINT_ENABLED_FEATURES)
    set(_basemsg "The following features are defined/enabled (+):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY ${PROJECT_NAME}_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(${_feature})
            # add feature to text
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
            # get description
            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)
            # print description, if not standard ON/OFF, print what is set to
            if(_desc)
                if(NOT "${${_feature}}" STREQUAL "ON" AND NOT "${${_feature}}" STREQUAL
                                                          "TRUE")
                    set(_currentFeatureText
                        "${_currentFeatureText}: ${_desc} -- [\"${${_feature}}\"]")
                else()
                    string(REGEX REPLACE "^${PROJECT_NAME}_USE_" "" _feature_tmp
                                         "${_feature}")
                    string(TOLOWER "${_feature_tmp}" _feature_tmp_l)
                    timemory_capitalize("${_feature_tmp}" _feature_tmp_c)
                    foreach(_var _feature _feature_tmp _feature_tmp_l _feature_tmp_c)
                        set(_ver "${${${_var}}_VERSION}")
                        if(NOT "${_ver}" STREQUAL "")
                            set(_desc "${_desc} -- [found version ${_ver}]")
                            break()
                        endif()
                        unset(_ver)
                    endforeach()
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
                endif()
                set(_desc NOTFOUND)
            endif()
        endif()
    endforeach()

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function timemory_print_disabled_features() Print disabled features plus their
# docstrings.
#
function(TIMEMORY_PRINT_DISABLED_FEATURES)
    set(_basemsg "The following features are NOT defined/enabled (-):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY ${PROJECT_NAME}_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(NOT ${_feature})
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")

            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)

            if(_desc)
                set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
                set(_desc NOTFOUND)
            endif(_desc)
        endif()
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function timemory_print_enabled_interfaces() Print enabled INTERFACE libraries plus
# their docstrings.
#
function(TIMEMORY_PRINT_ENABLED_INTERFACES)
    set(_basemsg "The following INTERFACE libraries are enabled:")
    set(_currentFeatureText "${_basemsg}")
    get_property(_enabled GLOBAL PROPERTY ${PROJECT_NAME}_ENABLED_INTERFACES)
    get_property(_disabled GLOBAL PROPERTY ${PROJECT_NAME}_DISABLED_INTERFACES)
    if(NOT "${_enabled}" STREQUAL "")
        list(REMOVE_DUPLICATES _enabled)
        list(SORT _enabled)
    endif()
    foreach(_LIB ${_disabled})
        if("${_LIB}" IN_LIST _enabled)
            list(REMOVE_ITEM _enabled ${_LIB})
        endif()
    endforeach()
    foreach(_feature ${_enabled})
        # add feature to text
        set(_currentFeatureText "${_currentFeatureText}\n     timemory::${_feature}")
    endforeach()

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function timemory_print_disabled_interfaces() Print disabled interfaces plus their
# docstrings.
#
function(TIMEMORY_PRINT_DISABLED_INTERFACES)
    set(_basemsg
        "The following INTERFACE libraries are NOT enabled (empty INTERFACE libraries):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_disabled GLOBAL PROPERTY ${PROJECT_NAME}_DISABLED_INTERFACES)
    if(NOT "${_disabled}" STREQUAL "")
        list(REMOVE_DUPLICATES _disabled)
        list(SORT _disabled)
    endif()
    foreach(_feature ${_disabled})
        set(_currentFeatureText "${_currentFeatureText}\n     timemory::${_feature}")
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function timemory_print_features() Print all features plus their docstrings.
#
function(TIMEMORY_PRINT_FEATURES)
    message(STATUS "")
    timemory_print_enabled_features()
    timemory_print_disabled_features()
    message(STATUS "")
    timemory_print_enabled_interfaces()
    timemory_print_disabled_interfaces()
endfunction()

# ----------------------------------------------------------------------------------------#
# function timemory_get_enabled_languages() Gets the enabled languages property
#
function(timemory_get_enabled_languages _VAR)
    get_property(_ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    set(${_VAR}
        ${_ENABLED_LANGUAGES}
        PARENT_SCOPE)
endfunction()

cmake_policy(POP)
