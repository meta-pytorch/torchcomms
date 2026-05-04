# include guard
if(__pyctestinit_is_loaded)
    return()
endif()
set(__pyctestinit_is_loaded ON)

include(ProcessorCount)
processorcount(CTEST_PROCESSOR_COUNT)

cmake_policy(SET CMP0009 NEW)
cmake_policy(SET CMP0011 NEW)

# -------------------------------------------------------------------------------------- #
# -- Commands
# -------------------------------------------------------------------------------------- #
find_program(CTEST_CMAKE_COMMAND NAMES cmake)
find_program(UNAME_COMMAND NAMES uname)

find_program(GIT_COMMAND NAMES git)
find_program(VALGRIND_COMMAND NAMES valgrind)
find_program(GCOV_COMMAND NAMES gcov)
find_program(LCOV_COMMAND NAMES llvm-cov)
find_program(MEMORYCHECK_COMMAND NAMES valgrind)

set(MEMORYCHECK_TYPE Valgrind)
# set(MEMORYCHECK_TYPE Purify) set(MEMORYCHECK_TYPE BoundsChecker) set(MEMORYCHECK_TYPE
# ThreadSanitizer) set(MEMORYCHECK_TYPE AddressSanitizer) set(MEMORYCHECK_TYPE
# LeakSanitizer) set(MEMORYCHECK_TYPE MemorySanitizer) set(MEMORYCHECK_TYPE
# UndefinedBehaviorSanitizer)
set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full")

# -------------------------------------------------------------------------------------- #
# -- Include file if exists
# -------------------------------------------------------------------------------------- #
macro(include_if _TESTFILE)
    if(EXISTS "${_TESTFILE}")
        include("${_TESTFILE}")
    else()
        if(NOT "${ARGN}" STREQUAL "")
            include("${ARGN}")
        endif()
    endif()
endmacro()

# -------------------------------------------------------------------------------------- #
# -- Settings
# -------------------------------------------------------------------------------------- #
# -- Process timeout in seconds
if(NOT DEFINED CTEST_TIMEOUT)
    set(CTEST_TIMEOUT "7200")
endif()
# -- Set output to English
set(ENV{LC_MESSAGES} "en_EN")

# -------------------------------------------------------------------------------------- #
# -- Set if defined
# -------------------------------------------------------------------------------------- #
macro(SET_IF_DEFINED IF_DEF_VAR PREFIX_VAR SET_VAR)
    if(DEFINED "${IF_DEF_VAR}")
        set(${SET_VAR} "${PREFIX_VAR} \"${${IF_DEF_VAR}}\"")
    endif()
endmacro()

# -------------------------------------------------------------------------------------- #
# -- Copy ctest configuration file
# -------------------------------------------------------------------------------------- #
macro(COPY_CTEST_CONFIG_FILES)
    if(NOT "${CMAKE_CURRENT_LIST_DIR}" STREQUAL "${CTEST_BINARY_DIRECTORY}"
       AND NOT "${CTEST_SOURCE_DIRECTORY}" STREQUAL "${CTEST_BINARY_DIRECTORY}")
        # -- CTest Config
        configure_file(${CMAKE_CURRENT_LIST_DIR}/CTestConfig.cmake
                       ${CTEST_BINARY_DIRECTORY}/CTestConfig.cmake COPYONLY)
        # -- CTest Custom
        configure_file(${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake
                       ${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake COPYONLY)
    endif()
endmacro()

# -------------------------------------------------------------------------------------- #
# -- Run submit scripts
# -------------------------------------------------------------------------------------- #
macro(READ_PRESUBMIT_SCRIPTS)
    # check
    file(GLOB_RECURSE PRESUBMIT_SCRIPTS
         "${CTEST_BINARY_DIRECTORY}/*CTestPreSubmitScript.cmake")
    if("${PRESUBMIT_SCRIPTS}" STREQUAL "")
        message(
            STATUS
                "No scripts matching '${CTEST_BINARY_DIRECTORY}/*CTestPreSubmitScript.cmake' were found"
            )
    endif()
    foreach(_FILE ${PRESUBMIT_SCRIPTS})
        message(STATUS "Including pre-submit script: \"${_FILE}\"...")
        include("${_FILE}")
    endforeach()
endmacro()

# -------------------------------------------------------------------------------------- #
# -- Read CTestNotes.cmake file
# -------------------------------------------------------------------------------------- #
macro(READ_NOTES)
    # check
    file(GLOB_RECURSE NOTE_FILES "${CTEST_BINARY_DIRECTORY}/*CTestNotes.cmake")
    foreach(_FILE ${NOTE_FILES})
        message(STATUS "Including CTest notes files: \"${_FILE}\"...")
        include("${_FILE}")
    endforeach()
endmacro()

# -------------------------------------------------------------------------------------- #
# -- Check to see if there is a ctest token (for CDash submission)
# -------------------------------------------------------------------------------------- #
macro(CHECK_FOR_CTEST_TOKEN)

    # set using token to off
    set(CTEST_USE_TOKEN OFF)
    # set token to empty
    set(CTEST_TOKEN "")

    if(NOT "${CTEST_TOKEN_FILE}" STREQUAL "")
        string(REGEX REPLACE "^~" "$ENV{HOME}" CTEST_TOKEN_FILE "${CTEST_TOKEN_FILE}")
    endif()

    # check for a file containing token
    if(NOT "${CTEST_TOKEN_FILE}" STREQUAL "" AND EXISTS "${CTEST_TOKEN_FILE}")
        message(STATUS "Reading CTest token file: ${CTEST_TOKEN_FILE}")
        file(READ "${CTEST_TOKEN_FILE}" CTEST_TOKEN)
        string(REPLACE "\n" "" CTEST_TOKEN "${CTEST_TOKEN}")
    endif()

    # if no file, check the environment
    if("${CTEST_TOKEN}" STREQUAL "" AND NOT "$ENV{CTEST_TOKEN}" STREQUAL "")
        set(CTEST_TOKEN "$ENV{CTEST_TOKEN}")
    endif()

    # if non-empty token, set CTEST_USE_TOKEN to ON
    if(NOT "${CTEST_TOKEN}" STREQUAL "")
        set(CTEST_USE_TOKEN ON)
    endif()

endmacro()

# -------------------------------------------------------------------------------------- #
# -- Submit command
# -------------------------------------------------------------------------------------- #
macro(SUBMIT_COMMAND)
    check_for_ctest_token()

    if(NOT CTEST_USE_TOKEN)
        # standard submit
        ctest_submit(
            ${ARGN}
            RETURN_VALUE submit_ret
            RETRY_COUNT 3
            RETRY_DELAY 5
            CAPTURE_CMAKE_ERROR submit_err)
    else()
        # submit with token
        ctest_submit(
            ${ARGN}
            RETURN_VALUE submit_ret
            # HTTPHEADER "Authorization: Bearer ${CTEST_TOKEN}"
            RETRY_COUNT 3
            RETRY_DELAY 5
            CAPTURE_CMAKE_ERROR submit_err)
    endif()

    if(NOT submit_ret EQUAL 0)
        message(WARNING "Submission failed with exit code: ${submit_ret}")
    endif()
endmacro()
