if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/PyCTestPreCustomInit.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/PyCTestPreCustomInit.cmake")
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/CustomInit.cmake")

    set(_SUBMIT_STAGES)
    include("${CMAKE_CURRENT_LIST_DIR}/CustomInit.cmake")
    include_if("${CMAKE_CURRENT_LIST_DIR}/PyCTestPostCustomInit.cmake")

    include_if("${CMAKE_CURRENT_LIST_DIR}/CTestConfig.cmake")
    include_if("${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake")
    copy_ctest_config_files()
    ctest_read_custom_files("${CMAKE_CURRENT_LIST_DIR}")

    set(CTEST_DROP_SITE_CDASH TRUE)
    set(CTEST_SUBMIT_URL "https://my.cdash.org/submit.php?project=timemory")
    set(CTEST_GIT_INIT_SUBMODULES TRUE)
    set(CTEST_USE_LAUNCHERS TRUE)
    set(CTEST_NIGHTLY_START_TIME "05:00:00 UTC")

    list(FIND STAGES "Start" DO_START)
    list(FIND STAGES "Update" DO_UPDATE)
    list(FIND STAGES "Configure" DO_CONFIGURE)
    list(FIND STAGES "Build" DO_BUILD)
    list(FIND STAGES "Test" DO_TEST)
    list(FIND STAGES "Coverage" DO_COVERAGE)
    list(FIND STAGES "MemCheck" DO_MEMCHECK)
    list(FIND STAGES "Submit" DO_SUBMIT)

    message(STATUS "")
    message(STATUS "STAGES = ${STAGES}")
    message(STATUS "")

    macro(SUBMIT_STAGE)
        if("${DO_SUBMIT}" GREATER -1)
            submit_command(PARTS ${_SUBMIT_STAGES} ${ARGN})
            set(_SUBMIT_STAGES)
        endif()
    endmacro()

    macro(HANDLE_ERROR _message _ret)
        if(NOT ${${_ret}} EQUAL 0)
            read_notes()
            read_presubmit_scripts()
            submit_stage(Notes ExtraFiles Upload Submit)
            message(FATAL_ERROR "${_message} failed: ${${_ret}}")
        endif()
    endmacro()

    if(NOT CTEST_MODEL)
        set(CTEST_MODEL Continuous)
    endif()

    if(NOT CTEST_UPDATE_COMMAND)
        if(GIT_COMMAND)
            set(CTEST_UPDATE_COMMAND ${GIT_COMMAND})
            set(CTEST_GIT_COMMAND ${GIT_COMMAND})
        else()
            set(CTEST_UPDATE_COMMAND git)
            set(CTEST_GIT_COMMAND git)
        endif()
    endif()

    set_if_defined(CTEST_START START _CTEST_START)
    set_if_defined(CTEST_END END _CTEST_END)
    set_if_defined(CTEST_STRIDE STRIDE _CTEST_STRIDE)
    set_if_defined(CTEST_INCLUDE INCLUDE _CTEST_INCLUDE)
    set_if_defined(CTEST_EXCLUDE EXCLUDE _CTEST_EXCLUDE)
    set_if_defined(CTEST_INCLUDE_LABEL INCLUDE_LABEL _CTEST_INCLUDE_LABEL)
    set_if_defined(CTEST_EXCLUDE_LABEL EXCLUDE_LABEL _CTEST_EXCLUDE_LABEL)
    set_if_defined(CTEST_PARALLEL_LEVEL PARALLEL_LEVEL _CTEST_PARALLEL_LEVEL)
    set_if_defined(CTEST_STOP_TIME STOP_TIME _CTEST_STOP_TIME)
    set_if_defined(CTEST_COVERAGE_LABELS LABELS _CTEST_COVERAGE_LABELS)

    if(CTEST_APPEND)
        set(DO_START -1)
    endif()

    # ---------------------------------------------------------------------------------- #
    # Start
    #
    if("${DO_START}" GREATER -1)
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Running CTEST_START stage...")
        message(STATUS "")
        set(_CTEST_APPEND)
        set(_CTEST_VERB "Running")
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Appending CTEST_START stage...")
        message(STATUS "")
        set(_CTEST_APPEND APPEND)
        set(_CTEST_VERB "Appending")
    endif()

    ctest_start(${CTEST_MODEL} "${CTEST_SOURCE_DIRECTORY}" "${CTEST_BINARY_DIRECTORY}"
                ${_CTEST_APPEND})
    list(APPEND _SUBMIT_STAGES Start)

    # ---------------------------------------------------------------------------------- #
    # Update
    #
    if("${DO_UPDATE}" GREATER -1 AND (NOT "${CTEST_UPDATE_COMMAND}" STREQUAL ""
                                      OR NOT "${CTEST_UPDATE_TYPE}" STREQUAL ""))
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_UPDATE stage...")
        message(STATUS "")
        ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}" RETURN_VALUE update_ret)

        list(APPEND _SUBMIT_STAGES Update)
        handle_error("Update" update_ret)
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_UPDATE stage...")
        message(STATUS "")
    endif()

    # ---------------------------------------------------------------------------------- #
    # Configure
    #
    if("${DO_CONFIGURE}" GREATER -1 AND NOT "${CTEST_CONFIGURE_COMMAND}" STREQUAL "")
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_CONFIGURE stage...")
        message(STATUS "")
        ctest_configure(
            BUILD "${CTEST_BINARY_DIRECTORY}"
            SOURCE ${CTEST_SOURCE_DIRECTORY}
            ${_CTEST_APPEND}
            OPTIONS "${CTEST_CONFIGURE_OPTIONS}"
            RETURN_VALUE configure_ret)

        submit_stage(Configure)
        handle_error("Configure" configure_ret)
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_CONFIGURE stage...")
        message(STATUS "")
    endif()

    # ---------------------------------------------------------------------------------- #
    # Build
    #
    if("${DO_BUILD}" GREATER -1 AND NOT "${CTEST_BUILD_COMMAND}" STREQUAL "")
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_BUILD stage...")
        message(STATUS "")
        ctest_build(
            BUILD "${CTEST_BINARY_DIRECTORY}"
            ${_CTEST_APPEND}
            RETURN_VALUE build_ret)

        submit_stage(Build)
        handle_error("Build" build_ret)
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_BUILD stage...")
        message(STATUS "")
    endif()

    # ---------------------------------------------------------------------------------- #
    # Test
    #
    if("${DO_TEST}" GREATER -1)
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_TEST stage...")
        message(STATUS "")

        set(_CTEST_REPEAT)
        if(NOT CMAKE_VERSION VERSION_LESS 3.17)
            set(_CTEST_REPEAT "REPEAT" "UNTIL_PASS:3")
        endif()

        ctest_test(
            BUILD "${CTEST_BINARY_DIRECTORY}"
            RETURN_VALUE test_ret
            ${_CTEST_APPEND}
            ${_CTEST_START}
            ${_CTEST_END}
            ${_CTEST_STRIDE}
            ${_CTEST_INCLUDE}
            ${_CTEST_EXCLUDE}
            ${_CTEST_INCLUDE_LABEL}
            ${_CTEST_EXCLUDE_LABEL}
            ${_CTEST_PARALLEL_LEVEL}
            ${_CTEST_STOP_TIME}
            ${_CTEST_REPEAT}
            SCHEDULE_RANDOM OFF)

        submit_stage(Test)
        handle_error("Test" test_ret)
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_TEST stage...")
        message(STATUS "")
    endif()

    # ---------------------------------------------------------------------------------- #
    # Coverage
    #
    if("${DO_COVERAGE}" GREATER -1 AND NOT "${CTEST_COVERAGE_COMMAND}" STREQUAL "")
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_COVERAGE stage...")
        message(STATUS "")
        execute_process(
            COMMAND ${CTEST_COVERAGE_COMMAND} ${CTEST_COVERAGE_EXTRA_FLAGS}
            WORKING_DIRECTORY ${CTEST_BINARY_DIRECTORY}
            ERROR_QUIET)
        ctest_coverage(
            BUILD "${CTEST_BINARY_DIRECTORY}"
            ${_CTEST_APPEND} ${_CTEST_COVERAGE_LABELS}
            RETURN_VALUE coverage_ret)
        # remove the "coverage.xml" file after it has been processed on macOS because the
        # file-system is not case-sensitive
        if(APPLE AND EXISTS "${CTEST_BINARY_DIRECTORY}/coverage.xml")
            execute_process(COMMAND ${CTEST_CMAKE_COMMAND} -E remove
                                    ${CTEST_BINARY_DIRECTORY}/coverage.xml ERROR_QUIET)
        endif()

        submit_stage(Coverage)
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_COVERAGE stage...")
        message(STATUS "")
    endif()

    # ---------------------------------------------------------------------------------- #
    # MemCheck
    #
    if("${DO_MEMCHECK}" GREATER -1 AND NOT "${CTEST_MEMORYCHECK_COMMAND}" STREQUAL "")
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] ${_CTEST_VERB} CTEST_MEMCHECK stage...")
        message(STATUS "")
        ctest_memcheck(
            RETURN_VALUE
                memcheck_ret
                ${_CTEST_APPEND}
                ${_CTEST_START}
                ${_CTEST_END}
                ${_CTEST_STRIDE}
                ${_CTEST_INCLUDE}
                ${_CTEST_EXCLUDE}
                ${_CTEST_INCLUDE_LABEL}
                ${_CTEST_EXCLUDE_LABEL}
                ${_CTEST_PARALLEL_LEVEL})

        submit_stage(MemCheck)
    else()
        message(STATUS "")
        message(STATUS "[${CTEST_BUILD_NAME}] Skipping CTEST_MEMCHECK stage...")
        message(STATUS "")
    endif()

    # ---------------------------------------------------------------------------------- #
    # Submit
    #
    read_notes()
    read_presubmit_scripts()
    submit_stage(Notes ExtraFiles Upload Submit)

    message(STATUS "")
    message(STATUS "[${CTEST_BUILD_NAME}] Finished ${CTEST_MODEL} Stages (${STAGES})")
    message(STATUS "")

endif()
