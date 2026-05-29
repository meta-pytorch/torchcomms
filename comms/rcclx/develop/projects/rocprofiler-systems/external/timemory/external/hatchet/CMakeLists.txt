
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

execute_process(
    COMMAND             python ${CMAKE_CURRENT_LIST_DIR}/.version.py
    OUTPUT_VARIABLE     FULL_VERSION_STRING
    WORKING_DIRECTORY   ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
string(REGEX REPLACE "[A-Za-z].*" "" VERSION_STRING "${FULL_VERSION_STRING}")
set(HATCHET_VERSION "${VERSION_STRING}")

project(hatchet LANGUAGES C VERSION ${HATCHET_VERSION})

if(TARGET libpytimemory)
    # where we want to install timemory
    set(TIMEMORY_HATCHET_DIR ${CMAKE_BINARY_DIR}/timemory/hatchet)

    # make sure same compiler is used
    set(ENV{CC} "${CMAKE_C_COMPILER}")

    # really annoying python 3.8 warnings
    if(cxx_timemory_python_compile_options_Wno_deprecated_declarations)
        set(CFLAGS "-Wno-deprecated-declarations")
    endif()

    set(SETUP_PY_FILE .setup.${CMAKE_SYSTEM_NAME}.${CMAKE_C_COMPILER_ID}.${CMAKE_C_COMPILER_VERSION}.${CMAKE_BUILD_TYPE}.py)
    # create custom setup.py with (potentially) extra flags
    configure_file(${PROJECT_SOURCE_DIR}/setup.py ${PROJECT_SOURCE_DIR}/${SETUP_PY_FILE} @ONLY)

    # target for adding a dependency to libpytimemory target
    add_custom_target(hatchet
        COMMAND
            ${PYTHON_EXECUTABLE} ${SETUP_PY_FILE} --quiet build_ext --inplace
        WORKING_DIRECTORY
            ${PROJECT_SOURCE_DIR}
        COMMENT
            "[hatchet] ${PYTHON_EXECUTABLE} ${SETUP_PY_FILE} --quiet build_ext --inplace")

    # create a post-build command sequence for our dummy hatchet target
    add_custom_command(TARGET hatchet
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory
                ${PROJECT_SOURCE_DIR}/hatchet
                ${TIMEMORY_HATCHET_DIR}
        WORKING_DIRECTORY
            ${PROJECT_SOURCE_DIR}
        COMMENT
            "[hatchet] Copying to ${TIMEMORY_HATCHET_DIR}...")
endif()
