# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# Creates a 'format' target that runs clang-format
#
# ########################################################################################

set(_FMT ON)
# Visual Studio GUI reports "errors" occasionally
if(WIN32)
    set(_FMT OFF)
endif()

option(TIMEMORY_FORMAT_TARGET "Enable a clang-format target" ${_FMT})
mark_as_advanced(TIMEMORY_FORMAT_TARGET)

if(NOT TIMEMORY_FORMAT_TARGET)
    return()
endif()

# prefer clang-format 9.0, unset if 6 (old version)
if("${TIMEMORY_CLANG_FORMATTER}" MATCHES "6")
    unset(TIMEMORY_CLANG_FORMATTER CACHE)
endif()

# C / C++ formatting
find_program(TIMEMORY_CLANG_FORMATTER NAMES clang-format-9 clang-format-9.0
                                            clang-format-mp-9.0 clang-format)

# python formatting
find_program(TIMEMORY_BLACK_FORMATTER NAMES black)

# cmake formatting
find_program(TIMEMORY_CMAKE_FORMATTER NAMES cmake-format)

mark_as_advanced(TIMEMORY_CLANG_FORMATTER)
mark_as_advanced(TIMEMORY_BLACK_FORMATTER)
mark_as_advanced(TIMEMORY_CMAKE_FORMATTER)

if(TIMEMORY_CLANG_FORMATTER
   OR TIMEMORY_BLACK_FORMATTER
   OR TIMEMORY_CMAKE_FORMATTER)
    # name of the format target
    set(FORMAT_NAME format)
    if(TARGET format)
        set(FORMAT_NAME format-timemory)
    endif()

    # generic format target
    add_custom_target(${FORMAT_NAME})
endif()

if(TIMEMORY_CLANG_FORMATTER)
    file(
        GLOB_RECURSE
        _headers
        ${PROJECT_SOURCE_DIR}/source/tools/*.hpp
        ${PROJECT_SOURCE_DIR}/source/tests/*.hpp
        ${PROJECT_SOURCE_DIR}/source/python/*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/*.h
        ${PROJECT_SOURCE_DIR}/source/timemory/*.hpp)
    if(TIMEMORY_SOURCE_GROUP)
        source_group(TREE ${PROJECT_SOURCE_DIR}/source FILES ${headers})
    endif()
    file(
        GLOB_RECURSE
        sources
        ${PROJECT_SOURCE_DIR}/source/*.c
        ${PROJECT_SOURCE_DIR}/source/*.cu
        ${PROJECT_SOURCE_DIR}/source/*.cpp
        ${PROJECT_SOURCE_DIR}/source/*.cpp.in
        ${PROJECT_SOURCE_DIR}/source/tools/*.c
        ${PROJECT_SOURCE_DIR}/source/tools/*.cu
        ${PROJECT_SOURCE_DIR}/source/tools/*.cpp
        ${PROJECT_SOURCE_DIR}/source/tools/*.cpp.in
        ${PROJECT_SOURCE_DIR}/source/tests/*.c
        ${PROJECT_SOURCE_DIR}/source/tests/*.cu
        ${PROJECT_SOURCE_DIR}/source/tests/*.cpp
        ${PROJECT_SOURCE_DIR}/source/tests/*.cpp.in)
    if(TIMEMORY_SOURCE_GROUP)
        source_group(TREE ${PROJECT_SOURCE_DIR}/source FILES ${sources})
    endif()
    if(TIMEMORY_BUILD_EXAMPLES)
        file(
            GLOB_RECURSE
            examples
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.h
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.c
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.hpp
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.cpp
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.cuh
            ${PROJECT_SOURCE_DIR}/examples/ex-*/*.cu)
        if(TIMEMORY_SOURCE_GROUP)
            source_group(TREE ${PROJECT_SOURCE_DIR}/examples FILES ${examples})
        endif()
    else()
        set(examples)
    endif()

    # remove rapidjson and rapidxml
    set(_tpl_base "${PROJECT_SOURCE_DIR}/source/timemory/tpls")
    file(GLOB_RECURSE _tpl_headers "${_tpl_base}/*")
    file(GLOB_RECURSE _ext_headers "${_tpl_base}/cereal/cereal/external/*")
    set(headers)
    set(tpl_headers)
    foreach(_h ${_headers})
        if(NOT "${_h}" IN_LIST _tpl_headers)
            list(APPEND headers ${_h})
        elseif(NOT "${_h}" IN_LIST _ext_headers)
            list(APPEND tpl_headers ${_h})
        endif()
    endforeach()
    unset(_tpl_headers)
    unset(_ext_headers)
    unset(_headers)

    # always have files
    set(_COMMAND COMMAND ${TIMEMORY_CLANG_FORMATTER} -i ${headers} COMMAND
                 ${TIMEMORY_CLANG_FORMATTER} -i ${sources})

    # might have many files
    if(tpl_headers)
        set(_COMMAND ${_COMMAND} COMMAND ${TIMEMORY_CLANG_FORMATTER} -i ${tpl_headers})
    endif()

    # might be empty
    if(TIMEMORY_BUILD_EXAMPLES)
        set(_COMMAND ${_COMMAND} COMMAND ${TIMEMORY_CLANG_FORMATTER} -i ${examples})
    endif()

    # source specific target
    add_custom_target(
        ${FORMAT_NAME}-source
        ${_COMMAND}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT
            "[${PROJECT_NAME}] Running C/C++ formatter '${TIMEMORY_CLANG_FORMATTER}'..."
        SOURCES ${headers} ${sources} ${examples})

    add_dependencies(${FORMAT_NAME} ${FORMAT_NAME}-source)
endif()

if(TIMEMORY_BLACK_FORMATTER)
    add_custom_target(
        ${FORMAT_NAME}-python
        COMMAND ${TIMEMORY_BLACK_FORMATTER} -q ${PROJECT_SOURCE_DIR}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT
            "[${PROJECT_NAME}] Running Python formatter '${TIMEMORY_BLACK_FORMATTER}'...")

    add_dependencies(${FORMAT_NAME} ${FORMAT_NAME}-python)
endif()

if(TIMEMORY_CMAKE_FORMATTER)
    file(
        GLOB_RECURSE
        CMAKE_FORMAT_FILES
        ${PROJECT_SOURCE_DIR}/cmake/*.cmake
        ${PROJECT_SOURCE_DIR}/cmake/*.cmake.in
        ${PROJECT_SOURCE_DIR}/source/*CMakeLists.txt
        ${PROJECT_SOURCE_DIR}/examples/*CMakeLists.txt)

    list(INSERT CMAKE_FORMAT_FILES 0 ${PROJECT_SOURCE_DIR}/CMakeLists.txt
         ${PROJECT_SOURCE_DIR}/source/CMakeLists.txt
         ${PROJECT_SOURCE_DIR}/external/CMakeLists.txt)

    add_custom_target(
        ${FORMAT_NAME}-cmake
        COMMAND ${TIMEMORY_CMAKE_FORMATTER} -i ${CMAKE_FORMAT_FILES}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT
            "[${PROJECT_NAME}] Running CMake formatter '${TIMEMORY_CMAKE_FORMATTER}'..."
        SOURCES ${CMAKE_FORMAT_FILES})

    # don't add dependency bc cmake-format is really slow
endif()
