cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(line_profiler LANGUAGES C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(NOT Python_ADDITIONAL_VERSIONS)
    set(Python_ADDITIONAL_VERSIONS 3.9 3.8 3.7 3.6)
endif()

set(CMAKE_UNITY_BUILD OFF)
set(_PYVERSION 3.6)
if(TIMEMORY_PYTHON_VERSION)
    set(_PYVERSION "${TIMEMORY_PYTHON_VERSION}")
    unset(Python_ADDITIONAL_VERSIONS)
endif()

include(ConfigPython)

find_package(PythonInterp ${_PYVERSION} EXACT REQUIRED)
find_package(PythonExtensions ${_PYVERSION} EXACT REQUIRED)
find_package(PythonLibs ${_PYVERSION} EXACT REQUIRED)

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    find_package(Cython REQUIRED)
else()
    find_package(Cython)
    if(NOT Cython_FOUND)
        return()
    endif()
endif()

add_subdirectory(line_profiler)

# install kernprof as the __main__
configure_file(${PROJECT_SOURCE_DIR}/kernprof.py
    ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}/line_profiler/__main__.py
    COPYONLY)
