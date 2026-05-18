cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(timemory-Python-Example LANGUAGES C CXX)

if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    set(TARGET_BINARY_DIR ${CMAKE_BINARY_DIR})
    if(NOT TIMEMORY_USE_PYTHON)
        return()
    endif()
else()
    set(TARGET_BINARY_DIR ${PROJECT_BINARY_DIR})
endif()

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

function(CONFIGURE_PYTHON_SCRIPT)
    foreach(_TYPE ${ARGN})
        set(FILENAME ex_python_${_TYPE})
        configure_file(${PROJECT_SOURCE_DIR}/ex_${_TYPE}.py
                       ${TARGET_BINARY_DIR}/ex_python_${_TYPE} @ONLY)
        install(
            FILES ${TARGET_BINARY_DIR}/ex_python_${_TYPE}
            DESTINATION bin
            PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ
                        WORLD_EXECUTE WORLD_READ
            COMPONENT examples
            OPTIONAL)
    endforeach()
endfunction()

configure_python_script(sample tracer profiler general builtin external)

if(NOT TARGET pybind11::module)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        return()
    endif()
endif()

option(USE_MPI "Enable MPI in timemory example" ON)
if(USE_MPI)
    find_package(MPI QUIET)
    if(NOT MPI_FOUND)
        return()
    endif()
endif()

set(_OMP)
option(USE_OPENMP "Enable OpenMP in timemory example" OFF)
if(TIMEMORY_USE_OMPT OR USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        set(_OMP OpenMP::OpenMP_CXX)
    endif()
endif()

configure_python_script(bindings)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(TIMEMORY_INTERFACE_LIBRARY timemory-ex-python-bindings)
find_package(
    timemory
    COMPONENTS headers
    OPTIONAL_COMPONENTS mpi cxx shared)

pybind11_add_module(libex_python_bindings THIN_LTO
                    ${PROJECT_SOURCE_DIR}/libex_bindings.cpp)

include(CheckCXXCompilerFlag)

set(_COMPILE_OPTS) # for -Wno-attributes if available by the compiler
if(TARGET timemory-compile-options)
    set(_COMPILE_OPTS timemory-compile-options)
elseif(TARGET timemory::timemory-compile-options)
    set(_COMPILE_OPTS timemory::timemory-compile-options)
endif()

add_library(libex_python_bindings_flags INTERFACE)
check_cxx_compiler_flag("-Wno-range-loop-analysis" cxx_no_range_loop_analysis)
check_cxx_compiler_flag("-Wno-unused-value" cxx_no_unused_value)

if(cxx_no_range_loop_analysis)
    target_compile_options(libex_python_bindings_flags INTERFACE -Wno-range-loop-analysis)
endif()
if(cxx_no_unused_value)
    target_compile_options(libex_python_bindings_flags INTERFACE -Wno-unused-value)
endif()

target_compile_definitions(libex_python_bindings PRIVATE USE_MPI)
target_link_libraries(
    libex_python_bindings PRIVATE MPI::MPI_C MPI::MPI_CXX ${_OMP} ${_COMPILE_OPTS}
                                  libex_python_bindings_flags)

set_target_properties(libex_python_bindings PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

configure_file(${PROJECT_SOURCE_DIR}/ex_bindings.py
               ${TARGET_BINARY_DIR}/ex_python_bindings @ONLY)

install(
    TARGETS libex_python_bindings
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)

install(
    FILES ${TARGET_BINARY_DIR}/ex_python_bindings
    DESTINATION bin
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
    COMPONENT examples
    OPTIONAL)
