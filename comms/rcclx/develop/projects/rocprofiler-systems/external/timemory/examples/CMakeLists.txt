cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

cmake_policy(PUSH)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

# ----------------------------------------------------------------------------------------#
# timemory examples project
#
project(timemory-Examples C CXX)

# ----------------------------------------------------------------------------------------#
# Add the examples
#

# disable unity builds
set(CMAKE_UNITY_BUILD OFF)

add_subdirectory(ex-minimal)
add_subdirectory(ex-derived)
add_subdirectory(ex-optional)
add_subdirectory(ex-cxx-basic)
add_subdirectory(ex-cxx-tuple)
add_subdirectory(ex-statistics)
add_subdirectory(ex-cxx-overhead)
add_subdirectory(ex-array-of-bundles)

if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    set(DEFAULT_CHOICE ON)
    set(INVERSE_CHOICE OFF)
else()
    set(DEFAULT_CHOICE OFF)
    set(INVERSE_CHOICE ON)
endif()

option(TIMEMORY_BUILD_C_EXAMPLES "Enable building these examples" ${DEFAULT_CHOICE})
option(TIMEMORY_BUILD_TPL_EXAMPLES "Enable building these examples" ${DEFAULT_CHOICE})
option(TIMEMORY_BUILD_GPU_EXAMPLES "Enable building these examples" ${DEFAULT_CHOICE})
option(TIMEMORY_BUILD_PYTHON_EXAMPLES "Enable building these examples" ${DEFAULT_CHOICE})
option(TIMEMORY_BUILD_FORTRAN_EXAMPLES "Enable building these examples" ${DEFAULT_CHOICE})
option(TIMEMORY_BUILD_ROOFLINE_EXAMPLES "Enable building these examples"
       ${DEFAULT_CHOICE})
option(TIMEMORY_BUILD_DYNAMIC_INSTRUMENT_EXAMPLES "Enable building these examples"
       ${DEFAULT_CHOICE})
option(TIMEMORY_BUILD_COMPILER_INSTRUMENT_EXAMPLES "Enable building these examples"
       ${DEFAULT_CHOICE})
option(TIMEMORY_BUILD_GPU_KERNEL_INSTRUMENT_EXAMPLES
       "Enable building GPU kernel instrumentation examples" ${DEFAULT_CHOICE})
option(TIMEMORY_BUILD_COMPONENT_EXAMPLES "Enable building these examples"
       ${INVERSE_CHOICE})

# C language related
if(TIMEMORY_BUILD_C_EXAMPLES)
    add_subdirectory(ex-c)
endif()

if(TIMEMORY_BUILD_FORTRAN_EXAMPLES)
    add_subdirectory(ex-fortran)
endif()

# external package related
if(TIMEMORY_BUILD_TPL_EXAMPLES)
    add_subdirectory(ex-caliper)
    add_subdirectory(ex-gotcha)
    add_subdirectory(ex-likwid)
endif()

# GPU related
if(TIMEMORY_BUILD_GPU_EXAMPLES OR TIMEMORY_BUILD_GPU_KERNEL_INSTRUMENT_EXAMPLES)
    add_subdirectory(ex-gpu)
endif()

# Python related
if(TIMEMORY_BUILD_PYTHON_EXAMPLES)
    add_subdirectory(ex-python)
endif()

# Roofline related
if(TIMEMORY_BUILD_ROOFLINE_EXAMPLES)
    add_subdirectory(ex-ert)
    add_subdirectory(ex-cpu-roofline)
    add_subdirectory(ex-gpu-roofline)
endif()

# Dynamic instrumentation
if(TIMEMORY_BUILD_DYNAMIC_INSTRUMENT_EXAMPLES)
    add_subdirectory(ex-custom-dynamic-instr)
endif()

# Compiler instrumentation
if(TIMEMORY_BUILD_COMPILER_INSTRUMENT_EXAMPLES)
    add_subdirectory(ex-compiler-instrument)
endif()

# Header-only vs. library
if(TIMEMORY_BUILD_COMPONENT_EXAMPLES)
    add_subdirectory(ex-cxx-header-only)
endif()

cmake_policy(POP)
