cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(timemory-CXX-Overhead-Example LANGUAGES C CXX)

# set this locally to a release build so that the fibonacci calcs are always optimized
set(CMAKE_BUILD_TYPE "Release")
set(EXE_NAME ex_cxx_overhead)
set(COMPONENTS
    compile-options
    analysis-tools
    OPTIONAL_COMPONENTS
    arch
    core
    common
    papi
    rusage-component
    trip_count-component)

option(USE_CALIPER "Enable caliper" OFF)
option(USE_MPI "Enable MPI" OFF)
option(USE_TAU "Enable TAU" OFF)

set(USE_CALIPER_COMPONENTS caliper)
set(USE_MPI_COMPONENTS mpi)
set(USE_TAU_COMPONENTS tau)

foreach(_OPT CALIPER MPI TAU)
    if(USE_${_OPT})
        list(APPEND COMPONENTS ${USE_${_OPT}_COMPONENTS})
    endif()
endforeach()

set(timemory_FIND_COMPONENTS_INTERFACE timemory-cxx-overhead-example)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory-cxx-overhead-example)
install(
    TARGETS ${EXE_NAME}
    DESTINATION bin
    OPTIONAL)
