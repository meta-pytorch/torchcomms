cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_BUILD_FORTRAN)
    return()
endif()

project(timemory-Fortran-Example LANGUAGES Fortran)

set(EXE_NAME ex_fortran)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

set(timemory_FIND_COMPONENTS_INTERFACE timemory::fortran-example)
find_package(timemory REQUIRED COMPONENTS fortran)

add_executable(${EXE_NAME} ${EXE_NAME}.f90)
target_link_libraries(${EXE_NAME} timemory::fortran-example)
install(
    TARGETS ${EXE_NAME}
    DESTINATION bin
    OPTIONAL)
