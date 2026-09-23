cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if(WIN32 OR ("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_USE_GOTCHA))
    return()
endif()

project(timemory-gotcha-Example LANGUAGES C CXX)

include(GNUInstallDirs)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-gotcha-example)
find_package(
    timemory REQUIRED
    COMPONENTS headers gotcha
    OPTIONAL_COMPONENTS cxx)

add_library(ex_gotcha_lib SHARED ex_gotcha_lib.hpp ex_gotcha_lib.cpp)
target_link_libraries(ex_gotcha_lib PUBLIC timemory-gotcha-example)

add_executable(ex_gotcha ex_gotcha.cpp)
target_link_libraries(ex_gotcha PRIVATE ex_gotcha_lib)

add_executable(ex_gotcha_replacement ex_gotcha_replacement.cpp)
target_link_libraries(ex_gotcha_replacement PRIVATE timemory::timemory-headers
                                                    timemory::timemory-gotcha)

add_library(ex_gotcha_lib_mpi SHARED ex_gotcha_lib.hpp ex_gotcha_lib.cpp)
target_link_libraries(ex_gotcha_lib_mpi PUBLIC timemory-gotcha-example
                                               timemory::timemory-mpi)

add_executable(ex_gotcha_mpi ex_gotcha.cpp)
target_link_libraries(ex_gotcha_mpi PRIVATE ex_gotcha_lib_mpi timemory::timemory-mpi)

install(
    TARGETS ex_gotcha ex_gotcha_mpi ex_gotcha_replacement
    DESTINATION bin
    OPTIONAL)

install(
    TARGETS ex_gotcha_lib ex_gotcha_lib_mpi
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)
