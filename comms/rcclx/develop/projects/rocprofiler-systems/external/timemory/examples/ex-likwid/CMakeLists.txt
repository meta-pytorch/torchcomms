cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_USE_LIKWID)
    return()
endif()

project(timemory-LIKWID-Example LANGUAGES CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-likwid-example)
set(COMPONENTS compile-options analysis-tools likwid)
find_package(
    timemory REQUIRED
    COMPONENTS ${COMPONENTS}
    OPTIONAL_COMPONENTS cxx)

add_executable(ex_likwid ex_likwid.cpp)
target_link_libraries(ex_likwid timemory-likwid-example)
install(
    TARGETS ex_likwid
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

configure_file(${PROJECT_SOURCE_DIR}/ex_likwid.py ${CMAKE_BINARY_DIR}/ex_python_likwid
               @ONLY)

install(
    FILES ${CMAKE_BINARY_DIR}/ex_python_likwid
    DESTINATION bin
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
    COMPONENT examples
    OPTIONAL)
