cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_USE_CALIPER)
    return()
endif()

project(timemory-Caliper-Example LANGUAGES CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-caliper-example)
set(COMPONENTS compile-options analysis-tools caliper)
find_package(
    timemory REQUIRED
    COMPONENTS ${COMPONENTS}
    OPTIONAL_COMPONENTS cxx)

add_executable(ex_caliper ex_caliper.cpp)
target_link_libraries(ex_caliper timemory-caliper-example)
install(
    TARGETS ex_caliper
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

configure_file(${PROJECT_SOURCE_DIR}/ex_caliper.py ${CMAKE_BINARY_DIR}/ex_python_caliper
               @ONLY)

install(
    FILES ${CMAKE_BINARY_DIR}/ex_python_caliper
    DESTINATION bin
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
    COMPONENT examples
    OPTIONAL)
