cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_BUILD_C)
    return()
endif()

project(timemory-C-Example LANGUAGES C)

set(timemory_FIND_COMPONENTS_INTERFACE timemory::c-example)
find_package(
    timemory REQUIRED
    COMPONENTS c
    OPTIONAL_COMPONENTS compile-options analysis-tools)

set(EXE_NAME ex_c)
add_executable(${EXE_NAME} ${EXE_NAME}.c)
target_link_libraries(${EXE_NAME} timemory::c-example)
install(
    TARGETS ${EXE_NAME}
    DESTINATION bin
    OPTIONAL)

set(EXE_NAME ex_c_timing)
add_executable(${EXE_NAME} ${EXE_NAME}.c)
target_link_libraries(${EXE_NAME} timemory::c-example)
install(
    TARGETS ${EXE_NAME}
    DESTINATION bin
    OPTIONAL)
