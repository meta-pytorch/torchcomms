cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(timemory-Compiler-Instrument-Example LANGUAGES CXX)

set(timemory_FIND_COMPONENTS_INTERFACE compiler-instrument-example)
find_package(timemory OPTIONAL_COMPONENTS compiler-instrument)

if(NOT timemory_FOUND OR NOT TARGET timemory::timemory-compiler-instrument)
    return()
endif()

add_executable(ex_compiler_instrument ex_compiler_instrument.cpp)

target_link_libraries(ex_compiler_instrument PRIVATE compiler-instrument-example)

install(
    TARGETS ex_compiler_instrument
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)
