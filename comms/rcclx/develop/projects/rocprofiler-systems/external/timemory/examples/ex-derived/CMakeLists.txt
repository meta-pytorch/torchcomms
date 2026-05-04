cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(timemory-Derived-Example LANGUAGES CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-derived-example)
set(COMPONENTS headers compile-options analysis-tools OPTIONAL_COMPONENTS cxx)

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

# headers are always available
add_executable(ex_derived ex_derived.cpp)
target_link_libraries(ex_derived timemory-derived-example)
install(
    TARGETS ex_derived
    DESTINATION bin
    OPTIONAL)
