cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(timemory-Statistics-Example LANGUAGES C CXX)

# this example manipulates global type-traits which significantly change the
# implementation of the components. Thus, this example cannot link to the pre-compiled
# library w/ extern component templates
set(timemory_FIND_COMPONENTS_INTERFACE timemory-statistics-example)
set(COMPONENTS headers compile-options analysis-tools OPTIONAL_COMPONENTS mpi core common)

option(USE_PAPI "Enable PAPI support" OFF)
option(USE_CALIPER "Enable Caliper support" OFF)
if(USE_PAPI)
    list(APPEND COMPONENTS papi)
endif()
if(USE_CALIPER)
    list(APPEND COMPONENTS caliper)
endif()
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

# headers are always available
add_executable(ex_cxx_statistics ex_statistics.cpp)
target_link_libraries(ex_cxx_statistics timemory-statistics-example)
install(
    TARGETS ex_cxx_statistics
    DESTINATION bin
    OPTIONAL)
