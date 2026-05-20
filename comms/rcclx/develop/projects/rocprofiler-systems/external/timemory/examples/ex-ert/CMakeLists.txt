cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

include(CheckLanguage)

# travis has limited resources so running ERT takes a while if("$ENV{USER}" STREQUAL
# "travis" AND "$ENV{CONTINUOUS_INTEGRATION}" STREQUAL "true") return() endif()

set(_USE_CUDA ON)
# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    if(DEFINED BUILD_ERT AND NOT BUILD_ERT)
        return()
    endif()
    if(NOT TIMEMORY_BUILD_ERT)
        return()
    endif()
    if("$ENV{CI}" STREQUAL "true" AND TIMEMORY_BUILD_LTO)
        return()
    endif()
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "CUDA" IN_LIST LANGUAGES
       OR NOT CMAKE_CUDA_COMPILER
       OR NOT TIMEMORY_USE_CUDA)
        set(_USE_CUDA OFF)
        if(DEFINED USE_CUDA)
            set(USE_CUDA OFF)
        endif()
    endif()
endif()

project(timemory-ERT-Example LANGUAGES CXX)

option(USE_CUDA "Enable CUDA" ${_USE_CUDA})
option(USE_KOKKOS "Enable Kokkos" OFF)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-ert-example)
set(COMPONENTS
    headers
    compile-options
    vector
    analysis-tools
    OPTIONAL_COMPONENTS
    mpi
    cuda
    cuda-compiler
    ert
    cxx)

if(USE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        set(_USE_CUDA OFF)
    endif()
endif()

if(USE_CUDA)
    list(APPEND COMPONENTS cuda)
else()
    list(APPEND COMPONENTS hip hip-device)
endif()

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})
if(NOT TARGET timemory::timemory-ert)
    return()
endif()

add_executable(ex_ert ex_ert.cpp)
target_link_libraries(ex_ert timemory-ert-example)
install(
    TARGETS ex_ert
    DESTINATION bin
    COMPONENT examples
    OPTIONAL)

if(USE_CUDA OR TIMEMORY_USE_CUDA)
    set_source_files_properties(ex_ert.cpp PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)
    target_compile_definitions(ex_ert PRIVATE USE_GPU)
elseif(TIMEMORY_USE_HIP)
    target_compile_definitions(ex_ert PRIVATE USE_GPU)
endif()

if(TIMEMORY_BUILD_KOKKOS_SAMPLE)
    find_package(Kokkos 3.1)
    if(Kokkos_FOUND AND USE_KOKKOS)
        add_executable(ex_ert_kokkos ex_ert_kokkos.cpp)
        target_link_libraries(ex_ert_kokkos timemory-ert-example Kokkos::kokkos)
        install(
            TARGETS ex_ert_kokkos
            DESTINATION bin
            COMPONENT examples
            OPTIONAL)

        if(USE_CUDA OR TIMEMORY_USE_CUDA)
            set_source_files_properties(ex_ert_kokkos.cpp PROPERTIES LANGUAGE CUDA
                                                                     LINKER_LANGUAGE CUDA)
            target_compile_definitions(ex_ert_kokkos PRIVATE USE_CUDA)
        endif()
    endif()
endif()
