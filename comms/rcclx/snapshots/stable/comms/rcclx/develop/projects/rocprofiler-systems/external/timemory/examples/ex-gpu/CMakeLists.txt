cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

set(LANGS C CXX CUDA)
# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "CUDA" IN_LIST LANGUAGES
       OR NOT CMAKE_CUDA_COMPILER
       OR NOT TIMEMORY_USE_CUDA)
        if(TIMEMORY_USE_HIP)
            set(LANGS C CXX)
        else()
            return()
        endif()
    endif()
endif()

project(timemory-GPU-Examples LANGUAGES ${LANGS})

option(USE_LOGGING "Enable logging in debug mode" OFF)

add_library(logging INTERFACE)
target_compile_definitions(
    logging INTERFACE $<IF:$<BOOL:${USE_LOGGING}>,$<$<CONFIG:Debug>:DEBUG>,>)

add_library(timemory-gpu-example INTERFACE)
set(timemory_FIND_COMPONENTS_INTERFACE timemory-gpu-example)

find_package(
    timemory REQUIRED
    COMPONENTS threading compile-options analysis-tools
    OPTIONAL_COMPONENTS cxx cuda cuda-compiler cudart nvtx papi hip hip-device)

if(NOT CMAKE_CUDA_COMPILER)
    file(GLOB sources "${PROJECT_SOURCE_DIR}/*.cu")
    set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
endif()

if(TIMEMORY_BUILD_GPU_EXAMPLES)
    add_executable(ex_gpu_event ex_gpu_event.cu)
    target_link_libraries(ex_gpu_event PRIVATE timemory-gpu-example logging)
    install(
        TARGETS ex_gpu_event
        DESTINATION bin
        OPTIONAL)
endif()

if(NOT TIMEMORY_BUILD_GPU_KERNEL_INSTRUMENT_EXAMPLES)
    return()
endif()

add_executable(ex_kernel_instrument ex_kernel_instrument.cu)
target_link_libraries(ex_kernel_instrument PRIVATE timemory-gpu-example)
install(
    TARGETS ex_kernel_instrument
    DESTINATION bin
    OPTIONAL)

add_executable(ex_kernel_instrument_v2 ex_kernel_instrument_v2.cu)
target_link_libraries(ex_kernel_instrument_v2 PRIVATE timemory-gpu-example)
install(
    TARGETS ex_kernel_instrument_v2
    DESTINATION bin
    OPTIONAL)

add_subdirectory(v3)
