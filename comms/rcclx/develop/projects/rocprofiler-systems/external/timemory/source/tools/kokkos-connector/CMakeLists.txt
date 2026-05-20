cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_BUILD_KOKKOS_TOOLS)
    return()
endif()

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(STATUS "Error! Building from the source directory may overwrite Makefile")
    message(
        STATUS "Remove 'CMakeCache.txt' and 'CMakeFiles' and build in separate directory")
    message(FATAL_ERROR "In-source build")
endif()

set(timemory_SOURCE_DIR ${PROJECT_SOURCE_DIR})

#
# Project declaration
#
project(timemory-kokkos-connector LANGUAGES C CXX)

include(CheckLanguage)
include(GNUInstallDirs)
include(CMakeParseArguments)

# if built in kokkos-tools or in timemory
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME kokkos)
set(TIMEMORY_KOKKOS_TOOLS_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/timemory/kokkos-tools)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos-tools)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos-tools)

set(LINKER_LANGUAGE CXX)
set(CUDA_AVAILABLE OFF)
if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(LINKER_LANGUAGE CUDA)
    set(CUDA_AVAILABLE ON)
endif()
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()
if(NOT CMAKE_VERSION VERSION_LESS 3.18)
    cmake_policy(SET CMP0104 OLD)
endif()
set(BUILD_SHARED_LIBS ON)
set(BUILD_STATIC_LIBS OFF)
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "CXX language standard")
set(CMAKE_CXX_STANDARD_REQUIRED
    ON
    CACHE BOOL "CXX language flags required")
set(CMAKE_CUDA_STANDARD
    17
    CACHE STRING "CUDA language standard")
set(CMAKE_CUDA_STANDARD_REQUIRED
    ON
    CACHE BOOL "CUDA language flags required")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(TIMEMORY_KOKKOS_COMPONENTS
    ""
    CACHE STRING "Explicit list of timemory components")
set(TIMEMORY_BUILD_KOKKOS_CONNECTORS
    "all"
    CACHE STRING "List of connector configurations to build")

#
# Options
#
timemory_add_option(TIMEMORY_BUILD_KOKKOS_SAMPLE "Build sample test" OFF NO_FEATURE)
timemory_add_option(TIMEMORY_BUILD_KOKKOS_CUDA_CONNECTORS
                    "Compile GPU connectors as CUDA if necessary" ON NO_FEATURE)
timemory_add_option(TIMEMORY_BUILD_KOKKOS_CONFIG "Build various connector configurations"
                    OFF)
timemory_add_feature(TIMEMORY_BUILD_KOKKOS_CONNECTORS
                     "List of connector configurations to build" DOC)
string(TOUPPER "${TIMEMORY_BUILD_KOKKOS_CONNECTORS}" _CONNECTORS)

message(STATUS "")
#
# Configure common target
#
set(timemory_INTERFACE_LIBRARY)
foreach(_COMP headers vector dmp cxx mpip-library ompt-library ncclp-library
              ${TIMEMORY_KOKKOS_COMPONENTS})
    if(TARGET timemory::timemory-${_COMP})
        list(APPEND timemory_INTERFACE_LIBRARY timemory::timemory-${_COMP})
    endif()
endforeach()

#
# Function for common routines on libraries Any additional args are treated as libraries
#
function(configure_library _NAME)
    target_include_directories(${_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(${_NAME} PRIVATE ${timemory_INTERFACE_LIBRARY} ${ARGN})
    set_target_properties(${_NAME} PROPERTIES PREFIX "" LINKER_LANGUAGE
                                                        ${LINKER_LANGUAGE})
    install(
        TARGETS ${_NAME}
        DESTINATION ${TIMEMORY_KOKKOS_TOOLS_INSTALL_DIR}
        OPTIONAL)
endfunction()

add_custom_target(timemory-kokkos-tools)

#
# Build object library
#
add_library(
    kp_timemory_common OBJECT
    ${TIMEMORY_EXCLUDE_FROM_ALL} ${PROJECT_SOURCE_DIR}/kp_timemory_common.cpp
    ${PROJECT_SOURCE_DIR}/kp_timemory.hpp)

target_link_libraries(
    kp_timemory_common
    PUBLIC timemory::timemory-dmp timemory::timemory-threading timemory::timemory-headers
           timemory::timemory-compile-options timemory::timemory-develop-options
           ${timemory_INTERFACE_LIBRARY})

set(timemory_COMMON_SOURCES $<TARGET_OBJECTS:kp_timemory_common>)

#
# Build generic library with hidden symbols
#
add_library(
    kp_timemory SHARED
    ${TIMEMORY_EXCLUDE_FROM_ALL} ${PROJECT_SOURCE_DIR}/kp_timemory.cpp
    ${PROJECT_SOURCE_DIR}/kp_timemory.hpp ${PROJECT_SOURCE_DIR}/kp_timemory_common.cpp)
target_include_directories(kp_timemory PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(kp_timemory PRIVATE ${timemory_INTERFACE_LIBRARY})
target_link_libraries(
    kp_timemory
    PRIVATE timemory::timemory-dmp timemory::timemory-threading
            timemory::timemory-headers timemory::timemory-compile-options
            timemory::timemory-develop-options timemory::timemory-hidden-visibility)
target_compile_definitions(kp_timemory PRIVATE FORCE_HIDDEN_VISIBILITY)
set_target_properties(kp_timemory PROPERTIES PREFIX "" LINKER_LANGUAGE ${LINKER_LANGUAGE})
install(
    TARGETS kp_timemory
    DESTINATION ${TIMEMORY_KOKKOS_TOOLS_INSTALL_DIR}
    OPTIONAL)
add_dependencies(timemory-kokkos-tools kp_timemory)

#
# Build regex filtering library
#
add_library(
    kp_timemory_filter SHARED
    ${TIMEMORY_EXCLUDE_FROM_ALL} ${PROJECT_SOURCE_DIR}/kp_timemory_filter.cpp
    ${timemory_COMMON_SOURCES})
configure_library(kp_timemory_filter)
add_dependencies(timemory-kokkos-tools kp_timemory_filter)

#
# Generate a test using kokkos
#
function(ADD_KOKKOS_SAMPLE_TEST _NAME _TARG)
    if(NOT TIMEMORY_BUILD_KOKKOS_SAMPLE OR NOT TARGET kokkos-connector-sample)
        return()
    endif()

    string(REPLACE "_" "-" _NAME "${_NAME}")
    add_test(
        NAME kokkos-connector-test-${_NAME}
        COMMAND $<TARGET_FILE:kokkos-connector-sample>
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set(_TEST_ENV
        "KOKKOS_PROFILE_LIBRARY=$<TARGET_FILE:${_TARG}>"
        "OMP_PROC_BIND=spread"
        "OMP_PLACES=threads"
        "OMP_NUM_THREADS=2"
        "KOKKOS_NUM_THREADS=2"
        "TIMEMORY_TIME_OUTPUT=OFF"
        "TIMEMORY_COUT_OUTPUT=ON"
        "CALI_CONFIG_PROFILE=runtime-report")
    set_tests_properties(
        kokkos-connector-test-${_NAME}
        PROPERTIES
            PROCESSORS
            2
            LABELS
            "kokkos"
            ENVIRONMENT
            "${_TEST_ENV}"
            PASS_REGULAR_EXPRESSION
            ">>> kokkos/allocate/.*>>> kokkos/deep_copy/.*>>> kokkos/deallocate/.*(>>> kokkos/(Kokkos::|)parallel_reduce|)"
        )
endfunction()

#
# Build the sample (test executable)
#
if(TIMEMORY_BUILD_KOKKOS_SAMPLE)
    find_package(Kokkos 3.1 REQUIRED)
    add_executable(kokkos-connector-sample ${TIMEMORY_EXCLUDE_FROM_ALL} sample/sample.cpp)
    set_property(TARGET kokkos-connector-sample PROPERTY RULE_LAUNCH_COMPILE "")
    set_property(TARGET kokkos-connector-sample PROPERTY RULE_LAUNCH_LINK "")
    # use launch_compiler feature to ensure kokkos uses it's original compiler
    kokkos_compilation(TARGET kokkos-connector-sample)
    target_compile_definitions(
        kokkos-connector-sample
        PRIVATE -DKOKKOS_PROFILE_LIBRARY="$<TARGET_FILE:kp_timemory>")
    target_link_libraries(kokkos-connector-sample PRIVATE Kokkos::kokkos)
    set_target_properties(kokkos-connector-sample PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                             ${CMAKE_CURRENT_BINARY_DIR})
    add_dependencies(timemory-kokkos-tools kokkos-connector-sample)
    # kokkos_compilation(TARGET kokkos-connector-sample)
    add_kokkos_sample_test(libtimemory timemory::timemory-cxx)
    add_kokkos_sample_test(kp-timemory kp_timemory)
    add_kokkos_sample_test(kp-timemory-filter kp_timemory_filter)
endif()

# --------------------------------------------------------------------------------------##

if(timemory_MAIN_PROJECT)
    if(NOT TIMEMORY_BUILD_KOKKOS_CONFIG)
        timemory_message(STATUS "")
        timemory_message(
            STATUS
            "Disabled building explicit configurations with 'TIMEMORY_BUILD_KOKKOS_CONFIG=OFF'"
            )
        timemory_message(STATUS "")
    else()
        timemory_message(STATUS "")
        timemory_message(STATUS
                         "Building explicit connector configurations: ${_CONNECTORS}")
        timemory_message(
            STATUS
            "    Disable building explicit configurations with '-DTIMEMORY_BUILD_KOKKOS_CONFIG=OFF'"
            )
        timemory_message(
            STATUS
            "    Reduce explicit configurations with semi-colon delimited '-DTIMEMORY_BUILD_KOKKOS_CONNECTORS=\"...\"'"
            )
        timemory_message(STATUS "")
    endif()
endif()

# --------------------------------------------------------------------------------------##

set(_AVAILABLE_CONNECTOR_CONFIGS "all")

function(TIMEMORY_ADD_CONNECTOR_CONFIG _NAME)
    cmake_parse_arguments(CONNECTOR "GPU;DISABLE_COMPILE_OPTIONS" "CONDITION"
                          "COMPONENTS;TARGETS;DEFINITIONS;ENVIRONMENT" ${ARGN})

    if(NOT "${CONNECTOR_CONDITION}" STREQUAL "")
        set(_ENABLED OFF)
        if(${CONNECTOR_CONDITION})
            set(_ENABLED ON)
        endif()
        if(NOT _ENABLED)
            return()
        endif()
    endif()

    set(_AVAILABLE_CONNECTOR_CONFIGS
        ${_AVAILABLE_CONNECTOR_CONFIGS} ${_NAME}
        PARENT_SCOPE)

    if(NOT TIMEMORY_BUILD_KOKKOS_CONFIG)
        return()
    endif()

    if(NOT "${_NAME}" IN_LIST _CONNECTORS AND NOT "${_CONNECTORS}" STREQUAL "ALL")
        return()
    endif()

    if("${CONNECTOR_COMPONENTS}" STREQUAL "")
        timemory_message(
            AUTHOR_WARNING
            "Attempt to configure a kokkos-connector without any components: ${_NAME}")
    endif()

    if(CONNECTOR_GPU AND NOT CUDA_AVAILABLE)
        return()
    endif()

    # configure the extension
    set(_EXT cpp)
    if(CONNECTOR_GPU AND CUDA_AVAILABLE)
        if("${CMAKE_CXX_STANDARD}" GREATER "${CMAKE_CUDA_STANDARD}")
            if(TIMEMORY_BUILD_KOKKOS_CUDA_CONNECTORS)
                set(_MSG
                    "${_NAME} requested CUDA compilation but CMAKE_CXX_STANDARD (${CMAKE_CXX_STANDARD}) > CMAKE_CUDA_STANDARD (${CMAKE_CUDA_STANDARD})"
                    )
                set(_MSG
                    "${_MSG}. Compiling as C++... Set TIMEMORY_BUILD_KOKKOS_CUDA_CONNECTORS=OFF to skip compiling ${_NAME}"
                    )
                timemory_message(WARNING "${_MSG}")
            else()
                return()
            endif()
        else()
            set(_EXT cu)
        endif()
    endif()

    # configure the components
    string(REPLACE ";" "," _COMPONENTS "${CONNECTOR_COMPONENTS}")
    set(_ENVIRON)
    foreach(_ENV ${CONNECTOR_ENVIRONMENT})
        set(_ENVIRON "${_ENVIRON}tim::set_env(\"${_ENV}\", 1);\n")
    endforeach()
    string(REPLACE "=" "\", \"" _ENVIRON "${_ENVIRON}")
    timemory_message(STATUS "Building 'kp_timemory_${_NAME}' with '${_COMPONENTS}'...")
    set(KP_COMPONENTS ${_COMPONENTS})
    set(KP_ENVIRON ${_ENVIRON})
    set(KP_LIBNAME "kp_timemory_${_NAME}")

    set(SOURCE_FILE ${PROJECT_BINARY_DIR}/kp_timemory_${_NAME}.${_EXT})
    configure_file(${PROJECT_SOURCE_DIR}/kp_timemory.cpp.in ${SOURCE_FILE} @ONLY)

    if(NOT CONNECTOR_DISABLE_COMPILE_OPTIONS)
        list(APPEND CONNECTOR_TARGETS timemory::timemory-compile-options)
    endif()

    list(APPEND CONNECTOR_TARGETS timemory::timemory-dmp timemory::timemory-threading)

    add_library(kp_timemory_${_NAME} SHARED ${TIMEMORY_EXCLUDE_FROM_ALL} ${SOURCE_FILE}
                                            ${timemory_COMMON_SOURCES})
    configure_library(kp_timemory_${_NAME} ${CONNECTOR_TARGETS})
    target_compile_definitions(kp_timemory_${_NAME} PRIVATE ${CONNECTOR_DEFINITIONS})

    add_kokkos_sample_test(kp_timemory_${_NAME} kp_timemory_${_NAME})
    add_dependencies(timemory-kokkos-tools kp_timemory_${_NAME})
endfunction()

# --------------------------------------------------------------------------------------##

timemory_add_connector_config(
    trip_count
    COMPONENTS trip_count
    TARGETS timemory::timemory-headers)

timemory_add_connector_config(
    timers
    COMPONENTS wall_clock cpu_clock cpu_util
    TARGETS timemory::timemory-headers)

timemory_add_connector_config(
    memory
    COMPONENTS peak_rss page_rss virtual_memory current_peak_rss
    TARGETS timemory::timemory-headers)

timemory_add_connector_config(
    timers_memory
    COMPONENTS wall_clock cpu_clock cpu_util peak_rss page_rss virtual_memory
    TARGETS timemory::timemory-headers)

timemory_add_connector_config(
    io
    CONDITION "UNIX AND NOT APPLE"
    COMPONENTS read_bytes read_char written_bytes written_char
    TARGETS timemory::timemory-headers)

timemory_add_connector_config(
    context_switch
    COMPONENTS priority_context_switch voluntary_context_switch
    TARGETS timemory::timemory-headers)

timemory_add_connector_config(
    gperftools DISABLE_COMPILE_OPTIONS
    CONDITION "${TIMEMORY_USE_GPERFTOOLS}"
    COMPONENTS gperftools_cpu_profiler gperftools_heap_profiler
    TARGETS timemory::timemory-gperftools)

timemory_add_connector_config(
    papi
    CONDITION "${TIMEMORY_USE_PAPI}"
    COMPONENTS papi_array_t
    TARGETS timemory::timemory-papi-component)

timemory_add_connector_config(
    cpu_roofline
    CONDITION "${TIMEMORY_USE_PAPI}"
    COMPONENTS cpu_roofline_flops
    TARGETS timemory::timemory-ert timemory::timemory-cpu-roofline
            timemory::timemory-papi-component timemory::timemory-roofline-component)

if(timemory_SOURCE_DIR)
    execute_process(
        COMMAND ${timemory_SOURCE_DIR}/scripts/papi-avail-presets.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        ERROR_VARIABLE ERR_MSG
        OUTPUT_VARIABLE PAPI_AVAILABLE_PRESETS
        RESULT_VARIABLE RET)
    if(RET GREATER 0)
        timemory_message(
            STATUS
            "${timemory_SOURCE_DIR}/scripts/papi-avail-presets.sh exited with ${RET}:\n${ERR_MSG}"
            )
    else()
        timemory_message(STATUS "PAPI available presets: ${PAPI_AVAILABLE_PRESETS}")
    endif()
endif()

# creates a normal connector config and a rates version
function(TIMEMORY_ADD_PAPI_CONNECTOR_CONFIG _NAME)
    cmake_parse_arguments(CONNECTOR "" "" "COMPONENTS;TARGETS;DEFINITIONS;ENVIRONMENT"
                          ${ARGN})
    set(_ORIG_COMPONENTS "${CONNECTOR_COMPONENTS}")
    if(PAPI_AVAILABLE_PRESETS AND "${CONNECTOR_COMPONENTS}" MATCHES "papi_tuple")
        string(REPLACE "papi_tuple<" "" _COMPONENTS "${CONNECTOR_COMPONENTS}")
        string(REPLACE ">" "" _COMPONENTS "${_COMPONENTS}")
        string(REPLACE "," ";" _COMPONENTS "${_COMPONENTS}")
        foreach(_COMP ${_COMPONENTS})
            if(NOT "${_COMP}" IN_LIST PAPI_AVAILABLE_PRESETS)
                string(REPLACE ",${_COMP}" "" CONNECTOR_COMPONENTS
                               "${CONNECTOR_COMPONENTS}")
                string(REPLACE "${_COMP}," "" CONNECTOR_COMPONENTS
                               "${CONNECTOR_COMPONENTS}")
                string(REPLACE "${_COMP}" "" CONNECTOR_COMPONENTS
                               "${CONNECTOR_COMPONENTS}")
            endif()
        endforeach()
    endif()
    if(CONNECTOR_COMPONENTS STREQUAL "papi_tuple<>")
        timemory_message(
            STATUS
            "Skipping '${_NAME}' with '${_ORIG_COMPONENTS}': all presets are unavailable..."
            )
        return()
    endif()
    timemory_add_connector_config(
        ${_NAME}
        CONDITION "${TIMEMORY_USE_PAPI}"
        COMPONENTS ${CONNECTOR_COMPONENTS}
        TARGETS ${CONNECTOR_TARGETS}
        DEFINITIONS ${CONNECTOR_DEFINITIONS}
        ENVIRONMENT ${CONNECTOR_ENVIRONMENT}
        TARGETS timemory::timemory-papi)
    string(REPLACE "papi_tuple<" "papi_rate_tuple<wall_clock," CONNECTOR_COMPONENTS
                   "${CONNECTOR_COMPONENTS}")
    timemory_add_connector_config(
        ${_NAME}_wall_rate
        CONDITION "${TIMEMORY_USE_PAPI}"
        COMPONENTS ${CONNECTOR_COMPONENTS}
        TARGETS ${CONNECTOR_TARGETS}
        DEFINITIONS ${CONNECTOR_DEFINITIONS}
        ENVIRONMENT ${CONNECTOR_ENVIRONMENT}
        TARGETS timemory::timemory-papi)
    set(_AVAILABLE_CONNECTOR_CONFIGS
        "${_AVAILABLE_CONNECTOR_CONFIGS}"
        PARENT_SCOPE)
endfunction()

timemory_add_papi_connector_config(
    cpu_flops
    COMPONENTS papi_tuple<PAPI_DP_OPS,PAPI_SP_OPS>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_instructions
    COMPONENTS papi_tuple<PAPI_TOT_INS,PAPI_LD_INS,PAPI_SR_INS>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_cache_misses
    COMPONENTS papi_tuple<PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_tlb
    COMPONENTS papi_tuple<PAPI_TLB_DM,PAPI_TLB_IM,PAPI_TLB_TL,PAPI_TLB_SD>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_branching
    COMPONENTS
        papi_tuple<PAPI_BR_UCN,PAPI_BR_CN,PAPI_BR_TKN,PAPI_BR_NTK,PAPI_BR_MSP,PAPI_BR_PRC,PAPI_BR_INS>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_stalls
    COMPONENTS
        papi_tuple<PAPI_MEM_SCY,PAPI_MEM_RCY,PAPI_MEM_WCY,PAPI_RES_STL,PAPI_FP_STAL>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_l1_misses
    COMPONENTS papi_tuple<PAPI_L1_DCM,PAPI_L1_ICM,PAPI_L1_TCM,PAPI_L1_LDM,PAPI_L1_STM>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_l2_misses
    COMPONENTS papi_tuple<PAPI_L2_DCM,PAPI_L2_ICM,PAPI_L2_TCM,PAPI_L2_LDM,PAPI_L2_STM>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_l3_misses
    COMPONENTS papi_tuple<PAPI_L3_DCM,PAPI_L3_ICM,PAPI_L3_TCM,PAPI_L3_LDM,PAPI_L3_STM>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_l1_total
    COMPONENTS papi_tuple<PAPI_L1_TCM,PAPI_L1_TCH,PAPI_L1_TCA,PAPI_L1_TCR,PAPI_L1_TCW>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_l2_total
    COMPONENTS papi_tuple<PAPI_L2_TCM,PAPI_L2_TCH,PAPI_L2_TCA,PAPI_L2_TCR,PAPI_L2_TCW>
    TARGETS timemory::timemory-papi)

timemory_add_papi_connector_config(
    cpu_l3_total
    COMPONENTS papi_tuple<PAPI_L3_TCM,PAPI_L3_TCH,PAPI_L3_TCA,PAPI_L3_TCR,PAPI_L3_TCW>
    TARGETS timemory::timemory-papi)

timemory_add_connector_config(
    cuda_profiler
    CONDITION "${TIMEMORY_USE_CUDA}"
    COMPONENTS nvtx_marker
    TARGETS timemory::timemory-cuda-component)

timemory_add_connector_config(
    cuda_event
    CONDITION "${TIMEMORY_USE_CUDA}"
    COMPONENTS cuda_event
    TARGETS timemory::timemory-cuda-component)

timemory_add_connector_config(
    cuda_activity
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_activity
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    cuda_hw_counters
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_roofline
    CONDITION "${TIMEMORY_USE_CUPTI}"
    GPU
    COMPONENTS gpu_roofline_flops
    TARGETS timemory::timemory-ert timemory::timemory-gpu-roofline
            timemory::timemory-cupti-component timemory::timemory-roofline-component)

timemory_add_connector_config(
    gpu_instructions
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT
        "TIMEMORY_CUPTI_METRICS=inst_issued,inst_executed,thread_inst_executed,ldst_executed,inst_integer,inst_fp_16,inst_fp_32,inst_fp_64,inst_inter_thread_communication"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_cache_misses
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT
        "TIMEMORY_CUPTI_METRICS=tex_cache_transactions,tex_cache_hit_rate,tex0_cache_sector_queries,tex0_cache_sector_misses,tex1_cache_sector_queries,tex1_cache_sector_misses,l2_tex_hit_rate,dram_read_bytes,dram_write_bytes"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_branching
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT "TIMEMORY_CUPTI_METRICS=divergent_branch,branch,branch_efficiency"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_stalls
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT
        "TIMEMORY_CUPTI_METRICS=stall_inst_fetch,stall_exec_dependency,stall_memory_dependency,stall_texture,stall_sync,stall_other,stall_constant_memory_dependency,stall_pipe_busy,stall_memory_throttle,stall_not_selected"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_l2_total
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT
        "TIMEMORY_CUPTI_METRICS=l2_read_transactions,l2_write_transactions,l2_atomic_transactions,l2_global_load_bytes,l2_local_load_bytes,l2_local_global_store_bytes,l2_global_reduction_bytes,l2_global_atomic_store_bytes"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_efficiency
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT
        "TIMEMORY_CUPTI_METRICS=branch_efficiency,warp_execution_efficiency,warp_nonpred_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency,sm_efficiency,flop_hp_efficiency,flop_sp_efficiency,flop_dp_efficiency"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_ipc
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT "TIMEMORY_CUPTI_METRICS=ipc,issued_ipc"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_warps
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT
        "TIMEMORY_CUPTI_METRICS=warps_launched,active_warps,inst_per_warp,warp_execution_efficiency,warp_nonpred_execution_efficiency,unique_warps_launched,eligible_warps_per_cycle"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_double_flops
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT
        "TIMEMORY_CUPTI_METRICS=flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,flop_dp_efficiency"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_single_flops
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT
        "TIMEMORY_CUPTI_METRICS=flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_sp_efficiency"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    gpu_half_flops
    CONDITION "${TIMEMORY_USE_CUPTI}"
    COMPONENTS cupti_counters
    ENVIRONMENT
        "TIMEMORY_CUPTI_METRICS=flop_count_hp,flop_count_hp_add,flop_count_hp_mul,flop_count_hp_fma,flop_hp_efficiency"
    TARGETS timemory::timemory-cupti-component)

timemory_add_connector_config(
    vtune
    CONDITION "${TIMEMORY_USE_VTUNE}"
    COMPONENTS vtune_profiler vtune_frame vtune_event
    TARGETS timemory::timemory-vtune)

timemory_add_connector_config(
    caliper
    CONDITION "${TIMEMORY_USE_CALIPER}"
    COMPONENTS caliper_config caliper_marker
    TARGETS timemory::timemory-caliper)

timemory_add_connector_config(
    likwid
    CONDITION "${TIMEMORY_USE_LIKWID}"
    COMPONENTS likwid_marker likwid_nvmarker
    TARGETS timemory::timemory-likwid)

timemory_add_connector_config(
    tau
    CONDITION "${TIMEMORY_USE_TAU}"
    COMPONENTS tau_marker
    TARGETS timemory::timemory-tau)

set(USE_OMPT ${TIMEMORY_USE_OMPT})
if(TARGET timemory::timemory-ompt-shared)
    set(_OMPT_LIB timemory::timemory-ompt-shared)
else()
    set(USE_OMPT OFF)
endif()

timemory_add_connector_config(
    ompt
    CONDITION "${USE_OMPT}"
    COMPONENTS ompt_handle<tim::project::timemory>
    DEFINITIONS TIMEMORY_USE_OMPT_LIBRARY
    TARGETS ${_OMPT_LIB})

set_property(CACHE TIMEMORY_BUILD_KOKKOS_CONNECTORS
             PROPERTY STRINGS "${_AVAILABLE_CONNECTOR_CONFIGS}")
