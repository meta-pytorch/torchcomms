# return if skipping build or no other builds were enabled
if(TIMEMORY_SKIP_BUILD
   OR (NOT TIMEMORY_BUILD_TESTING
       AND NOT TIMEMORY_BUILD_EXAMPLES
       AND NOT TIMEMORY_CI
       AND NOT TIMEMORY_BUILD_MINIMAL_TESTING
       AND NOT TIMEMORY_BUILD_GOOGLE_TEST
      ))
    return()
endif()

if(WIN32 AND NOT TIMEMORY_USE_WINSOCK)
    option(TIMEMORY_SOCKET_TESTS "socket tests" OFF)
else()
    option(TIMEMORY_SOCKET_TESTS "socket tests" ON)
endif()
mark_as_advanced(TIMEMORY_SOCKET_TESTS)

# include the current directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_UNITY_BUILD OFF)
if(NOT DEFINED CTEST_TEST_TIMEOUT OR "${CTEST_TEST_TIMEOUT}" STREQUAL "")
    set(CTEST_TEST_TIMEOUT 180)
endif()

if(TIMEMORY_USE_COVERAGE)
    add_compile_definitions(TIMEMORY_USE_COVERAGE)
endif()

set(_EXCLUDE)
if(NOT TIMEMORY_BUILD_TESTING OR TIMEMORY_BUILD_EXCLUDE_FROM_ALL)
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

if(NOT WIN32)
    unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    unset(CMAKE_PDB_OUTPUT_DIRECTORY)
endif()

set(DEVICE_LANGUAGE CXX)
if(TIMEMORY_USE_CUDA)
    set(DEVICE_LANGUAGE CUDA)
endif()

option(TIMEMORY_USE_TESTING_LINK_LIBRARY "Google-tests like to shared or static library"
       ON)
mark_as_advanced(TIMEMORY_USE_TESTING_LINK_LIBRARY)

if(TARGET timemory::timemory-cxx-shared)
    set(_LIBRARY_TARGET timemory::timemory-cxx-shared)
elseif(TARGET timemory::timemory-cxx-static)
    set(_LIBRARY_TARGET timemory::timemory-cxx-static)
endif()

set(_LIBRARY)
if(TIMEMORY_USE_TESTING_LINK_LIBRARY)
    set(_LIBRARY ${_LIBRARY_TARGET})
endif()

if(TIMEMORY_USE_OMPT)
    find_package(OpenMP QUIET)
    if(OpenMP_FOUND)
        set(_OPENMP OpenMP::OpenMP_CXX)
    endif()
endif()

if(TIMEMORY_USE_PTL OR TIMEMORY_BUILD_TESTING)
    find_package(PTL QUIET)
endif()

# add_library(no-store-merge INTERFACE) add_target_cxx_flag_if_avail(no-store-merge
# "-fno-store-merging")
add_library(test-opt-flags INTERFACE)
if(NOT TIMEMORY_USE_COVERAGE)
    string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    target_compile_options(test-opt-flags INTERFACE ${_FLAGS})
    if(NOT TIMEMORY_USE_ARCH)
        add_target_flag_if_avail(test-opt-flags "-march=native")
    endif()
endif()

# Debug flags
add_library(test-debug-flags INTERFACE)
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
target_compile_options(test-debug-flags INTERFACE ${_FLAGS})
add_target_flag_if_avail(test-debug-flags "-Og" "-g")

# Any warnings trigger an error flags
add_library(test-werror-flags INTERFACE)
add_target_flag_if_avail(test-werror-flags "-W" "-Wall" "-Wextra" "-pedantic" "-Werror")
if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_target_flag_if_avail(test-werror-flags "-Wno-mismatched-tags")
endif()
target_link_libraries(test-werror-flags INTERFACE timemory::timemory-headers
                                                  timemory::timemory-compile-timing)
target_compile_definitions(test-werror-flags INTERFACE TIMEMORY_CMAKE)

add_library(common-test-libs INTERFACE)
target_link_libraries(
    common-test-libs
    INTERFACE timemory::timemory-headers timemory::timemory-compile-options
              timemory::timemory-develop-options timemory::timemory-analysis-tools
              timemory::timemory-vector)

if(TIMEMORY_USE_COVERAGE)
    # ensures all usage updates coverage
    target_link_libraries(common-test-libs INTERFACE ${_LIBRARY_TARGET}
                                                     timemory::timemory-extensions)
endif()

set(HYBRID_COMPONENTS)

if(TIMEMORY_USE_CUDA)
    list(APPEND HYBRID_COMPONENTS timemory::timemory-cuda timemory::timemory-cudart)
endif()

if(TIMEMORY_USE_PAPI)
    list(APPEND HYBRID_COMPONENTS timemory::timemory-papi)
endif()

if(TIMEMORY_USE_CUPTI AND HAS_CUDA_DRIVER_LIBRARY)
    list(APPEND HYBRID_COMPONENTS timemory::timemory-cupti
         timemory::timemory-cudart-device)
endif()

if(TIMEMORY_USE_CALIPER)
    list(APPEND HYBRID_COMPONENTS timemory::timemory-caliper)
endif()

if(TIMEMORY_USE_VTUNE)
    list(APPEND HYBRID_COMPONENTS timemory::timemory-vtune)
endif()

set(ENV{TIMEMORY_ENABLED} OFF)

# always make sure timemory-test exists
timemory_add_timemory_test_target()

add_subdirectory(external)

foreach(_LINK_TYPE shared static)
    foreach(_DEPTH_VALUE 0 5)
        set(compiler_instr_env
            "TIMEMORY_SUPPRESS_CONFIG=ON" "TIMEMORY_COMPILER_SUPPRESS_CONFIG=ON"
            "TIMEMORY_COMPILER_COUT_OUTPUT=ON" "TIMEMORY_COMPILER_FLAT_PROFILE=ON")
        if(NOT "${_DEPTH_VALUE}" STREQUAL "0")
            set(_EXTRA "_max_depth_${_DEPTH_VALUE}")
            list(APPEND compiler_instr_env "TIMEMORY_COMPILER_MAX_DEPTH=${_DEPTH_VALUE}")
        endif()

        if(TARGET timemory::timemory-compiler-instrument-${_LINK_TYPE})
            #
            timemory_add_google_test(
                compiler_instrument${_EXTRA}_${_LINK_TYPE}_tests
                TARGET compiler_instrument_${_LINK_TYPE}_tests
                SOURCES instrumentation_tests.cpp
                DEFINITIONS DISABLE_TIMEMORY
                LINK_LIBRARIES common-test-libs test-opt-flags
                               timemory::timemory-compiler-instrument-${_LINK_TYPE}
                ENVIRONMENT ${compiler_instr_env} PROPERTIES PROCESSOR_AFFINITY ON)
            #
            timemory_add_google_test(
                compiler_instrument_mt${_EXTRA}_${_LINK_TYPE}_tests RUN_SERIAL
                TARGET compiler_instrument_mt_${_LINK_TYPE}_tests
                SOURCES instrumentation_tests.cpp
                DEFINITIONS ENABLE_MT DISABLE_TIMEMORY
                LINK_LIBRARIES common-test-libs test-opt-flags
                               timemory::timemory-compiler-instrument-${_LINK_TYPE}
                ENVIRONMENT ${compiler_instr_env} PROPERTIES PROCESSOR_AFFINITY ON)
            #
            timemory_add_google_test(
                compiler_instrument_timemory${_EXTRA}_${_LINK_TYPE}_tests
                TARGET compiler_instrument_timemory_${_LINK_TYPE}_tests
                SOURCES instrumentation_tests.cpp
                LINK_LIBRARIES common-test-libs test-opt-flags ${_LIBRARY}
                               timemory::timemory-compiler-instrument-${_LINK_TYPE}
                ENVIRONMENT ${compiler_instr_env} PROPERTIES PROCESSOR_AFFINITY ON)
            #
            timemory_add_google_test(
                compiler_instrument_timemory_mt${_EXTRA}_${_LINK_TYPE}_tests RUN_SERIAL
                TARGET compiler_instrument_timemory_mt_${_LINK_TYPE}_tests
                SOURCES instrumentation_tests.cpp
                DEFINITIONS ENABLE_MT
                LINK_LIBRARIES common-test-libs test-opt-flags ${_LIBRARY}
                               timemory::timemory-compiler-instrument-${_LINK_TYPE}
                ENVIRONMENT ${compiler_instr_env} PROPERTIES PROCESSOR_AFFINITY ON)
        endif()
    endforeach()
endforeach()

if(TARGET custom-record-functions)
    timemory_add_google_test(
        library_tests DISCOVER_TESTS
        SOURCES library_tests.cpp
        LINK_LIBRARIES common-test-libs custom-record-functions timemory::timemory-core
                       PROPERTIES PROCESSOR_AFFINITY ON)
endif()

# tracing is only used via dynamic instrumentation so only test on linux
if(_LIBRARY_TARGET
   AND UNIX
   AND NOT APPLE)
    list(APPEND trace_tests_env "TIMEMORY_COLLAPSE_THREADS=ON")
    list(APPEND trace_tests_env "TIMEMORY_NODE_COUNT=2")
    list(APPEND trace_tests_env "TIMEMORY_MPI_THREAD=ON")
    list(APPEND trace_tests_env "TIMEMORY_MPI_THREAD_TYPE=funneled")

    timemory_add_google_test(
        trace_tests MPI NPROC 4
        SOURCES trace_tests.cpp
        LINK_LIBRARIES common-test-libs ${_LIBRARY_TARGET} timemory::timemory-plotting
                       timemory::timemory-mpip-library timemory::timemory-ompt-library
        ENVIRONMENT ${trace_tests_env} PROPERTIES PROCESSOR_AFFINITY ON)

    timemory_add_google_test(
        throttle_tests DISCOVER_TESTS RUN_SERIAL
        SOURCES throttle_tests.cpp
        LINK_LIBRARIES common-test-libs ${_LIBRARY_TARGET} test-opt-flags PROPERTIES
                       PROCESSOR_AFFINITY ON)
endif()

timemory_add_google_test(
    api_tests DISCOVER_TESTS
    SOURCES api_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-core ${_LIBRARY})

if(NOT WIN32)
    timemory_add_google_test(
        ring_buffer_tests
        SOURCES ring_buffer_tests.cpp
        LINK_LIBRARIES common-test-libs ${_LIBRARY} test-werror-flags)
endif()

timemory_add_google_test(
    type_trait_tests DISCOVER_TESTS
    SOURCES type_trait_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-core)

if(NOT WIN32)
    # no support for -Wall in windows flags
    timemory_add_google_test(
        warning_tests
        SOURCES warning_tests.cpp
        LINK_LIBRARIES test-werror-flags $<$<NOT:$<BOOL:WIN32>>:test-debug-flags>
                       timemory::timemory-core)
endif()

if(TIMEMORY_SOCKET_TESTS)
    timemory_add_google_test(
        socket_tests
        SOURCES socket_tests.cpp
        LINK_LIBRARIES common-test-libs timemory::timemory-core test-werror-flags
        TIMEOUT 15)
endif()

timemory_add_google_test(
    settings_tests DISCOVER_TESTS
    SOURCES settings_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-core ${_LIBRARY})

timemory_add_google_test(
    argparse_tests
    SOURCES argparse_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-core)

if(UNIX)
    timemory_add_google_test(
        backtrace_tests
        SOURCES backtrace_tests.cpp
        LINK_LIBRARIES common-test-libs test-debug-flags timemory::timemory-core
                       timemory::timemory-libunwind)
endif()

list(APPEND component_bundle_tests_env "TIMEMORY_COLLAPSE_PROCESSES=OFF")
list(APPEND component_bundle_tests_env "TIMEMORY_COLLAPSE_THREADS=OFF")
list(APPEND component_bundle_tests_env "TIMEMORY_MPI_THREAD=ON")
list(APPEND component_bundle_tests_env "TIMEMORY_MPI_THREAD_TYPE=serialized")
list(APPEND component_bundle_tests_env "TIMEMORY_MAX_WIDTH=140")
list(APPEND component_bundle_tests_env "TIMEMORY_WIDTH=16")
list(APPEND component_bundle_tests_env "TIMEMORY_PRECISION=6")
list(APPEND component_bundle_tests_env "TIMEMORY_SCIENTIFIC=ON")
list(APPEND component_bundle_tests_env "TIMEMORY_STACK_CLEARING=OFF")

timemory_add_google_test(
    traits_tests
    SOURCES traits_tests.cpp
    LINK_LIBRARIES common-test-libs ${_LIBRARY})

timemory_add_google_test(
    component_bundle_tests MPI RUN_SERIAL
    NPROCS 2
    SOURCES component_bundle_tests.cpp
    LINK_LIBRARIES common-test-libs ${_LIBRARY}
    ENVIRONMENT ${component_bundle_tests_env} PROPERTIES PROCESSOR_AFFINITY ON)

list(APPEND kokkosp_tests_env "TIMEMORY_COLLAPSE_THREADS=ON")
list(APPEND kokkosp_tests_env "TIMEMORY_NODE_COUNT=1")
list(APPEND kokkosp_tests_env "TIMEMORY_MPI_THREAD=OFF")
list(APPEND kokkosp_tests_env "TIMEMORY_TIMING_WIDTH=80")
list(APPEND kokkosp_tests_env "TIMEMORY_TIMING_PRECISION=6")
list(APPEND kokkosp_tests_env "TIMEMORY_TIMING_SCIENTIFIC=ON")
list(APPEND kokkosp_tests_env "TIMEMORY_MEMORY_WIDTH=80")
list(APPEND kokkosp_tests_env "TIMEMORY_MEMORY_PRECISION=6")
list(APPEND kokkosp_tests_env "TIMEMORY_MEMORY_SCIENTIFIC=ON")

timemory_add_google_test(
    kokkosp_tests MPI
    NPROCS 2
    SOURCES kokkosp_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-core ${_LIBRARY}
    ENVIRONMENT ${kokkosp_tests_env})

timemory_add_google_test(
    flat_tests DISCOVER_TESTS
    SOURCES flat_tests.cpp PROPERTIES PROCESSOR_AFFINITY ON
    LINK_LIBRARIES common-test-libs extern-test-templates)

timemory_add_google_test(
    timeline_tests DISCOVER_TESTS
    SOURCES timeline_tests.cpp PROPERTIES PROCESSOR_AFFINITY ON
    LINK_LIBRARIES common-test-libs extern-test-templates)

timemory_add_google_test(
    data_tracker_tests DISCOVER_TESTS RUN_SERIAL
    SOURCES data_tracker_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   ${_LIBRARY})

# if(TARGET timemory::timemory-io-component)
if(WIN32) # for now just enable for Windows
    timemory_add_google_test(
        io_tests
        SOURCES io_tests.cpp
        LINK_LIBRARIES common-test-libs ${_LIBRARY})
endif()

if(NOT WIN32)
    # cache structures are designed for use on Windows
    timemory_add_google_test(
        cache_tests
        SOURCES cache_tests.cpp
        LINK_LIBRARIES common-test-libs timemory::timemory-core ${_LIBRARY}
        TIMEOUT 300 PROPERTIES PROCESSOR_AFFINITY ON)
endif()

timemory_add_google_test(
    archive_storage_tests RUN_SERIAL MPI NPROC 2
    SOURCES archive_storage_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-dmp ${_LIBRARY})

timemory_add_google_test(
    threading_tests DISCOVER_TESTS RUN_SERIAL
    SOURCES threading_tests.cpp PROPERTIES PROCESSOR_AFFINITY ON
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   timemory::timemory-ompt ${_LIBRARY} ${_OPENMP})

if(TIMEMORY_USE_PTL AND TARGET PTL::ptl)
    list(APPEND ptl_tests_env "TIMEMORY_ENABLE_PTHREAD_GOTCHA_WRAPPER=ON")
    timemory_add_google_test(
        ptl_tests
        SOURCES ptl_tests.cpp
        LINK_LIBRARIES
            common-test-libs PTL::ptl ${_LIBRARY} $<$<NOT:$<BOOL:WIN32>>:test-debug-flags>
            timemory::timemory-sanitizer
        ENVIRONMENT ${ptl_tests_env}
        TIMEOUT 300 PROPERTIES PROCESSOR_AFFINITY ON)
endif()

timemory_add_google_test(
    stl_overload_tests DISCOVER_TESTS
    SOURCES stl_overload_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   ${_LIBRARY})

timemory_add_google_test(
    user_bundle_tests DISCOVER_TESTS
    SOURCES user_bundle_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   ${_LIBRARY})

if(NOT WIN32)
    timemory_add_google_test(
        mangle_tests DISCOVER_TESTS
        SOURCES mangle_tests.cpp
        LINK_LIBRARIES common-test-libs timemory::timemory-plotting
                       timemory::timemory-core ${_LIBRARY})
endif()

timemory_add_google_test(
    tuple_tests DISCOVER_TESTS RUN_SERIAL
    SOURCES tuple_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-papi timemory::timemory-plotting
                   timemory::timemory-caliper ${_LIBRARY})

timemory_add_google_test(
    macro_tests DISCOVER_TESTS
    SOURCES macro_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   ${_LIBRARY})

timemory_add_google_test(
    hybrid_tests DISCOVER_TESTS RUN_SERIAL
    SOURCES hybrid_tests.cpp
    LINK_LIBRARIES common-test-libs ${HYBRID_COMPONENTS} timemory::timemory-plotting
                   timemory::timemory-core ${_LIBRARY})

timemory_add_google_test(
    timing_tests DISCOVER_TESTS RUN_SERIAL
    SOURCES timing_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   ${_LIBRARY})

timemory_add_google_test(
    rusage_tests DISCOVER_TESTS
    SOURCES rusage_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   ${_LIBRARY})

if(TIMEMORY_USE_LIBUNWIND)
    timemory_add_google_test(
        libunwind_tests DISCOVER_TESTS
        SOURCES libunwind_tests.cpp
        LINK_LIBRARIES common-test-libs timemory::timemory-plotting
                       timemory::timemory-core timemory::timemory-libunwind ${_LIBRARY})
endif()

set(VALID_PARANOID OFF)
set(PARANOID_FILE "/proc/sys/kernel/perf_event_paranoid")
if(EXISTS "${PARANOID_FILE}")
    file(READ "${PARANOID_FILE}" PARANOID_VALUE LIMIT_COUNT 1)
    if(PARANOID_VALUE LESS 2)
        set(VALID_PARANOID ON)
    endif()
endif()
set(PARANOID_VALUE 3)
function(PARANOID_WARNING_MESSAGE _FILE _VALUE)
    if(NOT TIMEMORY_BUILD_GOOGLE_TEST OR NOT EXISTS "${_FILE}")
        return()
    endif()

    message(
        AUTHOR_WARNING
            "
Hardware counters are not available: \n\t${_FILE} is ${_VALUE}
${ARGN}
perf_event_paranoid controls use of the performance events system by unprivileged users
(without CAP_SYS_ADMIN). The default value is 2.
    -1:  Allow use of (almost) all events by all users
         Ignore mlock limit after perf_event_mlock_kb without CAP_IPC_LOCK
    >=0: Disallow ftrace function tracepoint by users without CAP_SYS_ADMIN
         Disallow raw tracepoint access by users without CAP_SYS_ADMIN
    >=1: Disallow CPU event access by users without CAP_SYS_ADMIN
    >=2: Disallow kernel profiling by users without CAP_SYS_ADMIN
If you have sudo access, you can change this setting via one of the following commands:
$ echo -1 | sudo tee ${_FILE}
$ echo 0 | sudo tee ${_FILE}
$ echo 1 | sudo tee ${_FILE}
")
endfunction()

if(TIMEMORY_USE_PAPI AND VALID_PARANOID)
    timemory_add_google_test(
        papi_tests DISCOVER_TESTS
        SOURCES papi_tests.cpp PROPERTIES PROCESSOR_AFFINITY ON
        LINK_LIBRARIES
            common-test-libs timemory::timemory-papi timemory::timemory-plotting
            timemory::timemory-arch timemory::timemory-roofline-options test-opt-flags
            ${_LIBRARY})

    # zen cpus *think* PAPI_DP_OPS is valid but it is always zero
    if(TARGET papi_tests)
        target_compile_definitions(
            papi_tests PRIVATE DISABLE_DP=$<IF:$<STREQUAL:${CpuArch_TARGET},zen>,1,0>)
    endif()
elseif(TIMEMORY_USE_PAPI)
    if(PARANOID_VALUE)
        paranoid_warning_message(${PARANOID_FILE} ${PARANOID_VALUE}
                                 "Disabling PAPI validation tests")
    else()
        paranoid_warning_message(${PARANOID_FILE} "unknown"
                                 "Disabling PAPI validation tests")
    endif()
endif()

# doesn't require valid HW-counter values
if(TIMEMORY_USE_PAPI)
    # roofline
    timemory_add_google_test(
        cpu_roofline_tests DISCOVER_TESTS RUN_SERIAL
        SOURCES cpu_roofline_tests.cpp PROPERTIES PROCESSOR_AFFINITY ON
        LINK_LIBRARIES
            common-test-libs timemory::timemory-papi timemory::timemory-plotting
            timemory::timemory-arch timemory::timemory-roofline-options test-opt-flags
            ${_LIBRARY})
endif()

timemory_add_google_test(
    apply_tests DISCOVER_TESTS
    SOURCES apply_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   ${_LIBRARY})

timemory_add_google_test(
    chained_tests DISCOVER_TESTS
    SOURCES chained_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   ${_LIBRARY})

if(TIMEMORY_USE_ARCH)
    timemory_add_google_test(
        aligned_allocator_tests DISCOVER_TESTS
        SOURCES aligned_allocator_tests.cpp
        LINK_LIBRARIES common-test-libs timemory::timemory-arch timemory::timemory-core
                       timemory::timemory-plotting ${_LIBRARY})
endif()

set_source_files_properties(
    cuda_tests.cpp cupti_tests.cpp cupti_profiler_tests.cpp
    PROPERTIES LANGUAGE ${DEVICE_LANGUAGE} LINKER_LANGUAGE ${DEVICE_LANGUAGE})

if(TIMEMORY_USE_CUDA)
    timemory_add_google_test(
        cuda_tests DISCOVER_TESTS RUN_SERIAL
        SOURCES cuda_tests.cpp
        LINK_LIBRARIES
            common-test-libs timemory::timemory-arch timemory::timemory-core
            timemory::timemory-cuda timemory::timemory-cudart timemory::timemory-plotting
            ${_LIBRARY})
    #
    timemory_add_google_test(
        cuda_nvtx_tests DISCOVER_TESTS
        SOURCES cuda_tests.cpp
        LINK_LIBRARIES
            common-test-libs timemory::timemory-arch timemory::timemory-core
            timemory::timemory-cuda timemory::timemory-cudart timemory::timemory-plotting
            ${_LIBRARY})
endif()

if(TIMEMORY_USE_CUPTI AND HAS_CUDA_DRIVER_LIBRARY)
    set(cupti_tests_env
        "TIMEMORY_CUPTI_ACTIVITY_LEVEL=2;TIMEMORY_TIMING_PRECISION=9;TIMEMORY_TIMING_UNITS=msec"
        )
    add_library(cupti-compile-options INTERFACE)
    target_compile_options(
        cupti-compile-options
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:NVIDIA>:--default-stream=per-thread>>
        )

    timemory_add_google_test(
        cupti_tests DISCOVER_TESTS RUN_SERIAL
        SOURCES cupti_tests.cpp
        LINK_LIBRARIES
            common-test-libs timemory::timemory-arch timemory::timemory-core
            timemory::timemory-cuda timemory::timemory-cupti timemory::timemory-plotting
            cupti-compile-options ${_LIBRARY}
        ENVIRONMENT ${cupti_tests_env})

    # NVPERF library
    set(nvperf_valid_archs volta turing)
    if(CUPTI_nvperf_host_LIBRARY AND "${TIMEMORY_CUDA_ARCH}" IN_LIST nvperf_valid_archs)
        timemory_add_google_test(
            cupti_profiler_tests DISCOVER_TESTS RUN_SERIAL
            SOURCES cupti_profiler_tests.cpp
            LINK_LIBRARIES
                common-test-libs
                timemory::timemory-arch
                timemory::timemory-core
                timemory::timemory-cuda
                timemory::timemory-cupti
                timemory::timemory-plotting
                timemory::timemory-cudart-device
                cupti-compile-options
                ${_LIBRARY})
    endif()
endif()

if(TIMEMORY_USE_GPERFTOOLS)
    timemory_add_google_test(
        gperftools_cpu_tests
        SOURCES gperftools_cpu_tests.cpp
        LINK_LIBRARIES common-test-libs timemory::timemory-gperftools
                       timemory::timemory-plotting timemory::timemory-core ${_LIBRARY}
        ENVIRONMENT "CPUPROFILE_FREQUENCY=1000;CPUPROFILE_REALTIME=1")
    timemory_add_google_test(
        gperftools_heap_tests DISCOVER_TESTS
        SOURCES gperftools_heap_tests.cpp
        LINK_LIBRARIES
            common-test-libs timemory::timemory-arch timemory::timemory-gperftools
            timemory::timemory-plotting timemory::timemory-core ${_LIBRARY})
endif()

if(NOT WIN32)
    timemory_add_google_test(
        gperftools_fake_cpu_tests DISCOVER_TESTS
        SOURCES gperftools_cpu_tests.cpp
        LINK_LIBRARIES
            common-test-libs timemory::timemory-headers timemory::timemory-plotting
            timemory::timemory-analysis-tools timemory::timemory-core ${_LIBRARY})
endif()

if(TIMEMORY_USE_GOTCHA)
    timemory_add_google_test(
        gotcha_tests DISCOVER_TESTS
        SOURCES gotcha_tests.cpp
        LINK_LIBRARIES
            common-test-libs gotcha-tests-lib timemory::timemory-headers
            timemory::timemory-analysis-tools timemory::timemory-plotting ${_LIBRARY})
endif()

timemory_add_google_test(
    priority_tests DISCOVER_TESTS
    SOURCES priority_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   ${_LIBRARY})

if(TIMEMORY_USE_MPI)
    #
    list(APPEND mpi_base_env "TIMEMORY_MPI_THREAD_TYPE=serialized")
    list(APPEND mpi_base_env "TIMEMORY_SEPARATOR_FREQ=0")
    #
    set(mpi_tests_env ${mpi_base_env})
    list(APPEND mpi_tests_env
         "TIMEMORY_OUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/timemory-mpi-tests-output")
    #
    timemory_add_google_test(
        mpi_tests MPI NPROC 2
        SOURCES mpi_tests.cpp
        LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-dmp
                       timemory::timemory-core ${_LIBRARY}
        ENVIRONMENT "${mpi_tests_env}")
    #
    set(mpi_tests_diff_env ${mpi_base_env})
    list(APPEND mpi_tests_diff_env "TIMEMORY_DIFF_OUTPUT=ON")
    list(APPEND mpi_tests_diff_env "TIMEMORY_OUTPUT_PATH=timemory-mpi-tests-diff-output")
    list(APPEND mpi_tests_diff_env
         "TIMEMORY_INPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/timemory-mpi-tests-output")
    #
    timemory_add_google_test(
        mpi_tests_diff MPI NPROC 2
        DEPENDS mpi_tests
        COMMAND $<TARGET_FILE:mpi_tests>
        LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-dmp
                       timemory::timemory-core ${_LIBRARY}
        ENVIRONMENT "${mpi_tests_diff_env}")

endif()

if(TIMEMORY_USE_UPCXX)
    timemory_add_google_test(
        upcxx_tests DISCOVER_TESTS
        SOURCES upcxx_tests.cpp
        LINK_LIBRARIES common-test-libs timemory::timemory-plotting
                       timemory::timemory-upcxx timemory::timemory-core ${_LIBRARY})
endif()

timemory_add_google_test(
    variadic_tests DISCOVER_TESTS
    SOURCES variadic_tests.cpp
    LINK_LIBRARIES
        common-test-libs timemory::timemory-plotting timemory::timemory-core
        timemory::timemory-dmp $<$<NOT:$<BOOL:WIN32>>:test-debug-flags> ${_LIBRARY})

timemory_add_google_test(
    derived_tests DISCOVER_TESTS
    SOURCES derived_tests.cpp
    LINK_LIBRARIES common-test-libs timemory::timemory-plotting timemory::timemory-core
                   timemory::timemory-dmp ${_LIBRARY})

set(_OMPT_TARGET)
if(TARGET timemory::timemory-ompt-shared)
    set(_OMPT_TARGET timemory::timemory-ompt-shared)
elseif(TARGET timemory::timemory-ompt-static)
    set(_OMPT_TARGET timemory::timemory-ompt-static)
endif()

if(_OMPT_TARGET AND _OPENMP)
    timemory_add_google_test(
        ompt_handle_tests DISCOVER_TESTS
        SOURCES ompt_handle_tests.cpp
        LINK_LIBRARIES common-test-libs ${_OMPT_TARGET} ${_OPENMP})
endif()

if(TARGET timemory::timemory-ncclp-library AND TARGET nccl_test_interface)
    timemory_add_google_test(
        nccl_tests DISCOVER_TESTS RUN_SERIAL
        SOURCES nccl_tests.cpp
        LINK_LIBRARIES common-test-libs nccl_test_interface
                       timemory::timemory-ncclp-library
        ENVIRONMENT "TIMEMORY_NCCLP_COMPONENTS=wall_clock")
endif()

if(TIMEMORY_BUILD_ERT)
    timemory_add_google_test(
        ert_tests
        SOURCES ert_tests.cpp PROPERTIES PROCESSOR_AFFINITY ON
        LINK_LIBRARIES common-test-libs timemory::timemory-ert ${_LIBRARY})
endif()

if(TARGET timemory-mallocp-shared)
    timemory_add_google_test(
        mallocp_tests DISCOVER_TESTS
        SOURCES mallocp_tests.cpp
        LINK_LIBRARIES common-test-libs timemory::timemory-mallocp-library)
endif()

if(TIMEMORY_USE_PYTHON AND PYTHON_EXECUTABLE)
    # test showing linked libraries of python module
    if(APPLE)
        find_program(OTOOL_EXE NAMES otool)
        if(OTOOL_EXE)
            set(LINKAGE_COMMAND ${OTOOL_EXE} -L)
        endif()
    elseif("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
        find_program(LDD_EXE NAMES ldd)
        if(LDD_EXE)
            set(LINKAGE_COMMAND ${LDD_EXE})
        endif()
    endif()

    if(LINKAGE_COMMAND)
        add_test(
            NAME python_module_linkage
            COMMAND ${LINKAGE_COMMAND} $<TARGET_FILE:libpytimemory>
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    endif()

    add_test(
        NAME python_tests
        COMMAND ${PYTHON_EXECUTABLE} -m timemory.test -v -v
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

# Tests recording the symbols of each compiled library
timemory_add_option(TIMEMORY_NM_TESTS "Include nm tests" OFF NO_FEATURE)
if(TIMEMORY_NM_TESTS)
    find_program(NM_EXE NAMES nm)
    mark_as_advanced(NM_EXE)
    if(NM_EXE AND NOT WIN32)
        set(_COMPILED_LIBS ${TIMEMORY_TOOL_LIBRARIES} ${TIMEMORY_COMPILED_LIBRARIES}
                           ${TIMEMORY_INTERMEDIATE_LIBRARIES})
        list(REMOVE_DUPLICATES _COMPILED_LIBS)
        if(TARGET libpytimemory AND NOT "libpytimemory" IN_LIST _COMPILED_LIBS)
            list(APPEND _COMPILED_LIBS "libpytimemory")
        endif()
        foreach(_LIB ${_COMPILED_LIBS})
            if(NOT TARGET ${_LIB})
                continue()
            endif()
            add_test(
                NAME nm-mangled-${_LIB}
                COMMAND ${NM_EXE} -g $<TARGET_FILE:${_LIB}>
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
            if(APPLE
               AND "$ENV{USER}" STREQUAL "travis"
               AND "$ENV{TRAVIS}" STREQUAL "true")
                continue()
            endif()
            add_test(
                NAME nm-demangled-${_LIB}
                COMMAND ${NM_EXE} -g --demangle $<TARGET_FILE:${_LIB}>
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endforeach()
    endif()
endif()

# disable during code-coverage
if(TIMEMORY_BUILD_GOOGLE_TEST AND NOT TIMEMORY_USE_COVERAGE)
    add_executable(dynamic_instrument_tests ${_EXCLUDE} instrumentation_tests.cpp)
    target_compile_definitions(dynamic_instrument_tests PRIVATE DISABLE_TIMEMORY)
    target_link_libraries(dynamic_instrument_tests common-test-libs timemory-google-test
                          timemory::timemory-dmp google-test-debug-options)
    set_target_properties(
        dynamic_instrument_tests
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_dependencies(timemory-test dynamic_instrument_tests)

    add_executable(dynamic_instrument_timemory_tests ${_EXCLUDE}
                                                     instrumentation_tests.cpp)
    target_link_libraries(
        dynamic_instrument_timemory_tests common-test-libs ${_LIBRARY_TARGET}
        timemory-google-test timemory::timemory-dmp google-test-debug-options)
    set_target_properties(
        dynamic_instrument_timemory_tests
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_dependencies(timemory-test dynamic_instrument_timemory_tests)
endif()

# disable during code-coverage
if(TIMEMORY_USE_DYNINST
   AND TIMEMORY_BUILD_GOOGLE_TEST
   AND NOT TIMEMORY_USE_COVERAGE)
    if(NOT TARGET timemory-run)
        message(FATAL_ERROR "TIMEMORY_USE_DYNINST is ON but no target 'timemory-run'")
    endif()

    if(TIMEMORY_USE_MPI)
        set(LAUNCHER "${MPIEXEC_EXECUTABLE} -n 2")
    endif()

    set(TEST_ENVIRON)
    list(APPEND TEST_ENVIRON "TIMEMORY_TRACE_COMPONENTS=wall_clock")
    list(APPEND TEST_ENVIRON "TIMEMORY_GLOBAL_COMPONENTS=wall_clock:cpu_clock")

    set(_scripts # attach-process
        binary-rewrite-regex binary-rewrite-stubs binary-rewrite launch-process
        region-sync)

    if(TIMEMORY_USE_MPI AND TIMEMORY_USE_GOTCHA)
        list(APPEND _scripts binary-rewrite-mpip)
        list(APPEND TEST_ENVIRON "TIMEMORY_MPIP_COMPONENTS=wall_clock")
    endif()

    if(TARGET custom-dynamic-instr)
        set(USE_PAPI 0)
        if(TIMEMORY_USE_PAPI)
            set(USE_PAPI 1)
        endif()
        list(APPEND _scripts custom-components)
    endif()

    configure_file(${CMAKE_CURRENT_LIST_DIR}/scripts/common.sh.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dynamic/common.sh @ONLY)

    # the executable being instrumented
    set(_DYNAMIC_TARGETS dynamic_instrument_tests dynamic_instrument_timemory_tests)

    # skip these types when timemory exists in the source
    set(_SKIP_DYNAMIC_WITH_TIMEMORY "launch-process" "binary-rewrite-stubs")

    foreach(_TARGET ${_DYNAMIC_TARGETS})
        foreach(_FILE ${_scripts})
            # do not run launch-process + timemory
            if("${_TARGET}" STREQUAL "dynamic_instrument_timemory_tests"
               AND "${_FILE}" IN_LIST _SKIP_DYNAMIC_WITH_TIMEMORY)
                continue()
            endif()

            # set the relative or absolute path
            if("${_FILE}" STREQUAL "launch-process")
                set(COMMAND ${_TARGET})
            else()
                set(COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_TARGET})
            endif()

            set(TEST_NAME ${_TARGET}-${_FILE})
            configure_file(
                ${CMAKE_CURRENT_LIST_DIR}/scripts/${_FILE}.sh.in
                ${CMAKE_CURRENT_BINARY_DIR}/dynamic/${_TARGET}-${_FILE}.sh @ONLY)
            add_test(
                NAME ${_TARGET}-${_FILE}
                COMMAND ${CMAKE_CURRENT_BINARY_DIR}/dynamic/${_TARGET}-${_FILE}.sh
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dynamic)
            set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "${TEST_ENVIRON}"
                                                         TIMEOUT 480 RUN_SERIAL ON)
            set_property(
                TEST ${_TARGET}-${_FILE}
                APPEND
                PROPERTY DEPENDS ${_TARGET} timemory-run)
        endforeach()
    endforeach()
endif()

if(TIMEMORY_USE_CTP AND compile-time-perf_ANALYZER_EXECUTABLE)
    add_test(
        NAME compile-time-perf
        COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target
                analyze-timemory-compile-time
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endif()
