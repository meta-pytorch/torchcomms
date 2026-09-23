# ex-cxx-overhead

This example demonstrates the measurement of instrumentaion overheads (both in timing and resident set size) for timemory with increasing number of instrumentation components used.

## Build

See [examples](../README.md##Build).

## Expected Output

```bash
$ ./ex_cxx_overhead
#------------------------- tim::manager initialized [id=0][pid=5626] -------------------------#

#----------------------------------------------------------------------------------------#
# Environment settings:
#               DYLD_INSERT_LIBRARIES    =
#                          LD_PRELOAD    =
#                      PAT_RT_PERFCTR    =
#              TIMEMORY_ADD_SECONDARY    =      true
#       TIMEMORY_ALLOW_SIGNAL_HANDLER    =      true
#                TIMEMORY_AUTO_OUTPUT    =      true
#                     TIMEMORY_BANNER    =      true
#     TIMEMORY_CALIPER_CONFIG_ENABLED    =      true
# TIMEMORY_CALIPER_LOOP_MARKER_ENABLED   =      true
#     TIMEMORY_CALIPER_MARKER_ENABLED    =      true
#         TIMEMORY_COLLAPSE_PROCESSES    =      true
#           TIMEMORY_COLLAPSE_THREADS    =      true
#                TIMEMORY_COUT_OUTPUT    =      true
#               TIMEMORY_CPU_AFFINITY    =      false
#                TIMEMORY_CPU_ENABLED    =      true
#    TIMEMORY_CPU_ROOFLINE_OP_ENABLED    =      true
#           TIMEMORY_CPU_UTIL_ENABLED    =      true
#                    TIMEMORY_CRAYPAT    =
#      TIMEMORY_CUDA_EVENT_BATCH_SIZE    =      5
#       TIMEMORY_CUPTI_ACTIVITY_KINDS    =
#       TIMEMORY_CUPTI_ACTIVITY_LEVEL    =      1
#               TIMEMORY_CUPTI_DEVICE    =      0
#               TIMEMORY_CUPTI_EVENTS    =
#              TIMEMORY_CUPTI_METRICS    =
#   TIMEMORY_CURRENT_PEAK_RSS_ENABLED    =      true
#                 TIMEMORY_DART_COUNT    =      1
#                 TIMEMORY_DART_LABEL    =      true
#                TIMEMORY_DART_OUTPUT    =      false
#                  TIMEMORY_DART_TYPE    =
#                      TIMEMORY_DEBUG    =      false
#          TIMEMORY_DESTRUCTOR_REPORT    =      false
#                TIMEMORY_DIFF_OUTPUT    =      false
#        TIMEMORY_DISABLE_ALL_SIGNALS    =      false
#                    TIMEMORY_ENABLED    =      true
#         TIMEMORY_ENABLE_ALL_SIGNALS    =      false
#      TIMEMORY_ENABLE_SIGNAL_HANDLER    =      false
#              TIMEMORY_ERT_ALIGNMENT    =      0
#             TIMEMORY_ERT_BLOCK_SIZE    =      1024
#              TIMEMORY_ERT_GRID_SIZE    =      0
#          TIMEMORY_ERT_MAX_DATA_SIZE    =      0
#      TIMEMORY_ERT_MAX_DATA_SIZE_CPU    =      0
#      TIMEMORY_ERT_MAX_DATA_SIZE_GPU    =      500000000
#       TIMEMORY_ERT_MIN_WORKING_SIZE    =      0
#   TIMEMORY_ERT_MIN_WORKING_SIZE_CPU    =      64
#   TIMEMORY_ERT_MIN_WORKING_SIZE_GPU    =      10000000
#            TIMEMORY_ERT_NUM_STREAMS    =      1
#            TIMEMORY_ERT_NUM_THREADS    =      0
#        TIMEMORY_ERT_NUM_THREADS_CPU    =      12
#        TIMEMORY_ERT_NUM_THREADS_GPU    =      1
#               TIMEMORY_ERT_SKIP_OPS    =
#                TIMEMORY_FILE_OUTPUT    =      true
#          TIMEMORY_FLAMEGRAPH_OUTPUT    =      true
#               TIMEMORY_FLAT_PROFILE    =      false
#           TIMEMORY_INPUT_EXTENSIONS    =      json,xml
#                 TIMEMORY_INPUT_PATH    =
#               TIMEMORY_INPUT_PREFIX    =
#       TIMEMORY_INSTRUCTION_ROOFLINE    =      false
#              TIMEMORY_IO_IN_ENABLED    =      true
#             TIMEMORY_IO_OUT_ENABLED    =      true
#                TIMEMORY_JSON_OUTPUT    =      false
#        TIMEMORY_KERNEL_MODE_ENABLED    =      true
#               TIMEMORY_LIBRARY_CTOR    =      true
#      TIMEMORY_LIKWID_MARKER_ENABLED    =      true
#    TIMEMORY_MAJOR_PAGE_FLTS_ENABLED    =      true
#      TIMEMORY_MALLOC_GOTCHA_ENABLED    =      true
#                  TIMEMORY_MAX_DEPTH    =      65535
#       TIMEMORY_MAX_THREAD_BOOKMARKS    =      50
#                  TIMEMORY_MAX_WIDTH    =      120
#           TIMEMORY_MEMORY_PRECISION    =      3
#          TIMEMORY_MEMORY_SCIENTIFIC    =      false
#               TIMEMORY_MEMORY_UNITS    =      kB
#               TIMEMORY_MEMORY_WIDTH    =      -1
#    TIMEMORY_MINOR_PAGE_FLTS_ENABLED    =      true
#    TIMEMORY_MONOTONIC_CLOCK_ENABLED    =      true
# TIMEMORY_MONOTONIC_RAW_CLOCK_ENABLED   =      true
#               TIMEMORY_MPI_FINALIZE    =      false
#                   TIMEMORY_MPI_INIT    =      false
#                 TIMEMORY_MPI_THREAD    =      true
#            TIMEMORY_MPI_THREAD_TYPE    =
#                 TIMEMORY_NODE_COUNT    =      0
#    TIMEMORY_NVTX_MARKER_DEVICE_SYNC    =      true
#  TIMEMORY_OMPT_DATA_TRACKER_ENABLED    =      true
#        TIMEMORY_OMPT_HANDLE_ENABLED    =      true
#                TIMEMORY_OUTPUT_PATH    =      timemory-ex-cxx-overhead-output
#              TIMEMORY_OUTPUT_PREFIX    =
#           TIMEMORY_PAGE_RSS_ENABLED    =      true
#        TIMEMORY_PAPI_ARRAY0_ENABLED    =      true
#                TIMEMORY_PAPI_ATTACH    =      false
#                TIMEMORY_PAPI_EVENTS    =
#         TIMEMORY_PAPI_FAIL_ON_ERROR    =      false
#          TIMEMORY_PAPI_MULTIPLEXING    =      true
#              TIMEMORY_PAPI_OVERFLOW    =      0
#                 TIMEMORY_PAPI_QUIET    =      false
#       TIMEMORY_PAPI_VECTOR0_ENABLED    =      true
#           TIMEMORY_PEAK_RSS_ENABLED    =      true
#                TIMEMORY_PLOT_OUTPUT    =      true
#                  TIMEMORY_PRECISION    =      -1
#      TIMEMORY_PRIO_CXT_SWCH_ENABLED    =      true
#        TIMEMORY_PROCESS_CPU_ENABLED    =      true
#      TIMEMORY_PROC_CPU_UTIL_ENABLED    =      true
#                 TIMEMORY_PYTHON_EXE    =      /home/mhaseeb/repos/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/python-3.7.7-2dybrjceqs3qc4k7ci56t56bvzb4csxc/bin/python
#         TIMEMORY_READ_BYTES_ENABLED    =      true
#        TIMEMORY_ROOFLINE_EVENTS_CPU    =
#        TIMEMORY_ROOFLINE_EVENTS_GPU    =
#              TIMEMORY_ROOFLINE_MODE    =      op
#          TIMEMORY_ROOFLINE_MODE_CPU    =      op
#          TIMEMORY_ROOFLINE_MODE_GPU    =      op
#       TIMEMORY_ROOFLINE_TYPE_LABELS    =      false
#   TIMEMORY_ROOFLINE_TYPE_LABELS_CPU    =      false
#   TIMEMORY_ROOFLINE_TYPE_LABELS_GPU    =      false
#                 TIMEMORY_SCIENTIFIC    =      false
#             TIMEMORY_STACK_CLEARING    =      true
#           TIMEMORY_SUPPRESS_PARSING    =      false
#                TIMEMORY_SYS_ENABLED    =      true
#                 TIMEMORY_TARGET_PID    =      5626
#                TIMEMORY_TEXT_OUTPUT    =      true
#         TIMEMORY_THREAD_CPU_ENABLED    =      true
#    TIMEMORY_THREAD_CPU_UTIL_ENABLED    =      true
#             TIMEMORY_THROTTLE_COUNT    =      10000
#             TIMEMORY_THROTTLE_VALUE    =      10000
# TIMEMORY_TIM::COMPONENT::OMPT_TARGET_DATA_TAG_LONG_ENABLED     =      true
#           TIMEMORY_TIMELINE_PROFILE    =      false
#                TIMEMORY_TIME_FORMAT    =      %F_%I.%M_%p
#                TIMEMORY_TIME_OUTPUT    =      false
#           TIMEMORY_TIMING_PRECISION    =      6
#          TIMEMORY_TIMING_SCIENTIFIC    =      true
#               TIMEMORY_TIMING_UNITS    =
#               TIMEMORY_TIMING_WIDTH    =      -1
#         TIMEMORY_TRIP_COUNT_ENABLED    =      true
#             TIMEMORY_UPCXX_FINALIZE    =      true
#                 TIMEMORY_UPCXX_INIT    =      true
#        TIMEMORY_USER_BUNDLE_ENABLED    =      true
#               TIMEMORY_USER_ENABLED    =      true
#          TIMEMORY_USER_MODE_ENABLED    =      true
#                    TIMEMORY_VERBOSE    =      0
#     TIMEMORY_VIRTUAL_MEMORY_ENABLED    =      true
#       TIMEMORY_VOL_CXT_SWCH_ENABLED    =      true
#               TIMEMORY_WALL_ENABLED    =      true
#                      TIMEMORY_WIDTH    =      12
#      TIMEMORY_WRITTEN_BYTES_ENABLED    =      true
#----------------------------------------------------------------------------------------#

Running fibonacci(n = 43, cutoff = 28)...

run/ex_cxx_overhead.cpp:199 [with timing = mode::none] answer : 433494437 (# measurments: 0, # unique: 0)
[warmup]>>>  run/ex_cxx_overhead.cpp:199 [with timing = mode::none] : 1.082041e+00 sec wall, 1.080000e+00 sec cpu,        0.000 KB peak_rss [laps: 1]
Initializing caliper...

run/ex_cxx_overhead.cpp:199 [with timing = mode::none] answer : 433494437 (# measurments: 0, # unique: 0)
run/ex_cxx_overhead.cpp:199 [with timing = mode::blank] answer : 433494437 (# measurments: 1596, # unique: 15)
run/ex_cxx_overhead.cpp:199 [with timing = mode::blank_pointer] answer : 433494437 (# measurments: 1596, # unique: 15)
run/ex_cxx_overhead.cpp:199 [with timing = mode::basic] answer : 433494437 (# measurments: 1596, # unique: 1596)
run/ex_cxx_overhead.cpp:199 [with timing = mode::basic_pointer] answer : 433494437 (# measurments: 1596, # unique: 1596)


Report from 6384 total measurements and 3222 unique measurements:
    >>>  run/ex_cxx_overhead.cpp:199 [with timing = mode::none]                                         : 1.082036e+00 sec wall, 1.090000e+00 sec cpu,        0.000 KB peak_rss [laps: 1]

    >>>  run/ex_cxx_overhead.cpp:199 [with timing = mode::blank]                                        : 1.083617e+00 sec wall, 1.080000e+00 sec cpu,        0.000 KB peak_rss [laps: 1]
    >>>  difference vs. 15 unique measurements and 1596 total measurements (mode::blank)                : 1.580769e-03 sec wall, -1.000000e-02 sec cpu,        0.000 KB peak_rss
    >>>  average overhead of 15 unique measurements and 1596 total measurements (mode::blank)           : 9.900000e-07 sec wall, -6.265000e-06 sec cpu,        0.000 KB peak_rss

    >>>  run/ex_cxx_overhead.cpp:199 [with timing = mode::blank_pointer]                                : 1.082705e+00 sec wall, 1.090000e+00 sec cpu,        0.000 KB peak_rss [laps: 1]
    >>>  difference vs. 15 unique measurements and 1596 total measurements (mode::blank_pointer)        : 6.690730e-04 sec wall, 0.000000e+00 sec cpu,        0.000 KB peak_rss
    >>>  average overhead of 15 unique measurements and 1596 total measurements (mode::blank_pointer)   : 4.190000e-07 sec wall, 0.000000e+00 sec cpu,        0.000 KB peak_rss

    >>>  run/ex_cxx_overhead.cpp:199 [with timing = mode::basic]                                        : 1.083832e+00 sec wall, 1.080000e+00 sec cpu,      520.000 KB peak_rss [laps: 1]
    >>>  difference vs. 1596 unique measurements and 1596 total measurements (mode::basic)              : 1.795429e-03 sec wall, -1.000000e-02 sec cpu,      520.000 KB peak_rss
    >>>  average overhead of 1596 unique measurements and 1596 total measurements (mode::basic)         : 1.124000e-06 sec wall, -6.265000e-06 sec cpu,        0.325 KB peak_rss

    >>>  run/ex_cxx_overhead.cpp:199 [with timing = mode::basic_pointer]                                : 1.084386e+00 sec wall, 1.090000e+00 sec cpu,      264.000 KB peak_rss [laps: 1]
    >>>  difference vs. 1596 unique measurements and 1596 total measurements (mode::basic_pointer)      : 2.349486e-03 sec wall, 0.000000e+00 sec cpu,      264.000 KB peak_rss
    >>>  average overhead of 1596 unique measurements and 1596 total measurements (mode::basic_pointer) : 1.472000e-06 sec wall, 0.000000e+00 sec cpu,        0.165 KB peak_rss


[INFO]> L1 cache size: 32 KB, L2 cache size: 262 KB, L3 cache size: 15728 KB, max cache size: 15728 KB

Expected size: 3222, actual size: 3219
[vol_cxt_swch]|0> Outputting 'timemory-ex-cxx-overhead-output/vol_cxt_swch.json'...
[vol_cxt_swch]|0> Outputting 'timemory-ex-cxx-overhead-output/vol_cxt_swch.txt'...
Opening 'timemory-ex-cxx-overhead-output/vol_cxt_swch.jpeg' for output...
Closed 'timemory-ex-cxx-overhead-output/vol_cxt_swch.jpeg'...

```