# ex-cpu-roofline

This example demonstrates the use of timemory 's roofline component to execute a set of parallel algorithms on CPU to measure the theoretical peak and then plotting a customized roofline plot. The auto tuple used for instrumentation is as follows:

```c
auto_tuple_t = tim::auto_tuple_t<wall_clock, cpu_clock, cpu_util, roofline_t>;
roofline_ert_config_t = typename roofline_t::ert_config_type<float_type>;
```

## Roofline and Roofline Component

See [Roofline Components](../../docs/components/roofline.md)

## Build

See [examples](../README.md##Build). Further, this example requires PAPI support by enabling `-DTIMEMORY_USE_PAPI=ON`.

## Expected Output

```bash
$ ./ex_cpu_roofline
#------------------------- tim::manager initialized [id=0][pid=31883] -------------------------#

#----------------------------------------------------------------------------------------#
# Environment settings:
#               DYLD_INSERT_LIBRARIES    =
#                          LD_PRELOAD    =
#                      PAT_RT_PERFCTR    =
#              TIMEMORY_ADD_SECONDARY    =      true
#       TIMEMORY_ALLOW_SIGNAL_HANDLER    =      true
#                TIMEMORY_AUTO_OUTPUT    =      true
#                     TIMEMORY_BANNER    =      true
#         TIMEMORY_COLLAPSE_PROCESSES    =      true
#           TIMEMORY_COLLAPSE_THREADS    =      true
#                TIMEMORY_COUT_OUTPUT    =      true
#               TIMEMORY_CPU_AFFINITY    =      false
#                    TIMEMORY_CRAYPAT    =
#      TIMEMORY_CUDA_EVENT_BATCH_SIZE    =      5
#       TIMEMORY_CUPTI_ACTIVITY_KINDS    =
#       TIMEMORY_CUPTI_ACTIVITY_LEVEL    =      1
#               TIMEMORY_CUPTI_DEVICE    =      0
#               TIMEMORY_CUPTI_EVENTS    =
#              TIMEMORY_CUPTI_METRICS    =
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
#                TIMEMORY_JSON_OUTPUT    =      true
#               TIMEMORY_LIBRARY_CTOR    =      true
#                  TIMEMORY_MAX_DEPTH    =      65535
#       TIMEMORY_MAX_THREAD_BOOKMARKS    =      50
#                  TIMEMORY_MAX_WIDTH    =      120
#           TIMEMORY_MEMORY_PRECISION    =      -1
#          TIMEMORY_MEMORY_SCIENTIFIC    =      false
#               TIMEMORY_MEMORY_UNITS    =
#               TIMEMORY_MEMORY_WIDTH    =      -1
#               TIMEMORY_MPI_FINALIZE    =      false
#                   TIMEMORY_MPI_INIT    =      false
#                 TIMEMORY_MPI_THREAD    =      true
#            TIMEMORY_MPI_THREAD_TYPE    =
#                 TIMEMORY_NODE_COUNT    =      0
#    TIMEMORY_NVTX_MARKER_DEVICE_SYNC    =      true
#                TIMEMORY_OUTPUT_PATH    =      timemory-ex-cpu-roofline-output
#              TIMEMORY_OUTPUT_PREFIX    =
#                TIMEMORY_PAPI_ATTACH    =      false
#                TIMEMORY_PAPI_EVENTS    =
#         TIMEMORY_PAPI_FAIL_ON_ERROR    =      false
#          TIMEMORY_PAPI_MULTIPLEXING    =      true
#              TIMEMORY_PAPI_OVERFLOW    =      0
#                 TIMEMORY_PAPI_QUIET    =      false
#                TIMEMORY_PLOT_OUTPUT    =      true
#                  TIMEMORY_PRECISION    =      -1
#                 TIMEMORY_PYTHON_EXE    =      /home/mhaseeb/repos/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/python-3.7.7-2dybrjceqs3qc4k7ci56t56bvzb4csxc/bin/python
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
#                 TIMEMORY_TARGET_PID    =      31883
#                TIMEMORY_TEXT_OUTPUT    =      true
#             TIMEMORY_THROTTLE_COUNT    =      10000
#             TIMEMORY_THROTTLE_VALUE    =      10000
#           TIMEMORY_TIMELINE_PROFILE    =      false
#                TIMEMORY_TIME_FORMAT    =      %F_%I.%M_%p
#                TIMEMORY_TIME_OUTPUT    =      false
#           TIMEMORY_TIMING_PRECISION    =      -1
#          TIMEMORY_TIMING_SCIENTIFIC    =      false
#               TIMEMORY_TIMING_UNITS    =
#               TIMEMORY_TIMING_WIDTH    =      -1
#             TIMEMORY_UPCXX_FINALIZE    =      true
#                 TIMEMORY_UPCXX_INIT    =      true
#                    TIMEMORY_VERBOSE    =      1
#                      TIMEMORY_WIDTH    =      -1
#----------------------------------------------------------------------------------------#
fibonacci(35) = 9227465.0
fibonacci(35) = 9227465.0
fibonacci(38) = 39088169.0
fibonacci(38) = 39088169.0
fibonacci(43) = 433494437.0
fibonacci(43) = 433494437.0
random_fibonacci(35) = 869057.3
random_fibonacci(35) = 905093.2
random_fibonacci(38) = 2986368.0
random_fibonacci(38) = 2947118.2
random_fibonacci(43) = 22552211.2
random_fibonacci(43) = 22966171.1
Total time:    4.309 sec wall
[demangle-test]> type: tim::component::wall_clock*
[demangle-test]> type: tim::component::wall_clock const*

[INFO]> L1 cache size: 32 KB, L2 cache size: 262 KB, L3 cache size: 15728 KB, max cache size: 15728 KB

[INFO]> num-threads      : 2
[INFO]> min-working-set  : 64 B
[INFO]> max-data-size    : 31457280 B
[INFO]> alignment        : 64
[INFO]> data type        : double

[ops_main] Executing 1 ops...
[ops_main] Executing 4 ops...
[ops_main] Executing 8 ops...
[cpu_roofline_op]|0> Outputting 'timemory-ex-cpu-roofline-output/cpu_roofline_op.json'...
[cpu_roofline_op]|0> Outputting 'timemory-ex-cpu-roofline-output/cpu_roofline_op.txt'...
Opening 'timemory-ex-cpu-roofline-output/cpu_roofline_op_dp_operations.jpeg' for output...
Closed 'timemory-ex-cpu-roofline-output/cpu_roofline_op_dp_operations.jpeg'...
Opening 'timemory-ex-cpu-roofline-output/cpu_roofline_op_runtime.jpeg' for output...
Closed 'timemory-ex-cpu-roofline-output/cpu_roofline_op_runtime.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MODEL USED TO PROVIDE PERFORMANCE RELATIVE TO THE PEAK POSSIBLE PERFORMANCE ON A CPU ARCHITECTURE.                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LABEL                                                                                                                                                              | COUNT                 | DEPTH    | METRIC          | UNITS         | SUM              | MEAN             | MIN              | MAX              | STDDEV         | % SELF      |
| ----------------------------                                                                                                                                       | --------              | -------- | --------------- | --------      | ---------------- | ---------------- | ---------------- | ---------------- | -------------  | --------    |
| >>> overall_timer                                                                                                                                                  | 1                     | 0        | DP_operations   |               | 4.000            | 4.000            | 4.000            | 4.000            | 0.000          | 0.0         |
|                                                                                                                                                                    |                       |          | Runtime         | sec           | 3.309            | 3.309            | 3.309            | 3.309            | 0.000          | 2.1         |
| >>>                                                                                                                                                                | _fibonacci(35)        | 2        | 1               | DP_operations |                  | 44791057.000     | 22395528.500     | 44791057.000     | 44791057.000   | 0.000       | 0.0 |
|                                                                                                                                                                    |                       |          | Runtime         | sec           | 0.068            | 0.034            | 0.068            | 0.068            | 0.000          | 2.1         |
| >>>                                                                                                                                                                | _fibonacci(38)        | 2        | 1               | DP_operations |                  | 189737959.000    | 94868979.500     | 189737959.000    | 189737959.000  | 0.000       | 0.0 |
|                                                                                                                                                                    |                       |          | Runtime         | sec           | 0.246            | 0.123            | 0.246            | 0.256            | 0.008          | 2.1         |
| >>>                                                                                                                                                                | _fibonacci(43)        | 2        | 1               | DP_operations |                  | 2104226200.000   | 1052113100.000   | 2104226200.000   | 2104226200.000 | 0.000       | 0.0 |
|                                                                                                                                                                    |                       |          | Runtime         | sec           | 2.336            | 1.168            | 2.336            | 2.366            | 0.021          | 2.1         |
| >>>                                                                                                                                                                | _random_fibonacci(35) | 2        | 1               | DP_operations |                  | 13919993.000     | 6959996.500      | 13919993.000     | 14486393.000   | 400505.281  | 0.0 |
|                                                                                                                                                                    |                       |          | Runtime         | sec           | 0.022            | 0.011            | 0.022            | 0.031            | 0.007          | 2.1         |
| >>>                                                                                                                                                                | _random_fibonacci(38) | 2        | 1               | DP_operations |                  | 47833538.000     | 23916769.000     | 47179658.000     | 47833538.000   | 462362.982  | 0.0 |
|                                                                                                                                                                    |                       |          | Runtime         | sec           | 0.067            | 0.034            | 0.067            | 0.069            | 0.002          | 2.1         |
| >>>                                                                                                                                                                | _random_fibonacci(43) | 2        | 1               | DP_operations |                  | 361020638.000    | 180510319.000    | 361020638.000    | 367673033.000  | 4703953.616 | 0.0 |
|                                                                                                                                                                    |                       |          | Runtime         | sec           | 0.502            | 0.251            | 0.502            | 0.515            | 0.009          | 2.1         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

[cpu_util]|0> Outputting 'timemory-ex-cpu-roofline-output/cpu_util.json'...
[cpu_util]|0> Outputting 'timemory-ex-cpu-roofline-output/cpu_util.txt'...
Opening 'timemory-ex-cpu-roofline-output/cpu_util.jpeg' for output...
Closed 'timemory-ex-cpu-roofline-output/cpu_util.jpeg'...

|-----------------------------------------------------------------------------------------------------------------|
| PERCENTAGE OF CPU-CLOCK TIME DIVIDED BY WALL-CLOCK TIME                                                           |
| ----------------------------------------------------------------------------------------------------------------- |
| LABEL                                                                                                             | COUNT                 | DEPTH   | METRIC     | UNITS    | SUM     | MEAN    | MIN     | MAX     | STDDEV   | % SELF   |
| ----------------------------                                                                                      | -------               | ------- | ---------- | -------  | ------- | ------- | ------- | ------- | -------- | -------- |
| >>> overall_timer                                                                                                 | 1                     | 0       | cpu_util   | %        | 197.9   | 197.9   | 197.9   | 197.9   | 0.0      | 0.0      |
| >>>                                                                                                               | _fibonacci(35)        | 2       | 1          | cpu_util | %       | 192.3   | 96.2    | 192.2   | 192.4    | 0.2      | 100.0 |
| >>>                                                                                                               | _fibonacci(38)        | 2       | 1          | cpu_util | %       | 195.2   | 97.6    | 195.0   | 195.4    | 0.3      | 100.0 |
| >>>                                                                                                               | _fibonacci(43)        | 2       | 1          | cpu_util | %       | 199.3   | 99.6    | 198.6   | 199.9    | 0.9      | 100.0 |
| >>>                                                                                                               | _random_fibonacci(35) | 2       | 1          | cpu_util | %       | 169.2   | 84.6    | 158.9   | 184.2    | 17.9     | 100.0 |
| >>>                                                                                                               | _random_fibonacci(38) | 2       | 1          | cpu_util | %       | 190.9   | 95.4    | 187.9   | 193.9    | 4.3      | 100.0 |
| >>>                                                                                                               | _random_fibonacci(43) | 2       | 1          | cpu_util | %       | 199.6   | 99.8    | 198.1   | 201.2    | 2.1      | 100.0 |
| ----------------------------------------------------------------------------------------------------------------- |

[cpu]|0> Outputting 'timemory-ex-cpu-roofline-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-ex-cpu-roofline-output/cpu.json'...
[cpu]|0> Outputting 'timemory-ex-cpu-roofline-output/cpu.txt'...
Opening 'timemory-ex-cpu-roofline-output/cpu.jpeg' for output...
Closed 'timemory-ex-cpu-roofline-output/cpu.jpeg'...

|----------------------------------------------------------------------------------------------------------------------|
| TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                                                                     |
| ---------------------------------------------------------------------------------------------------------------------- |
| LABEL                                                                                                                  | COUNT                 | DEPTH    | METRIC   | UNITS    | SUM      | MEAN     | MIN      | MAX      | STDDEV   | % SELF   |
| ----------------------------                                                                                           | --------              | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>> overall_timer                                                                                                      | 1                     | 0        | cpu      | sec      | 6.550    | 6.550    | 6.550    | 6.550    | 0.000    | 0.0      |
| >>>                                                                                                                    | _fibonacci(35)        | 2        | 1        | cpu      | sec      | 0.260    | 0.130    | 0.130    | 0.130    | 0.000    | 100.0 |
| >>>                                                                                                                    | _fibonacci(38)        | 2        | 1        | cpu      | sec      | 0.980    | 0.490    | 0.480    | 0.500    | 0.014    | 100.0 |
| >>>                                                                                                                    | _fibonacci(43)        | 2        | 1        | cpu      | sec      | 9.370    | 4.685    | 4.670    | 4.700    | 0.021    | 100.0 |
| >>>                                                                                                                    | _random_fibonacci(35) | 2        | 1        | cpu      | sec      | 0.090    | 0.045    | 0.040    | 0.050    | 0.007    | 100.0 |
| >>>                                                                                                                    | _random_fibonacci(38) | 2        | 1        | cpu      | sec      | 0.260    | 0.130    | 0.130    | 0.130    | 0.000    | 100.0 |
| >>>                                                                                                                    | _random_fibonacci(43) | 2        | 1        | cpu      | sec      | 2.030    | 1.015    | 1.010    | 1.020    | 0.007    | 100.0 |
| ---------------------------------------------------------------------------------------------------------------------- |

[wall]|0> Outputting 'timemory-ex-cpu-roofline-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-cpu-roofline-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-cpu-roofline-output/wall.txt'...
Opening 'timemory-ex-cpu-roofline-output/wall.jpeg' for output...
Closed 'timemory-ex-cpu-roofline-output/wall.jpeg'...

|----------------------------------------------------------------------------------------------------------------------|
| REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                               |
| ---------------------------------------------------------------------------------------------------------------------- |
| LABEL                                                                                                                  | COUNT                 | DEPTH    | METRIC   | UNITS    | SUM      | MEAN     | MIN      | MAX      | STDDEV   | % SELF   |
| ----------------------------                                                                                           | --------              | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| >>> overall_timer                                                                                                      | 1                     | 0        | wall     | sec      | 3.309    | 3.309    | 3.309    | 3.309    | 0.000    | 0.0      |
| >>>                                                                                                                    | _fibonacci(35)        | 2        | 1        | wall     | sec      | 0.135    | 0.068    | 0.068    | 0.068    | 0.000    | 100.0 |
| >>>                                                                                                                    | _fibonacci(38)        | 2        | 1        | wall     | sec      | 0.502    | 0.251    | 0.246    | 0.256    | 0.008    | 100.0 |
| >>>                                                                                                                    | _fibonacci(43)        | 2        | 1        | wall     | sec      | 4.702    | 2.351    | 2.336    | 2.366    | 0.021    | 100.0 |
| >>>                                                                                                                    | _random_fibonacci(35) | 2        | 1        | wall     | sec      | 0.053    | 0.027    | 0.022    | 0.031    | 0.007    | 100.0 |
| >>>                                                                                                                    | _random_fibonacci(38) | 2        | 1        | wall     | sec      | 0.136    | 0.068    | 0.067    | 0.069    | 0.002    | 100.0 |
| >>>                                                                                                                    | _random_fibonacci(43) | 2        | 1        | wall     | sec      | 1.017    | 0.508    | 0.502    | 0.515    | 0.009    | 100.0 |
| ---------------------------------------------------------------------------------------------------------------------- |


[metadata::manager::finalize]> Outputting 'timemory-ex-cpu-roofline-output/metadata.json'...
|0>>>  overall_timer :  [laps: 2]


#---------------------- tim::manager destroyed [rank=0][id=0][pid=31883] ----------------------#
```
