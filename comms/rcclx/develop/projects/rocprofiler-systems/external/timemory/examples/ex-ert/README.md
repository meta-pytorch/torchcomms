# ex-ert

These examples executes microkernels (with and without using Kokkos respectively) with varying workload (working set sizes and float/double types) to demonstrate the usage of empirical roofline toolkit (ERT) component for generation of performance data for roofline analysis. 

## Build

See [examples](../README.md##Build). Further, the ex-ert-kokkos example requires kokkos support enabled by enabling `-DTIMEMORY_BUILD_KOKKOS_TOOLS=ON` flag.

## Expected Output
```bash
$ ./ex_ert
#------------------------- tim::manager initialized [id=0][pid=8418] -------------------------#


[ert-example]> Executing run_ert_float_cpu_1_threads_64_min-ws_31457280_max-size...

[ert::executor]> working-set = 64, max-size = 31457280, num-thread = 1, num-stream = 0, grid-size = 0, block-size = 0, align-size = 32, data-type = float

|0>>>  run_ert_float_cpu_1_threads_64_min-ws_31457280_max-size :   11.325 sec wall,    0.030 sec sys,   11.300 sec user,  100.0 % cpu_util,   62.084 MB peak_rss,  [laps: 1]

[ert-example]> Executing run_ert_float_cpu_2_threads_64_min-ws_31457280_max-size...

[ert::executor]> working-set = 64, max-size = 31457280, num-thread = 2, num-stream = 0, grid-size = 0, block-size = 0, align-size = 32, data-type = float

|0>>>  run_ert_float_cpu_2_threads_64_min-ws_31457280_max-size :    5.413 sec wall,    0.020 sec sys,   10.800 sec user,  199.9 % cpu_util,   31.352 MB peak_rss,  [laps: 1]

[ert-example]> Executing run_ert_double_cpu_1_threads_64_min-ws_31457280_max-size...

[ert::executor]> working-set = 64, max-size = 31457280, num-thread = 1, num-stream = 0, grid-size = 0, block-size = 0, align-size = 64, data-type = double

|0>>>  run_ert_double_cpu_1_threads_64_min-ws_31457280_max-size :    2.325 sec wall,    0.020 sec sys,    2.300 sec user,   99.8 % cpu_util,   92.136 MB peak_rss,  [laps: 1]

[ert-example]> Executing run_ert_double_cpu_2_threads_64_min-ws_31457280_max-size...

[ert::executor]> working-set = 64, max-size = 31457280, num-thread = 2, num-stream = 0, grid-size = 0, block-size = 0, align-size = 64, data-type = double

|0>>>  run_ert_double_cpu_2_threads_64_min-ws_31457280_max-size :    1.217 sec wall,    0.000 sec sys,    2.440 sec user,  200.5 % cpu_util,    0.692 MB peak_rss,  [laps: 1]

[0]> Outputting 'timemory-ex-ert-output/ert_results.json'...
[peak_rss]|0> Outputting 'timemory-ex-ert-output/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-ex-ert-output/peak_rss.txt'...
Opening 'timemory-ex-ert-output/peak_rss.jpeg' for output...
Closed 'timemory-ex-ert-output/peak_rss.jpeg'...

|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                       MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF SWAP IS ENABLED                      |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                             LABEL                              | COUNT  | DEPTH  | METRIC   | UNITS  |  SUM    |  MEAN   |  MIN    |  MAX    | STDDEV | % SELF |
|----------------------------------------------------------------|--------|--------|----------|--------|---------|---------|---------|---------|--------|--------|
| >>> run_ert                                                    |      1 |      0 | peak_rss | MB     | 186.596 | 186.596 | 186.596 | 186.596 |  0.000 |    0.2 |
| >>> |_run_ert_float_cpu_1_threads_64_min-ws_31457280_max-size  |      1 |      1 | peak_rss | MB     |  62.084 |  62.084 |  62.084 |  62.084 |  0.000 |  100.0 |
| >>> |_run_ert_float_cpu_2_threads_64_min-ws_31457280_max-size  |      1 |      1 | peak_rss | MB     |  31.352 |  31.352 |  31.352 |  31.352 |  0.000 |  100.0 |
| >>> |_run_ert_double_cpu_1_threads_64_min-ws_31457280_max-size |      1 |      1 | peak_rss | MB     |  92.136 |  92.136 |  92.136 |  92.136 |  0.000 |  100.0 |
| >>> |_run_ert_double_cpu_2_threads_64_min-ws_31457280_max-size |      1 |      1 | peak_rss | MB     |   0.692 |   0.692 |   0.692 |   0.692 |  0.000 |  100.0 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------|

[cpu_util]|0> Outputting 'timemory-ex-ert-output/cpu_util.json'...
[cpu_util]|0> Outputting 'timemory-ex-ert-output/cpu_util.txt'...
Opening 'timemory-ex-ert-output/cpu_util.jpeg' for output...
Closed 'timemory-ex-ert-output/cpu_util.jpeg'...

|-----------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                PERCENTAGE OF CPU-CLOCK TIME DIVIDED BY WALL-CLOCK TIME                                              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|
|                             LABEL                              | COUNT | DEPTH | METRIC   | UNITS | SUM   | MEAN  | MIN   | MAX   | STDDEV | % SELF |
|----------------------------------------------------------------|-------|-------|----------|-------|-------|-------|-------|-------|--------|--------|
| >>> run_ert                                                    |     1 |     0 | cpu_util | %     |   0.0 |   0.0 |   0.0 |   0.0 |    0.0 |    0.0 |
| >>> |_run_ert_float_cpu_1_threads_64_min-ws_31457280_max-size  |     1 |     1 | cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
| >>> |_run_ert_float_cpu_2_threads_64_min-ws_31457280_max-size  |     1 |     1 | cpu_util | %     | 199.9 | 199.9 | 199.9 | 199.9 |    0.0 |  100.0 |
| >>> |_run_ert_double_cpu_1_threads_64_min-ws_31457280_max-size |     1 |     1 | cpu_util | %     |  99.8 |  99.8 |  99.8 |  99.8 |    0.0 |  100.0 |
| >>> |_run_ert_double_cpu_2_threads_64_min-ws_31457280_max-size |     1 |     1 | cpu_util | %     | 200.5 | 200.5 | 200.5 | 200.5 |    0.0 |  100.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|

[user]|0> Outputting 'timemory-ex-ert-output/user.flamegraph.json'...
[user]|0> Outputting 'timemory-ex-ert-output/user.json'...
[user]|0> Outputting 'timemory-ex-ert-output/user.txt'...
Opening 'timemory-ex-ert-output/user.jpeg' for output...
Closed 'timemory-ex-ert-output/user.jpeg'...

|----------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                CPU TIME SPENT IN USER-MODE                                                               |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|
|                             LABEL                              | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|----------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> run_ert                                                    |      1 |      0 | user   | sec    | 27.810 | 27.810 | 27.810 | 27.810 |  0.000 |    3.5 |
| >>> |_run_ert_float_cpu_1_threads_64_min-ws_31457280_max-size  |      1 |      1 | user   | sec    | 11.300 | 11.300 | 11.300 | 11.300 |  0.000 |  100.0 |
| >>> |_run_ert_float_cpu_2_threads_64_min-ws_31457280_max-size  |      1 |      1 | user   | sec    | 10.800 | 10.800 | 10.800 | 10.800 |  0.000 |  100.0 |
| >>> |_run_ert_double_cpu_1_threads_64_min-ws_31457280_max-size |      1 |      1 | user   | sec    |  2.300 |  2.300 |  2.300 |  2.300 |  0.000 |  100.0 |
| >>> |_run_ert_double_cpu_2_threads_64_min-ws_31457280_max-size |      1 |      1 | user   | sec    |  2.440 |  2.440 |  2.440 |  2.440 |  0.000 |  100.0 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|

[sys]|0> Outputting 'timemory-ex-ert-output/sys.flamegraph.json'...
[sys]|0> Outputting 'timemory-ex-ert-output/sys.json'...
[sys]|0> Outputting 'timemory-ex-ert-output/sys.txt'...
Opening 'timemory-ex-ert-output/sys.jpeg' for output...
Closed 'timemory-ex-ert-output/sys.jpeg'...

|----------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                               CPU TIME SPENT IN KERNEL-MODE                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|
|                             LABEL                              | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|----------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> run_ert                                                    |      1 |      0 | sys    | sec    |  0.350 |  0.350 |  0.350 |  0.350 |  0.000 |   80.0 |
| >>> |_run_ert_float_cpu_1_threads_64_min-ws_31457280_max-size  |      1 |      1 | sys    | sec    |  0.030 |  0.030 |  0.030 |  0.030 |  0.000 |  100.0 |
| >>> |_run_ert_float_cpu_2_threads_64_min-ws_31457280_max-size  |      1 |      1 | sys    | sec    |  0.020 |  0.020 |  0.020 |  0.020 |  0.000 |  100.0 |
| >>> |_run_ert_double_cpu_1_threads_64_min-ws_31457280_max-size |      1 |      1 | sys    | sec    |  0.020 |  0.020 |  0.020 |  0.020 |  0.000 |  100.0 |
| >>> |_run_ert_double_cpu_2_threads_64_min-ws_31457280_max-size |      1 |      1 | sys    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-ert-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-ert-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-ert-output/wall.txt'...
Opening 'timemory-ex-ert-output/wall.jpeg' for output...
Closed 'timemory-ex-ert-output/wall.jpeg'...

|----------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                         REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|
|                             LABEL                              | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|----------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> run_ert                                                    |      1 |      0 | wall   | sec    | 22.618 | 22.618 | 22.618 | 22.618 |  0.000 |   10.3 |
| >>> |_run_ert_float_cpu_1_threads_64_min-ws_31457280_max-size  |      1 |      1 | wall   | sec    | 11.325 | 11.325 | 11.325 | 11.325 |  0.000 |  100.0 |
| >>> |_run_ert_float_cpu_2_threads_64_min-ws_31457280_max-size  |      1 |      1 | wall   | sec    |  5.413 |  5.413 |  5.413 |  5.413 |  0.000 |  100.0 |
| >>> |_run_ert_double_cpu_1_threads_64_min-ws_31457280_max-size |      1 |      1 | wall   | sec    |  2.325 |  2.325 |  2.325 |  2.325 |  0.000 |  100.0 |
| >>> |_run_ert_double_cpu_2_threads_64_min-ws_31457280_max-size |      1 |      1 | wall   | sec    |  1.217 |  1.217 |  1.217 |  1.217 |  0.000 |  100.0 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|


[metadata::manager::finalize]> Outputting 'timemory-ex-ert-output/metadata.json'...


#---------------------- tim::manager destroyed [rank=0][id=0][pid=8418] ----------------------#
```