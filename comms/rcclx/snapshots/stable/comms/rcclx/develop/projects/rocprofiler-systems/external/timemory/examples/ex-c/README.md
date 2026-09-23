# ex-c

This example demonstrates the use of timemory timing components (timers), resident set size (rss) and context switches for instrumentation of C code. The instrumentation marker containing the following components is used for instrumentation.

```c
TIMEMORY_MARKER(func, WALL_CLOCK, SYS_CLOCK, USER_CLOCK, CPU_CLOCK,
                      CPU_UTIL, PAGE_RSS, PEAK_RSS, PRIORITY_CONTEXT_SWITCH,
                      VOLUNTARY_CONTEXT_SWITCH, CALIPER);
```

## Build

See [examples](../README.md##Build). This examples requires an additional `-DTIMEMORY_BUILD_C=ON` flag to be built.

## Expected Output

```console
$ ./ex_c_timing
'/Users/jrmadsen/devel/c++/timemory-develop/examples/ex-c/ex_c_timing.c' : main @ 105. Running fibonacci(43, 28)...
main (untimed): fibonacci(43, 43) = 433494437
main (timed): fibonacci(43, 28) = 433494437
# laps = 9576

'/Users/jrmadsen/devel/c++/timemory-develop/examples/ex-c/ex_c_timing.c' : main @ 134 --> n = 433494437, 433494437, 433494437
[cpu_util]|0> Outputting 'timemory-ex-c-timing-output/cpu_util.json'...
[cpu_util]|0> Outputting 'timemory-ex-c-timing-output/cpu_util.tree.json'...
[cpu_util]|0> Outputting 'timemory-ex-c-timing-output/cpu_util.txt'...

|-------------------------------------------------------------------------------------------------------------------------------------------------|
|                                              PERCENTAGE OF CPU-CLOCK TIME DIVIDED BY WALL-CLOCK TIME                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------|
|                           LABEL                            | COUNT | DEPTH | METRIC   | UNITS | SUM   | MEAN  | MIN   | MAX   | STDDEV | % SELF |
|------------------------------------------------------------|-------|-------|----------|-------|-------|-------|-------|-------|--------|--------|
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]            |     1 |     0 | cpu_util | %     |  99.5 |  99.5 |  99.5 |  99.5 |    0.0 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]      |     1 |     0 | cpu_util | %     |  98.6 |  98.6 |  98.6 |  98.6 |    0.0 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]      |     1 |     0 | cpu_util | %     |  99.7 |  99.7 |  99.7 |  99.7 |    0.0 |    0.0 |
| >>> |_fibonacci[using_tuple=0]                             |     1 |     1 | cpu_util | %     |  99.7 |  99.7 |  99.7 |  99.7 |    0.0 |    0.0 |
| >>>   |_fibonacci[using_tuple=0]                           |     2 |     2 | cpu_util | %     |  99.7 |  49.9 |  99.6 |  99.9 |    0.2 |    0.0 |
| >>>     |_fibonacci[using_tuple=0]                         |     4 |     3 | cpu_util | %     |  99.7 |  24.9 |  98.6 | 100.8 |    0.9 |    0.0 |
| >>>       |_fibonacci[using_tuple=0]                       |     8 |     4 | cpu_util | %     |  99.7 |  12.5 |  97.4 | 103.1 |    1.9 |    0.0 |
| >>>         |_fibonacci[using_tuple=0]                     |    16 |     5 | cpu_util | %     |  99.7 |   6.2 |  91.4 | 110.2 |    5.2 |    0.0 |
| >>>           |_fibonacci[using_tuple=0]                   |    32 |     6 | cpu_util | %     |  99.7 |   3.1 |  76.8 | 140.7 |   14.2 |    0.0 |
| >>>             |_fibonacci[using_tuple=0]                 |    64 |     7 | cpu_util | %     |  99.8 |   1.6 |  62.2 | 186.0 |   24.5 |    0.0 |
| >>>               |_fibonacci[using_tuple=0]               |   128 |     8 | cpu_util | %     |  99.8 |   0.8 |   0.0 | 493.8 |   66.0 |    1.0 |
| >>>                 |_fibonacci[using_tuple=0]             |   247 |     9 | cpu_util | %     |  98.8 |   0.4 |   0.0 | 486.7 |  104.5 |    0.0 |
| >>>                   |_fibonacci[using_tuple=0]           |   382 |    10 | cpu_util | %     |  99.2 |   0.3 |   0.0 | 494.0 |  142.3 |    0.0 |
| >>>                     |_fibonacci[using_tuple=0]         |   386 |    11 | cpu_util | %     | 100.6 |   0.3 |   0.0 | 493.9 |  159.7 |    3.7 |
| >>>                       |_fibonacci[using_tuple=0]       |   232 |    12 | cpu_util | %     |  96.8 |   0.4 |   0.0 | 493.9 |  175.5 |   10.4 |
| >>>                         |_fibonacci[using_tuple=0]     |    79 |    13 | cpu_util | %     |  86.8 |   1.1 |   0.0 | 494.0 |  175.5 |    0.0 |
| >>>                           |_fibonacci[using_tuple=0]   |    14 |    14 | cpu_util | %     |  95.6 |   6.8 |   0.0 | 432.9 |  162.1 |    0.0 |
| >>>                             |_fibonacci[using_tuple=0] |     1 |    15 | cpu_util | %     | 449.2 | 449.2 | 449.2 | 449.2 |    0.0 |  100.0 |
|-------------------------------------------------------------------------------------------------------------------------------------------------|

[cpu]|0> Outputting 'timemory-ex-c-timing-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-ex-c-timing-output/cpu.json'...
[cpu]|0> Outputting 'timemory-ex-c-timing-output/cpu.tree.json'...
[cpu]|0> Outputting 'timemory-ex-c-timing-output/cpu.txt'...

|------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                  TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                                                  |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
|                           LABEL                            | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]            |      1 |      0 | cpu    | sec    |  1.830 |  1.830 |  1.830 |  1.830 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]      |      1 |      0 | cpu    | sec    |  1.870 |  1.870 |  1.870 |  1.870 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]      |      1 |      0 | cpu    | sec    |  1.850 |  1.850 |  1.850 |  1.850 |  0.000 |    0.0 |
| >>> |_fibonacci[using_tuple=0]                             |      1 |      1 | cpu    | sec    |  1.850 |  1.850 |  1.850 |  1.850 |  0.000 |    0.0 |
| >>>   |_fibonacci[using_tuple=0]                           |      2 |      2 | cpu    | sec    |  1.850 |  0.925 |  0.700 |  1.150 |  0.318 |    0.0 |
| >>>     |_fibonacci[using_tuple=0]                         |      4 |      3 | cpu    | sec    |  1.850 |  0.463 |  0.270 |  0.720 |  0.188 |    0.0 |
| >>>       |_fibonacci[using_tuple=0]                       |      8 |      4 | cpu    | sec    |  1.850 |  0.231 |  0.100 |  0.450 |  0.108 |    0.0 |
| >>>         |_fibonacci[using_tuple=0]                     |     16 |      5 | cpu    | sec    |  1.850 |  0.116 |  0.040 |  0.290 |  0.061 |    0.0 |
| >>>           |_fibonacci[using_tuple=0]                   |     32 |      6 | cpu    | sec    |  1.850 |  0.058 |  0.020 |  0.180 |  0.034 |    0.0 |
| >>>             |_fibonacci[using_tuple=0]                 |     64 |      7 | cpu    | sec    |  1.850 |  0.029 |  0.010 |  0.110 |  0.019 |    0.0 |
| >>>               |_fibonacci[using_tuple=0]               |    128 |      8 | cpu    | sec    |  1.850 |  0.014 |  0.000 |  0.070 |  0.011 |    1.6 |
| >>>                 |_fibonacci[using_tuple=0]             |    247 |      9 | cpu    | sec    |  1.820 |  0.007 |  0.000 |  0.050 |  0.007 |    7.1 |
| >>>                   |_fibonacci[using_tuple=0]           |    382 |     10 | cpu    | sec    |  1.690 |  0.004 |  0.000 |  0.030 |  0.006 |   24.9 |
| >>>                     |_fibonacci[using_tuple=0]         |    386 |     11 | cpu    | sec    |  1.270 |  0.003 |  0.000 |  0.020 |  0.005 |   51.2 |
| >>>                       |_fibonacci[using_tuple=0]       |    232 |     12 | cpu    | sec    |  0.620 |  0.003 |  0.000 |  0.010 |  0.004 |   72.6 |
| >>>                         |_fibonacci[using_tuple=0]     |     79 |     13 | cpu    | sec    |  0.170 |  0.002 |  0.000 |  0.010 |  0.004 |   82.4 |
| >>>                           |_fibonacci[using_tuple=0]   |     14 |     14 | cpu    | sec    |  0.030 |  0.002 |  0.000 |  0.010 |  0.004 |   66.7 |
| >>>                             |_fibonacci[using_tuple=0] |      1 |     15 | cpu    | sec    |  0.010 |  0.010 |  0.010 |  0.010 |  0.000 |  100.0 |
|------------------------------------------------------------------------------------------------------------------------------------------------------|

[user]|0> Outputting 'timemory-ex-c-timing-output/user.flamegraph.json'...
[user]|0> Outputting 'timemory-ex-c-timing-output/user.json'...
[user]|0> Outputting 'timemory-ex-c-timing-output/user.tree.json'...
[user]|0> Outputting 'timemory-ex-c-timing-output/user.txt'...

|------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                              CPU TIME SPENT IN USER-MODE                                                             |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
|                           LABEL                            | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]            |      1 |      0 | user   | sec    |  1.820 |  1.820 |  1.820 |  1.820 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]      |      1 |      0 | user   | sec    |  1.850 |  1.850 |  1.850 |  1.850 |  0.000 |    0.0 |
| >>> |_fibonacci[using_tuple=1]                             |      1 |      1 | user   | sec    |  1.850 |  1.850 |  1.850 |  1.850 |  0.000 |    0.0 |
| >>>   |_fibonacci[using_tuple=1]                           |      2 |      2 | user   | sec    |  1.850 |  0.925 |  0.690 |  1.160 |  0.332 |    0.0 |
| >>>     |_fibonacci[using_tuple=1]                         |      4 |      3 | user   | sec    |  1.850 |  0.463 |  0.260 |  0.710 |  0.186 |    0.0 |
| >>>       |_fibonacci[using_tuple=1]                       |      8 |      4 | user   | sec    |  1.850 |  0.231 |  0.090 |  0.430 |  0.103 |    0.0 |
| >>>         |_fibonacci[using_tuple=1]                     |     16 |      5 | user   | sec    |  1.850 |  0.116 |  0.030 |  0.270 |  0.059 |    0.5 |
| >>>           |_fibonacci[using_tuple=1]                   |     32 |      6 | user   | sec    |  1.840 |  0.058 |  0.010 |  0.160 |  0.033 |    0.0 |
| >>>             |_fibonacci[using_tuple=1]                 |     64 |      7 | user   | sec    |  1.840 |  0.029 |  0.000 |  0.100 |  0.019 |    0.0 |
| >>>               |_fibonacci[using_tuple=1]               |    128 |      8 | user   | sec    |  1.840 |  0.014 |  0.000 |  0.060 |  0.011 |    1.1 |
| >>>                 |_fibonacci[using_tuple=1]             |    247 |      9 | user   | sec    |  1.820 |  0.007 |  0.000 |  0.040 |  0.007 |    7.1 |
| >>>                   |_fibonacci[using_tuple=1]           |    382 |     10 | user   | sec    |  1.690 |  0.004 |  0.000 |  0.020 |  0.005 |   25.4 |
| >>>                     |_fibonacci[using_tuple=1]         |    386 |     11 | user   | sec    |  1.260 |  0.003 |  0.000 |  0.010 |  0.005 |   47.6 |
| >>>                       |_fibonacci[using_tuple=1]       |    232 |     12 | user   | sec    |  0.660 |  0.003 |  0.000 |  0.010 |  0.005 |   71.2 |
| >>>                         |_fibonacci[using_tuple=1]     |     79 |     13 | user   | sec    |  0.190 |  0.002 |  0.000 |  0.010 |  0.004 |   68.4 |
| >>>                           |_fibonacci[using_tuple=1]   |     14 |     14 | user   | sec    |  0.060 |  0.004 |  0.000 |  0.010 |  0.005 |  100.0 |
| >>>                             |_fibonacci[using_tuple=1] |      1 |     15 | user   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]      |      1 |      0 | user   | sec    |  1.840 |  1.840 |  1.840 |  1.840 |  0.000 |  100.0 |
|------------------------------------------------------------------------------------------------------------------------------------------------------|

[sys]|0> Outputting 'timemory-ex-c-timing-output/sys.flamegraph.json'...
[sys]|0> Outputting 'timemory-ex-c-timing-output/sys.json'...
[sys]|0> Outputting 'timemory-ex-c-timing-output/sys.tree.json'...
[sys]|0> Outputting 'timemory-ex-c-timing-output/sys.txt'...

|------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                             CPU TIME SPENT IN KERNEL-MODE                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
|                           LABEL                            | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]            |      1 |      0 | sys    | sec    |  0.010 |  0.010 |  0.010 |  0.010 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]      |      1 |      0 | sys    | sec    |  0.020 |  0.020 |  0.020 |  0.020 |  0.000 |    0.0 |
| >>> |_fibonacci[using_tuple=1]                             |      1 |      1 | sys    | sec    |  0.020 |  0.020 |  0.020 |  0.020 |  0.000 |    0.0 |
| >>>   |_fibonacci[using_tuple=1]                           |      2 |      2 | sys    | sec    |  0.020 |  0.010 |  0.010 |  0.010 |  0.000 |    0.0 |
| >>>     |_fibonacci[using_tuple=1]                         |      4 |      3 | sys    | sec    |  0.020 |  0.005 |  0.000 |  0.010 |  0.006 |    0.0 |
| >>>       |_fibonacci[using_tuple=1]                       |      8 |      4 | sys    | sec    |  0.020 |  0.003 |  0.000 |  0.010 |  0.005 |    0.0 |
| >>>         |_fibonacci[using_tuple=1]                     |     16 |      5 | sys    | sec    |  0.020 |  0.001 |  0.000 |  0.010 |  0.003 |    0.0 |
| >>>           |_fibonacci[using_tuple=1]                   |     32 |      6 | sys    | sec    |  0.020 |  0.001 |  0.000 |  0.010 |  0.002 |    0.0 |
| >>>             |_fibonacci[using_tuple=1]                 |     64 |      7 | sys    | sec    |  0.020 |  0.000 |  0.000 |  0.010 |  0.002 |    0.0 |
| >>>               |_fibonacci[using_tuple=1]               |    128 |      8 | sys    | sec    |  0.020 |  0.000 |  0.000 |  0.010 |  0.001 |    0.0 |
| >>>                 |_fibonacci[using_tuple=1]             |    247 |      9 | sys    | sec    |  0.020 |  0.000 |  0.000 |  0.010 |  0.001 |    0.0 |
| >>>                   |_fibonacci[using_tuple=1]           |    382 |     10 | sys    | sec    |  0.020 |  0.000 |  0.000 |  0.010 |  0.001 |   50.0 |
| >>>                     |_fibonacci[using_tuple=1]         |    386 |     11 | sys    | sec    |  0.010 |  0.000 |  0.000 |  0.010 |  0.001 |    0.0 |
| >>>                       |_fibonacci[using_tuple=1]       |    232 |     12 | sys    | sec    |  0.010 |  0.000 |  0.000 |  0.010 |  0.001 |  100.0 |
| >>>                         |_fibonacci[using_tuple=1]     |     79 |     13 | sys    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>>                           |_fibonacci[using_tuple=1]   |     14 |     14 | sys    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>>                             |_fibonacci[using_tuple=1] |      1 |     15 | sys    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]      |      1 |      0 | sys    | sec    |  0.010 |  0.010 |  0.010 |  0.010 |  0.000 |  100.0 |
|------------------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-c-timing-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-c-timing-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-c-timing-output/wall.tree.json'...
[wall]|0> Outputting 'timemory-ex-c-timing-output/wall.txt'...

|------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                       REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                       |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
|                           LABEL                            | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]            |      1 |      0 | wall   | sec    |  1.839 |  1.839 |  1.839 |  1.839 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]      |      1 |      0 | wall   | sec    |  1.896 |  1.896 |  1.896 |  1.896 |  0.000 |    0.0 |
| >>> |_fibonacci[using_tuple=1]                             |      1 |      1 | wall   | sec    |  1.896 |  1.896 |  1.896 |  1.896 |  0.000 |    0.0 |
| >>>   |_fibonacci[using_tuple=1]                           |      2 |      2 | wall   | sec    |  1.896 |  0.948 |  0.709 |  1.187 |  0.339 |    0.0 |
| >>>     |_fibonacci[using_tuple=1]                         |      4 |      3 | wall   | sec    |  1.896 |  0.474 |  0.265 |  0.737 |  0.195 |    0.0 |
| >>>       |_fibonacci[using_tuple=1]                       |      8 |      4 | wall   | sec    |  1.896 |  0.237 |  0.100 |  0.447 |  0.107 |    0.0 |
| >>>         |_fibonacci[using_tuple=1]                     |     16 |      5 | wall   | sec    |  1.896 |  0.118 |  0.038 |  0.279 |  0.061 |    0.0 |
| >>>           |_fibonacci[using_tuple=1]                   |     32 |      6 | wall   | sec    |  1.896 |  0.059 |  0.015 |  0.170 |  0.034 |    0.0 |
| >>>             |_fibonacci[using_tuple=1]                 |     64 |      7 | wall   | sec    |  1.895 |  0.030 |  0.006 |  0.107 |  0.019 |    0.0 |
| >>>               |_fibonacci[using_tuple=1]               |    128 |      8 | wall   | sec    |  1.894 |  0.015 |  0.002 |  0.068 |  0.010 |    0.7 |
| >>>                 |_fibonacci[using_tuple=1]             |    247 |      9 | wall   | sec    |  1.881 |  0.008 |  0.002 |  0.042 |  0.006 |    7.7 |
| >>>                   |_fibonacci[using_tuple=1]           |    382 |     10 | wall   | sec    |  1.736 |  0.005 |  0.002 |  0.026 |  0.003 |   26.1 |
| >>>                     |_fibonacci[using_tuple=1]         |    386 |     11 | wall   | sec    |  1.282 |  0.003 |  0.002 |  0.017 |  0.002 |   49.6 |
| >>>                       |_fibonacci[using_tuple=1]       |    232 |     12 | wall   | sec    |  0.646 |  0.003 |  0.002 |  0.010 |  0.001 |   68.7 |
| >>>                         |_fibonacci[using_tuple=1]     |     79 |     13 | wall   | sec    |  0.203 |  0.003 |  0.002 |  0.006 |  0.001 |   83.3 |
| >>>                           |_fibonacci[using_tuple=1]   |     14 |     14 | wall   | sec    |  0.034 |  0.002 |  0.002 |  0.004 |  0.001 |   93.8 |
| >>>                             |_fibonacci[using_tuple=1] |      1 |     15 | wall   | sec    |  0.002 |  0.002 |  0.002 |  0.002 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]      |      1 |      0 | wall   | sec    |  1.855 |  1.855 |  1.855 |  1.855 |  0.000 |    0.0 |
| >>> |_fibonacci[using_tuple=0]                             |      1 |      1 | wall   | sec    |  1.855 |  1.855 |  1.855 |  1.855 |  0.000 |    0.0 |
| >>>   |_fibonacci[using_tuple=0]                           |      2 |      2 | wall   | sec    |  1.855 |  0.928 |  0.700 |  1.155 |  0.321 |    0.0 |
| >>>     |_fibonacci[using_tuple=0]                         |      4 |      3 | wall   | sec    |  1.855 |  0.464 |  0.274 |  0.725 |  0.189 |    0.0 |
| >>>       |_fibonacci[using_tuple=0]                       |      8 |      4 | wall   | sec    |  1.855 |  0.232 |  0.101 |  0.455 |  0.109 |    0.0 |
| >>>         |_fibonacci[using_tuple=0]                     |     16 |      5 | wall   | sec    |  1.855 |  0.116 |  0.038 |  0.291 |  0.063 |    0.0 |
| >>>           |_fibonacci[using_tuple=0]                   |     32 |      6 | wall   | sec    |  1.855 |  0.058 |  0.014 |  0.181 |  0.035 |    0.0 |
| >>>             |_fibonacci[using_tuple=0]                 |     64 |      7 | wall   | sec    |  1.855 |  0.029 |  0.005 |  0.109 |  0.019 |    0.0 |
| >>>               |_fibonacci[using_tuple=0]               |    128 |      8 | wall   | sec    |  1.854 |  0.014 |  0.002 |  0.066 |  0.010 |    0.7 |
| >>>                 |_fibonacci[using_tuple=0]             |    247 |      9 | wall   | sec    |  1.841 |  0.007 |  0.002 |  0.041 |  0.006 |    7.5 |
| >>>                   |_fibonacci[using_tuple=0]           |    382 |     10 | wall   | sec    |  1.704 |  0.004 |  0.002 |  0.025 |  0.003 |   25.9 |
| >>>                     |_fibonacci[using_tuple=0]         |    386 |     11 | wall   | sec    |  1.263 |  0.003 |  0.002 |  0.015 |  0.002 |   49.3 |
| >>>                       |_fibonacci[using_tuple=0]       |    232 |     12 | wall   | sec    |  0.640 |  0.003 |  0.002 |  0.009 |  0.001 |   69.4 |
| >>>                         |_fibonacci[using_tuple=0]     |     79 |     13 | wall   | sec    |  0.196 |  0.002 |  0.002 |  0.006 |  0.001 |   84.0 |
| >>>                           |_fibonacci[using_tuple=0]   |     14 |     14 | wall   | sec    |  0.031 |  0.002 |  0.002 |  0.004 |  0.000 |   92.9 |
| >>>                             |_fibonacci[using_tuple=0] |      1 |     15 | wall   | sec    |  0.002 |  0.002 |  0.002 |  0.002 |  0.000 |  100.0 |
|------------------------------------------------------------------------------------------------------------------------------------------------------|

[vol_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/vol_cxt_swch.json'...
[vol_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/vol_cxt_swch.tree.json'...
[vol_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/vol_cxt_swch.txt'...

|---------------------------------------------------------------------------------------------------------------------------------|
|        NUMBER OF CONTEXT SWITCHES DUE TO A PROCESS VOLUNTARILY GIVING UP THE PROCESSOR BEFORE ITS TIME SLICE WAS COMPLETED      |
|---------------------------------------------------------------------------------------------------------------------------------|
|                        LABEL                          | COUNT | DEPTH |    METRIC    | SUM | MEAN | MIN | MAX | STDDEV | % SELF |
|-------------------------------------------------------|-------|-------|--------------|-----|------|-----|-----|--------|--------|
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]       |     1 |     0 | vol_cxt_swch |   0 |    0 |   0 |   0 |      0 |      0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)] |     1 |     0 | vol_cxt_swch |   0 |    0 |   0 |   0 |      0 |      0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)] |     1 |     0 | vol_cxt_swch |   0 |    0 |   0 |   0 |      0 |      0 |
|---------------------------------------------------------------------------------------------------------------------------------|

[prio_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/prio_cxt_swch.json'...
[prio_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/prio_cxt_swch.tree.json'...
[prio_cxt_swch]|0> Outputting 'timemory-ex-c-timing-output/prio_cxt_swch.txt'...

|-------------------------------------------------------------------------------------------------------------------------------------|
|   NUMBER OF CONTEXT SWITCH DUE TO HIGHER PRIORITY PROCESS BECOMING RUNNABLE OR BECAUSE THE CURRENT PROCESS EXCEEDED ITS TIME SLICE  |
|-------------------------------------------------------------------------------------------------------------------------------------|
|                        LABEL                          | COUNT | DEPTH |    METRIC     | SUM  | MEAN | MIN  | MAX  | STDDEV | % SELF |
|-------------------------------------------------------|-------|-------|---------------|------|------|------|------|--------|--------|
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]       |     1 |     0 | prio_cxt_swch | 1451 | 1451 | 1451 | 1451 |      0 |    100 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)] |     1 |     0 | prio_cxt_swch | 2607 | 2607 | 2607 | 2607 |      0 |    100 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)] |     1 |     0 | prio_cxt_swch | 1483 | 1483 | 1483 | 1483 |      0 |    100 |
|-------------------------------------------------------------------------------------------------------------------------------------|

[peak_rss]|0> Outputting 'timemory-ex-c-timing-output/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-ex-c-timing-output/peak_rss.tree.json'...
[peak_rss]|0> Outputting 'timemory-ex-c-timing-output/peak_rss.txt'...

|--------------------------------------------------------------------------------------------------------------------------------------------------------|
|                   MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF SWAP IS ENABLED                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|
|                           LABEL                            | COUNT  | DEPTH  | METRIC   | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------------------|--------|--------|----------|--------|--------|--------|--------|--------|--------|--------|
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]            |      1 |      0 | peak_rss | MB     |  0.012 |  0.012 |  0.012 |  0.012 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)]      |      1 |      0 | peak_rss | MB     |  0.152 |  0.152 |  0.152 |  0.152 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)]      |      1 |      0 | peak_rss | MB     |  0.098 |  0.098 |  0.098 |  0.098 |  0.000 |   54.2 |
| >>> |_fibonacci[using_tuple=0]                             |      1 |      1 | peak_rss | MB     |  0.045 |  0.045 |  0.045 |  0.045 |  0.000 |    9.1 |
| >>>   |_fibonacci[using_tuple=0]                           |      2 |      2 | peak_rss | MB     |  0.041 |  0.020 |  0.000 |  0.041 |  0.029 |    0.0 |
| >>>     |_fibonacci[using_tuple=0]                         |      4 |      3 | peak_rss | MB     |  0.041 |  0.010 |  0.000 |  0.041 |  0.020 |   10.0 |
| >>>       |_fibonacci[using_tuple=0]                       |      8 |      4 | peak_rss | MB     |  0.037 |  0.005 |  0.000 |  0.037 |  0.013 |    0.0 |
| >>>         |_fibonacci[using_tuple=0]                     |     16 |      5 | peak_rss | MB     |  0.037 |  0.002 |  0.000 |  0.037 |  0.009 |   11.1 |
| >>>           |_fibonacci[using_tuple=0]                   |     32 |      6 | peak_rss | MB     |  0.033 |  0.001 |  0.000 |  0.029 |  0.005 |    0.0 |
| >>>             |_fibonacci[using_tuple=0]                 |     64 |      7 | peak_rss | MB     |  0.033 |  0.001 |  0.000 |  0.029 |  0.004 |    0.0 |
| >>>               |_fibonacci[using_tuple=0]               |    128 |      8 | peak_rss | MB     |  0.033 |  0.000 |  0.000 |  0.016 |  0.002 |   12.5 |
| >>>                 |_fibonacci[using_tuple=0]             |    247 |      9 | peak_rss | MB     |  0.029 |  0.000 |  0.000 |  0.012 |  0.001 |    0.0 |
| >>>                   |_fibonacci[using_tuple=0]           |    382 |     10 | peak_rss | MB     |  0.029 |  0.000 |  0.000 |  0.012 |  0.001 |   57.1 |
| >>>                     |_fibonacci[using_tuple=0]         |    386 |     11 | peak_rss | MB     |  0.012 |  0.000 |  0.000 |  0.008 |  0.000 |    0.0 |
| >>>                       |_fibonacci[using_tuple=0]       |    232 |     12 | peak_rss | MB     |  0.012 |  0.000 |  0.000 |  0.008 |  0.001 |    0.0 |
| >>>                         |_fibonacci[using_tuple=0]     |     79 |     13 | peak_rss | MB     |  0.012 |  0.000 |  0.000 |  0.008 |  0.001 |   66.7 |
| >>>                           |_fibonacci[using_tuple=0]   |     14 |     14 | peak_rss | MB     |  0.004 |  0.000 |  0.000 |  0.004 |  0.001 |    0.0 |
| >>>                             |_fibonacci[using_tuple=0] |      1 |     15 | peak_rss | MB     |  0.004 |  0.004 |  0.004 |  0.004 |  0.000 |  100.0 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|

[page_rss]|0> Outputting 'timemory-ex-c-timing-output/page_rss.json'...
[page_rss]|0> Outputting 'timemory-ex-c-timing-output/page_rss.tree.json'...
[page_rss]|0> Outputting 'timemory-ex-c-timing-output/page_rss.txt'...

|---------------------------------------------------------------------------------------------------------------------------------------------------|
|                  AMOUNT OF MEMORY ALLOCATED IN PAGES OF MEMORY. UNLIKE PEAK_RSS, VALUE WILL FLUCTUATE AS MEMORY IS FREED/ALLOCATED                |
|---------------------------------------------------------------------------------------------------------------------------------------------------|
|                        LABEL                          | COUNT  | DEPTH  | METRIC   | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|-------------------------------------------------------|--------|--------|----------|--------|--------|--------|--------|--------|--------|--------|
| >>> get_timer/ex_c_timing.c:20/[main (untimed)]       |      1 |      0 | page_rss | MB     |  0.012 |  0.012 |  0.012 |  0.012 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + tuple)] |      1 |      0 | page_rss | MB     |  0.152 |  0.152 |  0.152 |  0.152 |  0.000 |  100.0 |
| >>> get_timer/ex_c_timing.c:20/[main (timed + timer)] |      1 |      0 | page_rss | MB     |  0.098 |  0.098 |  0.098 |  0.098 |  0.000 |  100.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------------|

```
