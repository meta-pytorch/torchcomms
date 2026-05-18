# ex-caliper

This example demonstrates an example of instrumentation via caliper markers. An example timemory caliper marker looks as follows:

```c
TIMEMORY_BASIC_CALIPER(total, auto_tuple_t, "[total-", 
                       scope_tag, "-scope]");
```

where `auto_tuple_t = tim::auto_tuple_t<wall_clock, caliper, user_clock, system_clock, cpu_util>;`

## Build

See [examples](../README.md##Build) for generic examples build. This examples requires an additional `-DTIMEMORY_USE_CALIPER=ON` flag to be built.

## Expected Output

```bash
$ ./ex_caliper
#------------------------- tim::manager initialized [id=0][pid=30656] -------------------------#

caliper: scope = 'process', attributes = 268
[0] TESTING [execute_test [scope: process]]...
Initializing caliper...
fibonacci process : 1401939521834966
[cpu_util]|0> Outputting 'timemory-ex-caliper-output/cpu_util.json'...
[cpu_util]|0> Outputting 'timemory-ex-caliper-output/cpu_util.txt'...
Opening 'timemory-ex-caliper-output/cpu_util.jpeg' for output...
Closed 'timemory-ex-caliper-output/cpu_util.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------|
|                                       PERCENTAGE OF CPU-CLOCK TIME DIVIDED BY WALL-CLOCK TIME                                      |
|------------------------------------------------------------------------------------------------------------------------------------|
|                     LABEL                      | COUNT | DEPTH | METRIC   | UNITS | SUM   | MEAN | MIN   | MAX   | STDDEV | % SELF |
|------------------------------------------------|-------|-------|----------|-------|-------|------|-------|-------|--------|--------|
| >>> ex_caliper/[total-process-scope]           |    10 |     0 | cpu_util | %     | 162.9 | 16.3 | 134.7 | 167.7 |   10.0 |    0.0 |
| >>> |_ex_caliper/[worker-thread-process-scope] |    10 |     1 | cpu_util | %     |   0.0 |  0.0 |   0.0 |   0.0 |    0.0 |    0.0 |
| >>> |_ex_caliper/[master-thread-process-scope] |    10 |     1 | cpu_util | %     | 199.3 | 19.9 | 138.6 | 224.1 |   25.4 |    0.0 |
| >>>   |_time_fibonacci/[process-master]        |    10 |     2 | cpu_util | %     | 199.4 | 19.9 | 138.8 | 224.2 |   25.3 |   18.1 |
| >>>     |_time_fibonacci/[process-worker]      |    10 |     3 | cpu_util | %     | 163.2 | 16.3 | 136.1 | 167.9 |    9.2 |  100.0 |
|------------------------------------------------------------------------------------------------------------------------------------|

[user]|0> Outputting 'timemory-ex-caliper-output/user.flamegraph.json'...
[user]|0> Outputting 'timemory-ex-caliper-output/user.json'...
[user]|0> Outputting 'timemory-ex-caliper-output/user.txt'...
Opening 'timemory-ex-caliper-output/user.jpeg' for output...
Closed 'timemory-ex-caliper-output/user.jpeg'...

|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                            CPU TIME SPENT IN USER-MODE                                                                           |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                     LABEL                      |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|------------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> ex_caliper/[total-process-scope]           |         10 |          0 | user       | sec        |   3.170000 |   0.317000 |   0.030000 |   1.120000 |   0.358982 |       23.0 |
| >>> |_ex_caliper/[worker-thread-process-scope] |         10 |          1 | user       | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_ex_caliper/[master-thread-process-scope] |         10 |          1 | user       | sec        |   2.440000 |   0.244000 |   0.010000 |   0.850000 |   0.279134 |        0.0 |
| >>>   |_time_fibonacci/[process-master]        |         10 |          2 | user       | sec        |   2.440000 |   0.244000 |   0.010000 |   0.850000 |   0.279134 |        0.0 |
| >>>     |_time_fibonacci/[process-worker]      |         10 |          3 | user       | sec        |   3.170000 |   0.317000 |   0.030000 |   1.120000 |   0.358982 |      100.0 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[sys]|0> Outputting 'timemory-ex-caliper-output/sys.flamegraph.json'...
[sys]|0> Outputting 'timemory-ex-caliper-output/sys.json'...
[sys]|0> Outputting 'timemory-ex-caliper-output/sys.txt'...
Opening 'timemory-ex-caliper-output/sys.jpeg' for output...
Closed 'timemory-ex-caliper-output/sys.jpeg'...

|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                           CPU TIME SPENT IN KERNEL-MODE                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                     LABEL                      |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|------------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> ex_caliper/[total-process-scope]           |         10 |          0 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_ex_caliper/[worker-thread-process-scope] |         10 |          1 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_ex_caliper/[master-thread-process-scope] |         10 |          1 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>>   |_time_fibonacci/[process-master]        |         10 |          2 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>>     |_time_fibonacci/[process-worker]      |         10 |          3 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-caliper-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-caliper-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-caliper-output/wall.txt'...
Opening 'timemory-ex-caliper-output/wall.jpeg' for output...
Closed 'timemory-ex-caliper-output/wall.jpeg'...

|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                     REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                     |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                     LABEL                      |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|------------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> ex_caliper/[total-process-scope]           |         10 |          0 | wall       | sec        |   1.946205 |   0.194620 |   0.020136 |   0.690515 |   0.218792 |       37.1 |
| >>> |_ex_caliper/[worker-thread-process-scope] |         10 |          1 | wall       | sec        |   0.000280 |   0.000028 |   0.000018 |   0.000057 |   0.000013 |      100.0 |
| >>> |_ex_caliper/[master-thread-process-scope] |         10 |          1 | wall       | sec        |   1.224115 |   0.122411 |   0.006561 |   0.425341 |   0.139322 |        0.0 |
| >>>   |_time_fibonacci/[process-master]        |         10 |          2 | wall       | sec        |   1.223921 |   0.122392 |   0.006520 |   0.425325 |   0.139325 |        0.0 |
| >>>     |_time_fibonacci/[process-worker]      |         10 |          3 | wall       | sec        |   1.942111 |   0.194211 |   0.019285 |   0.690192 |   0.218876 |      100.0 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
```