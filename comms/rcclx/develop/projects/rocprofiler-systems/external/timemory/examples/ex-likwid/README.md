# ex-likwid

This example demonstrates performance measurement using timemory's likwid markers.

## Build

See [examples](../README.md##Build). Further requires `-DTIMEMORY_USE_LIKWID=ON` to build.

## Expected Output

```bash
$ ./ex_likwid
#------------------------- tim::manager initialized [id=0][pid=10290] -------------------------#

[0] TESTING [execute_test [scope: perfmon-nvmon]]...
fibonacci perfmon-nvmon : 1407332192591406
[cpu_util]|0> Outputting 'timemory-ex-likwid-output/cpu_util.json'...
[cpu_util]|0> Outputting 'timemory-ex-likwid-output/cpu_util.txt'...
Opening 'timemory-ex-likwid-output/cpu_util.jpeg' for output...
Closed 'timemory-ex-likwid-output/cpu_util.jpeg'...

|-----------------------------------------------------------------------------------------------------------------------------------------|
|                                          PERCENTAGE OF CPU-CLOCK TIME DIVIDED BY WALL-CLOCK TIME                                        |
|-----------------------------------------------------------------------------------------------------------------------------------------|
|                       LABEL                         | COUNT | DEPTH | METRIC   | UNITS | SUM   | MEAN | MIN   | MAX   | STDDEV | % SELF |
|-----------------------------------------------------|-------|-------|----------|-------|-------|------|-------|-------|--------|--------|
| >>> ex_likwid/[total-perfmon-nvmon-scope]           |    10 |     0 | cpu_util | %     | 162.6 | 16.3 | 146.9 | 166.6 |    7.4 |    0.0 |
| >>> |_ex_likwid/[worker-thread-perfmon-nvmon-scope] |    10 |     1 | cpu_util | %     |   0.0 |  0.0 |   0.0 |   0.0 |    0.0 |    0.0 |
| >>> |_ex_likwid/[master-thread-perfmon-nvmon-scope] |    10 |     1 | cpu_util | %     | 197.7 | 19.8 | 155.8 | 204.1 |   16.4 |    0.0 |
| >>>   |_time_fibonacci/[perfmon-nvmon-master]       |    10 |     2 | cpu_util | %     | 197.7 | 19.8 | 156.1 | 204.3 |   16.4 |   17.6 |
| >>>     |_time_fibonacci/[perfmon-nvmon-worker]     |    10 |     3 | cpu_util | %     | 162.9 | 16.3 | 147.8 | 167.2 |    6.6 |  100.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------|

[user]|0> Outputting 'timemory-ex-likwid-output/user.flamegraph.json'...
[user]|0> Outputting 'timemory-ex-likwid-output/user.json'...
[user]|0> Outputting 'timemory-ex-likwid-output/user.txt'...
Opening 'timemory-ex-likwid-output/user.jpeg' for output...
Closed 'timemory-ex-likwid-output/user.jpeg'...

|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                               CPU TIME SPENT IN USER-MODE                                                                             |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                       LABEL                         |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|-----------------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> ex_likwid/[total-perfmon-nvmon-scope]           |         10 |          0 | user       | sec        |   3.130000 |   0.313000 |   0.030000 |   1.100000 |   0.352737 |       24.0 |
| >>> |_ex_likwid/[worker-thread-perfmon-nvmon-scope] |         10 |          1 | user       | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_ex_likwid/[master-thread-perfmon-nvmon-scope] |         10 |          1 | user       | sec        |   2.380000 |   0.238000 |   0.010000 |   0.840000 |   0.276035 |        0.0 |
| >>>   |_time_fibonacci/[perfmon-nvmon-master]       |         10 |          2 | user       | sec        |   2.380000 |   0.238000 |   0.010000 |   0.840000 |   0.276035 |        0.0 |
| >>>     |_time_fibonacci/[perfmon-nvmon-worker]     |         10 |          3 | user       | sec        |   3.130000 |   0.313000 |   0.030000 |   1.100000 |   0.352737 |      100.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[sys]|0> Outputting 'timemory-ex-likwid-output/sys.flamegraph.json'...
[sys]|0> Outputting 'timemory-ex-likwid-output/sys.json'...
[sys]|0> Outputting 'timemory-ex-likwid-output/sys.txt'...
Opening 'timemory-ex-likwid-output/sys.jpeg' for output...
Closed 'timemory-ex-likwid-output/sys.jpeg'...

|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                              CPU TIME SPENT IN KERNEL-MODE                                                                            |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                       LABEL                         |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|-----------------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> ex_likwid/[total-perfmon-nvmon-scope]           |         10 |          0 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_ex_likwid/[worker-thread-perfmon-nvmon-scope] |         10 |          1 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_ex_likwid/[master-thread-perfmon-nvmon-scope] |         10 |          1 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>>   |_time_fibonacci/[perfmon-nvmon-master]       |         10 |          2 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>>     |_time_fibonacci/[perfmon-nvmon-worker]     |         10 |          3 | sys        | sec        |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-likwid-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-likwid-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-likwid-output/wall.txt'...
Opening 'timemory-ex-likwid-output/wall.jpeg' for output...
Closed 'timemory-ex-likwid-output/wall.jpeg'...

|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                        REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                       |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                       LABEL                         |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|-----------------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> ex_likwid/[total-perfmon-nvmon-scope]           |         10 |          0 | wall       | sec        |   1.925248 |   0.192525 |   0.020349 |   0.678071 |   0.215506 |       37.5 |
| >>> |_ex_likwid/[worker-thread-perfmon-nvmon-scope] |         10 |          1 | wall       | sec        |   0.000285 |   0.000029 |   0.000018 |   0.000057 |   0.000013 |      100.0 |
| >>> |_ex_likwid/[master-thread-perfmon-nvmon-scope] |         10 |          1 | wall       | sec        |   1.203867 |   0.120387 |   0.006417 |   0.417397 |   0.136910 |        0.0 |
| >>>   |_time_fibonacci/[perfmon-nvmon-master]       |         10 |          2 | wall       | sec        |   1.203766 |   0.120377 |   0.006404 |   0.417389 |   0.136911 |        0.0 |
| >>>     |_time_fibonacci/[perfmon-nvmon-worker]     |         10 |          3 | wall       | sec        |   1.921988 |   0.192199 |   0.019745 |   0.677860 |   0.215565 |      100.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
```