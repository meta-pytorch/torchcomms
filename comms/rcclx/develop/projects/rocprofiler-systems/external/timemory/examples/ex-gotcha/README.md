# ex-gotcha

These examples demonstrate the use of GOTCHA wrappers by wrapping `puts` and `MPI` routines and then instrumenting them using timemory. The ex-gotcha-replacement demonstrates an example of replacing the STDLIB's `exp` function with a gotcha wrapped `expf` function.

## Build

See [examples](../README.md##Build). Further, these examples require timemory MPI and GOTCHA support to be enabled by enabling `-DTIMEMORY_USE_GOTCHA=ON` and `-DTIMEMORY_USE_MPI=ON` flags in timemory cmake build.

## Expected Output

```bash
$ ./ex_gotcha
#------------------------- tim::manager initialized [id=0][pid=9347] -------------------------#

put gotcha is available: true
mpi gotcha is available: true

size = 1
rank = 0
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   511.47
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   510.71
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   499.13
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   511.18
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   501.40
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   487.90
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   507.49
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   511.22
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   512.39
Testing puts gotcha wraper...

[iterations=15]>      single-precision exp = 1915398520832.000000
[iterations=15]>      double-precision exp = 1915398474579.756592

[0]> sum =   511.55
[peak_rss]|0> Outputting 'timemory-ex-gotcha-output/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-ex-gotcha-output/peak_rss.txt'...
Opening 'timemory-ex-gotcha-output/peak_rss.jpeg' for output...
Closed 'timemory-ex-gotcha-output/peak_rss.jpeg'...

|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                           MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF SWAP IS ENABLED                         |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|               LABEL                 |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|-------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> ./ex_gotcha                     |         10 |          0 | peak_rss   | KB         | 356.000000 |  35.600000 |   0.000000 | 356.000000 | 112.577085 |      100.0 |
| >>> |_Testing puts gotcha wraper... |         10 |          1 | peak_rss   | KB         |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_MPI_Barrier                   |         20 |          1 | peak_rss   | KB         |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_MPI_Allreduce                 |         10 |          1 | peak_rss   | KB         |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[cpu]|0> Outputting 'timemory-ex-gotcha-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-ex-gotcha-output/cpu.json'...
[cpu]|0> Outputting 'timemory-ex-gotcha-output/cpu.txt'...
Opening 'timemory-ex-gotcha-output/cpu.jpeg' for output...
Closed 'timemory-ex-gotcha-output/cpu.jpeg'...

|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                           TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                                                          |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|               LABEL                 |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|-------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> ./ex_gotcha                     |         10 |          0 | cpu        | msec       |  10.000000 |   1.000000 |   0.000000 |  10.000000 |   3.162278 |      100.0 |
| >>> |_Testing puts gotcha wraper... |         10 |          1 | cpu        | msec       |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_MPI_Barrier                   |         20 |          1 | cpu        | msec       |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
| >>> |_MPI_Allreduce                 |         10 |          1 | cpu        | msec       |   0.000000 |   0.000000 |   0.000000 |   0.000000 |   0.000000 |        0.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-gotcha-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-gotcha-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-gotcha-output/wall.txt'...
Opening 'timemory-ex-gotcha-output/wall.jpeg' for output...
Closed 'timemory-ex-gotcha-output/wall.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|               LABEL                 |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM      |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|-------------------------------------|------------|------------|------------|------------|-------------|------------|------------|------------|------------|------------|
| >>> ./ex_gotcha                     |         10 |          0 | wall       | msec       | 5006.735103 | 500.673510 | 500.343106 | 503.562931 |   1.015317 |      100.0 |
| >>> |_Testing puts gotcha wraper... |         10 |          1 | wall       | msec       |    0.150961 |   0.015096 |   0.014567 |   0.016830 |   0.000715 |      100.0 |
| >>> |_MPI_Barrier                   |         20 |          1 | wall       | msec       |    0.097922 |   0.004896 |   0.002705 |   0.008961 |   0.001192 |      100.0 |
| >>> |_MPI_Allreduce                 |         10 |          1 | wall       | msec       |    0.179468 |   0.017947 |   0.014507 |   0.042923 |   0.008874 |      100.0 |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|


[metadata::manager::finalize]> Outputting 'timemory-ex-gotcha-output/metadata.json'...


#---------------------- tim::manager destroyed [rank=0][id=0][pid=9347] ----------------------#
```