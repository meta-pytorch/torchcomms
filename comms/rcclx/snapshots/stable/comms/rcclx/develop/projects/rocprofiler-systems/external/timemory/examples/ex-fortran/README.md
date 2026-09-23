# ex-fortran

This example demonstrates the use of timemory fortran API

## Build

See [examples](../README.md##Build). This examples requires an additional `-DTIMEMORY_BUILD_FORTRAN=ON` flag to be built.

## Expected Output

```console
$ ./ex_fortran_timing
[cpu]|0> Outputting 'timemory-ex-fortran-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-ex-fortran-output/cpu.json'...
[cpu]|0> Outputting 'timemory-ex-fortran-output/cpu.tree.json'...
[cpu]|0> Outputting 'timemory-ex-fortran-output/cpu.txt'...

|---------------------------------------------------------------------------------------------------------|
|                            TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                           |
|---------------------------------------------------------------------------------------------------------|
|    LABEL      | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|---------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> main      |      1 |      0 | cpu    | sec    |  0.010 |  0.010 |  0.010 |  0.010 |  0.000 |  100.0 |
| >>> |_inner   |      1 |      1 | cpu    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>> |_indexed |      1 |      1 | cpu    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
|---------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-fortran-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-fortran-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-fortran-output/wall.tree.json'...
[wall]|0> Outputting 'timemory-ex-fortran-output/wall.txt'...

|---------------------------------------------------------------------------------------------------------|
|                                 REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                |
|---------------------------------------------------------------------------------------------------------|
|    LABEL      | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|---------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> main      |      1 |      0 | wall   | sec    |  0.001 |  0.001 |  0.001 |  0.001 |  0.000 |   66.6 |
| >>> |_inner   |      1 |      1 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>> |_indexed |      1 |      1 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
|---------------------------------------------------------------------------------------------------------|

[peak_rss]|0> Outputting 'timemory-ex-fortran-output/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-ex-fortran-output/peak_rss.tree.json'...
[peak_rss]|0> Outputting 'timemory-ex-fortran-output/peak_rss.txt'...

|-------------------------------------------------------------------------------------------------------|
|   MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF |
|                                             SWAP IS ENABLED                                           |
|-------------------------------------------------------------------------------------------------------|
|  LABEL    | COUNT  | DEPTH  | METRIC   | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|-----------|--------|--------|----------|--------|--------|--------|--------|--------|--------|--------|
| >>> inner |      1 |      0 | peak_rss | MB     |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
|-------------------------------------------------------------------------------------------------------|

[page_rss]|0> Outputting 'timemory-ex-fortran-output/page_rss.json'...
[page_rss]|0> Outputting 'timemory-ex-fortran-output/page_rss.tree.json'...
[page_rss]|0> Outputting 'timemory-ex-fortran-output/page_rss.txt'...

|----------------------------------------------------------------------------------------------------------------|
|        AMOUNT OF MEMORY ALLOCATED IN PAGES OF MEMORY. UNLIKE PEAK_RSS, VALUE WILL FLUCTUATE AS MEMORY IS       |
|                                                 FREED/ALLOCATED                                                |
|----------------------------------------------------------------------------------------------------------------|
|       LABEL        | COUNT  | DEPTH  | METRIC   | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|--------------------|--------|--------|----------|--------|--------|--------|--------|--------|--------|--------|
| >>> inner (pushed) |      1 |      0 | page_rss | MB     |  0.004 |  0.004 |  0.004 |  0.004 |  0.000 |  100.0 |
|----------------------------------------------------------------------------------------------------------------|

```
