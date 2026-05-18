# ex-statistics

This example demonstrates an example of generating measurement statistics for `monotonic_clock` and `wall_clock` components by setting appropriate component traits. Four components i.e. `monotonic_clock, wall_clock, cpu_clock, and current_peak_rss` are used for measurements where the monotonic_clock is set to `flat_storage` and `wall_clock` to generate statistics by setting appropriate traits as shown in following code snippet:

```c
// component tuple for measurements
using tuple_t = auto_tuple_t<wall_clock, monotonic_clock, cpu_clock, current_peak_rss>;

// enable statistics recording
TIMEMORY_DEFINE_CONCRETE_TRAIT(flat_storage, component::monotonic_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::wall_clock, true_type)
// disable statistics recording 
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::cpu_clock, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(record_statistics, component::current_peak_rss, true_type)
```

## Build

See [examples](../README.md##Build).

## Expected Output

```bash
$ ./ex_cxx_statistics
#------------------------- tim::manager initialized [id=0][pid=13711] -------------------------#

>>>  inst_fib :    0.100 sec wall,    0.100 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    0.200 sec wall,    0.200 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    0.301 sec wall,    0.301 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    0.401 sec wall,    0.401 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    0.501 sec wall,    0.501 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    0.601 sec wall,    0.601 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    0.701 sec wall,    0.701 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    0.802 sec wall,    0.802 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    0.902 sec wall,    0.902 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.002 sec wall,    1.002 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.102 sec wall,    1.102 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.203 sec wall,    1.203 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.303 sec wall,    1.303 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.403 sec wall,    1.403 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.503 sec wall,    1.503 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.603 sec wall,    1.603 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.704 sec wall,    1.704 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.804 sec wall,    1.804 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    1.904 sec wall,    1.904 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib :    2.004 sec wall,    2.004 sec monotonic_clock,    0.000 sec cpu,    8.160 MB,    8.424 MB current_peak_rss [laps: 1]
Answer = 51172666797121
>>>  nested/0 :    2.438 sec wall,    2.438 sec monotonic_clock,    0.440 sec cpu,    5.976 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  main@ex_statistics.cpp:103/total :    2.715 sec wall,    2.715 sec monotonic_clock,    0.720 sec cpu,    5.976 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    0.100 sec wall,    0.100 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    0.200 sec wall,    0.200 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    0.301 sec wall,    0.301 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    0.401 sec wall,    0.401 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    0.501 sec wall,    0.501 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    0.601 sec wall,    0.601 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    0.701 sec wall,    0.701 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    0.802 sec wall,    0.802 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    0.902 sec wall,    0.902 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    1.002 sec wall,    1.002 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    1.102 sec wall,    1.102 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    1.203 sec wall,    1.203 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  inst_fib                         :    1.303 sec wall,    1.303 sec monotonic_clock,    0.000 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
Answer = 158963259977675
>>>  nested/1                         :    1.883 sec wall,    1.883 sec monotonic_clock,    0.580 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
>>>  main@ex_statistics.cpp:103/total :    2.138 sec wall,    2.138 sec monotonic_clock,    0.830 sec cpu,    8.424 MB,    8.424 MB current_peak_rss [laps: 1]
Answer = 279487289961752
Answer = 111276995389891
Answer = 26821767161667
Answer = 28033794962287
Answer = 102505730364791
Answer = 253657299984579
Answer = 115219529955313
Answer = 55479297376141
[current_peak_rss]|0> Outputting 'timemory-ex-cxx-statistics-output/current_peak_rss.json'...
[current_peak_rss]|0> Outputting 'timemory-ex-cxx-statistics-output/current_peak_rss.txt'...
Opening 'timemory-ex-cxx-statistics-output/current_peak_rss.jpeg' for output...
Closed 'timemory-ex-cxx-statistics-output/current_peak_rss.jpeg'...

|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                ABSOLUTE VALUE OF HIGH-WATER MARK OF MEMORY ALLOCATION IN RAM                                              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
|                              LABEL                               | COUNT  | DEPTH  |     METRIC     | UNITS  | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>> main@ex_statistics.cpp:103/total                             |     10 |      0 | start peak rss | MB     | 11.234 |  5.976 | 14.828 |  3.023 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.119 |  8.424 | 14.828 |  2.576 |    0.0 |
| >>> |_nested/0                                                   |      2 |      1 | start peak rss | MB     |  8.686 |  5.976 | 11.396 |  3.833 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>   |_inst_fib                                                 |      2 |      2 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>       |_inst_fib                                             |      2 |      4 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>           |_inst_fib                                         |      2 |      6 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>             |_inst_fib                                       |      2 |      7 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>                   |_inst_fib                                 |      2 |     10 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | start peak rss | MB     |  9.778 |  8.160 | 11.396 |  2.288 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>                       |_inst_fib                             |      2 |     12 | start peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>                         |_inst_fib                           |      2 |     13 | start peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>                           |_inst_fib                         |      2 |     14 | start peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
| >>>                             |_inst_fib                       |      2 |     15 | start peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |   57.5 |
|                                                                  |        |        |  stop peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |   57.5 |
| >>>                               |_inst_fib                     |      1 |     16 | start peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
| >>>                                 |_inst_fib                   |      1 |     17 | start peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | start peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
| >>>                                     |_inst_fib               |      1 |     19 | start peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>                                       |_inst_fib             |      1 |     20 | start peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |    0.0 |
| >>>                                         |_inst_fib           |      1 |     21 | start peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |  100.0 |
|                                                                  |        |        |  stop peak rss | MB     |  8.424 |  8.424 |  8.424 |  0.000 |  100.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | start peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
| >>>     |_inst_fib/6                                             |      1 |      3 | start peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
| >>>       |_inst_fib/5                                           |      1 |      4 | start peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
| >>>         |_inst_fib/4                                         |      1 |      5 | start peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
| >>>           |_inst_fib/3                                       |      1 |      6 | start peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
| >>>             |_inst_fib/2                                     |      1 |      7 | start peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |    0.0 |
| >>>               |_inst_fib/1                                   |      1 |      8 | start peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |  100.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.396 | 11.396 | 11.396 |  0.000 |  100.0 |
| >>> |_nested/1                                                   |      2 |      1 | start peak rss | MB     |  9.910 |  8.424 | 11.396 |  2.102 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>   |_inst_fib                                                 |      2 |      2 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>       |_inst_fib                                             |      2 |      4 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>           |_inst_fib                                         |      2 |      6 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>             |_inst_fib                                       |      2 |      7 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>                       |_inst_fib                             |      2 |     12 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>                         |_inst_fib                           |      2 |     13 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
| >>>                           |_inst_fib                         |      2 |     14 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |   36.2 |
|                                                                  |        |        |  stop peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |   36.2 |
| >>>                             |_inst_fib                       |      1 |     15 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                               |_inst_fib                     |      1 |     16 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                 |_inst_fib                   |      1 |     17 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                     |_inst_fib               |      1 |     19 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                       |_inst_fib             |      1 |     20 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                         |_inst_fib           |      1 |     21 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>                                           |_inst_fib         |      1 |     22 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                             |_inst_fib       |      1 |     23 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                               |_inst_fib     |      1 |     24 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                                 |_inst_fib   |      1 |     25 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                                   |_inst_fib |      1 |     26 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |  100.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |  100.0 |
| >>> |_nested/2                                                   |      2 |      1 | start peak rss | MB     | 11.626 |  8.424 | 14.828 |  4.528 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>   |_inst_fib                                                 |      2 |      2 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>       |_inst_fib                                             |      2 |      4 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>           |_inst_fib                                         |      2 |      6 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>             |_inst_fib                                       |      2 |      7 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                       |_inst_fib                             |      2 |     12 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |   42.9 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |   42.9 |
| >>>                         |_inst_fib                           |      1 |     13 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                           |_inst_fib                         |      1 |     14 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |  100.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |  100.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>     |_inst_fib/7                                             |      1 |      3 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>       |_inst_fib/6                                           |      1 |      4 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>         |_inst_fib/5                                         |      1 |      5 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>           |_inst_fib/4                                       |      1 |      6 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>             |_inst_fib/3                                     |      1 |      7 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>               |_inst_fib/2                                   |      1 |      8 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>                 |_inst_fib/1                                 |      1 |      9 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |  100.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |  100.0 |
| >>> |_nested/3                                                   |      2 |      1 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>   |_inst_fib                                                 |      2 |      2 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>       |_inst_fib                                             |      2 |      4 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>           |_inst_fib                                         |      2 |      6 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>             |_inst_fib                                       |      2 |      7 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                       |_inst_fib                             |      2 |     12 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                         |_inst_fib                           |      2 |     13 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>                           |_inst_fib                         |      2 |     14 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
| >>>                             |_inst_fib                       |      2 |     15 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |   57.1 |
|                                                                  |        |        |  stop peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |   57.1 |
| >>>                               |_inst_fib                     |      1 |     16 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>                                 |_inst_fib                   |      1 |     17 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>                                     |_inst_fib               |      1 |     19 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>                                       |_inst_fib             |      1 |     20 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |    0.0 |
| >>>                                         |_inst_fib           |      1 |     21 | start peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |  100.0 |
|                                                                  |        |        |  stop peak rss | MB     | 11.120 | 11.120 | 11.120 |  0.000 |  100.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>     |_inst_fib/5                                             |      1 |      3 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>       |_inst_fib/4                                           |      1 |      4 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>         |_inst_fib/3                                         |      1 |      5 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>           |_inst_fib/2                                       |      1 |      6 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>             |_inst_fib/1                                     |      1 |      7 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |  100.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |  100.0 |
| >>> |_nested/4                                                   |      2 |      1 | start peak rss | MB     | 12.974 | 11.120 | 14.828 |  2.622 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>   |_inst_fib                                                 |      2 |      2 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>       |_inst_fib                                             |      2 |      4 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>           |_inst_fib                                         |      2 |      6 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>             |_inst_fib                                       |      2 |      7 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>                       |_inst_fib                             |      2 |     12 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>                         |_inst_fib                           |      2 |     13 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |    0.0 |
| >>>                           |_inst_fib                         |      2 |     14 | start peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |   43.5 |
|                                                                  |        |        |  stop peak rss | MB     | 13.112 | 11.396 | 14.828 |  2.427 |   43.5 |
| >>>                             |_inst_fib                       |      1 |     15 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                               |_inst_fib                     |      1 |     16 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|
| >>>                                 |_inst_fib                   |      1 |     17 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                     |_inst_fib               |      1 |     19 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                       |_inst_fib             |      1 |     20 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |    0.0 |
| >>>                                         |_inst_fib           |      1 |     21 | start peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |  100.0 |
|                                                                  |        |        |  stop peak rss | MB     | 14.828 | 14.828 | 14.828 |  0.000 |  100.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|

[cpu]|0> Outputting 'timemory-ex-cxx-statistics-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-ex-cxx-statistics-output/cpu.json'...
[cpu]|0> Outputting 'timemory-ex-cxx-statistics-output/cpu.txt'...
Opening 'timemory-ex-cxx-statistics-output/cpu.jpeg' for output...
Closed 'timemory-ex-cxx-statistics-output/cpu.jpeg'...

|---------------------------------------------------------------------------------------------------------------------------------|
|                                        TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                                       |
|---------------------------------------------------------------------------------------------------------------------------------|
|                              LABEL                               | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   | % SELF |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>> main@ex_statistics.cpp:103/total                             |     10 |      0 | cpu    | sec    |  8.620 |  0.862 |   30.2 |
| >>> |_nested/0                                                   |      2 |      1 | cpu    | sec    |  1.280 |  0.640 |   86.7 |
| >>>   |_inst_fib                                                 |      2 |      2 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>       |_inst_fib                                             |      2 |      4 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>           |_inst_fib                                         |      2 |      6 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>             |_inst_fib                                       |      2 |      7 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                   |_inst_fib                                 |      2 |     10 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                       |_inst_fib                             |      2 |     12 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                         |_inst_fib                           |      2 |     13 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                           |_inst_fib                         |      2 |     14 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                             |_inst_fib                       |      2 |     15 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                               |_inst_fib                     |      1 |     16 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                 |_inst_fib                   |      1 |     17 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                     |_inst_fib               |      1 |     19 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                                       |_inst_fib             |      1 |     20 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                         |_inst_fib           |      1 |     21 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | cpu    | sec    |  0.170 |  0.170 |  100.0 |
| >>>     |_inst_fib/6                                             |      1 |      3 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>       |_inst_fib/5                                           |      1 |      4 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>         |_inst_fib/4                                         |      1 |      5 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>           |_inst_fib/3                                       |      1 |      6 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>             |_inst_fib/2                                     |      1 |      7 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>               |_inst_fib/1                                   |      1 |      8 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>> |_nested/1                                                   |      2 |      1 | cpu    | sec    |  1.020 |  0.510 |  100.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>   |_inst_fib                                                 |      2 |      2 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>       |_inst_fib                                             |      2 |      4 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>           |_inst_fib                                         |      2 |      6 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>             |_inst_fib                                       |      2 |      7 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                       |_inst_fib                             |      2 |     12 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                         |_inst_fib                           |      2 |     13 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                           |_inst_fib                         |      2 |     14 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                             |_inst_fib                       |      1 |     15 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                               |_inst_fib                     |      1 |     16 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                 |_inst_fib                   |      1 |     17 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                     |_inst_fib               |      1 |     19 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                       |_inst_fib             |      1 |     20 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                         |_inst_fib           |      1 |     21 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                                           |_inst_fib         |      1 |     22 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                             |_inst_fib       |      1 |     23 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                               |_inst_fib     |      1 |     24 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                                 |_inst_fib   |      1 |     25 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                                   |_inst_fib |      1 |     26 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>> |_nested/2                                                   |      2 |      1 | cpu    | sec    |  1.420 |  0.710 |   88.7 |
| >>>   |_inst_fib                                                 |      2 |      2 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>       |_inst_fib                                             |      2 |      4 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>           |_inst_fib                                         |      2 |      6 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>             |_inst_fib                                       |      2 |      7 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                       |_inst_fib                             |      2 |     12 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                         |_inst_fib                           |      1 |     13 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                           |_inst_fib                         |      1 |     14 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | cpu    | sec    |  0.160 |  0.160 |  100.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>     |_inst_fib/7                                             |      1 |      3 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>       |_inst_fib/6                                           |      1 |      4 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>         |_inst_fib/5                                         |      1 |      5 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>           |_inst_fib/4                                       |      1 |      6 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>             |_inst_fib/3                                     |      1 |      7 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>               |_inst_fib/2                                   |      1 |      8 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                 |_inst_fib/1                                 |      1 |      9 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>> |_nested/3                                                   |      2 |      1 | cpu    | sec    |  1.280 |  0.640 |   86.7 |
| >>>   |_inst_fib                                                 |      2 |      2 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>       |_inst_fib                                             |      2 |      4 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>           |_inst_fib                                         |      2 |      6 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>             |_inst_fib                                       |      2 |      7 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                       |_inst_fib                             |      2 |     12 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                         |_inst_fib                           |      2 |     13 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                           |_inst_fib                         |      2 |     14 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                             |_inst_fib                       |      2 |     15 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                               |_inst_fib                     |      1 |     16 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                 |_inst_fib                   |      1 |     17 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                     |_inst_fib               |      1 |     19 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                       |_inst_fib             |      1 |     20 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                         |_inst_fib           |      1 |     21 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | cpu    | sec    |  0.170 |  0.170 |  100.0 |
| >>>     |_inst_fib/5                                             |      1 |      3 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>       |_inst_fib/4                                           |      1 |      4 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>         |_inst_fib/3                                         |      1 |      5 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>           |_inst_fib/2                                       |      1 |      6 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>             |_inst_fib/1                                     |      1 |      7 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>> |_nested/4                                                   |      2 |      1 | cpu    | sec    |  1.020 |  0.510 |  100.0 |
| >>>   |_inst_fib                                                 |      2 |      2 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>     |_inst_fib                                               |      2 |      3 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>       |_inst_fib                                             |      2 |      4 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>         |_inst_fib                                           |      2 |      5 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>           |_inst_fib                                         |      2 |      6 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>             |_inst_fib                                       |      2 |      7 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>               |_inst_fib                                     |      2 |      8 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                 |_inst_fib                                   |      2 |      9 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                     |_inst_fib                               |      2 |     11 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                       |_inst_fib                             |      2 |     12 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                         |_inst_fib                           |      2 |     13 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                           |_inst_fib                         |      2 |     14 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                             |_inst_fib                       |      1 |     15 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                               |_inst_fib                     |      1 |     16 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                                 |_inst_fib                   |      1 |     17 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                     |_inst_fib               |      1 |     19 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                       |_inst_fib             |      1 |     20 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
| >>>                                         |_inst_fib           |      1 |     21 | cpu    | sec    |  0.000 |  0.000 |    0.0 |
|---------------------------------------------------------------------------------------------------------------------------------|

[monotonic_clock]|0> Outputting 'timemory-ex-cxx-statistics-output/monotonic_clock.flamegraph.json'...
[monotonic_clock]|0> Outputting 'timemory-ex-cxx-statistics-output/monotonic_clock.json'...
[monotonic_clock]|0> Outputting 'timemory-ex-cxx-statistics-output/monotonic_clock.txt'...
Opening 'timemory-ex-cxx-statistics-output/monotonic_clock.jpeg' for output...
Closed 'timemory-ex-cxx-statistics-output/monotonic_clock.jpeg'...

|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                           WALL-CLOCK TIMER WHICH WILL CONTINUE TO INCREMENT EVEN WHILE THE SYSTEM IS ASLEEP                                         |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                              LABEL                               | COUNT  | DEPTH  |     METRIC      | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>> main@ex_statistics.cpp:103/total                             |     10 |      0 | monotonic_clock | sec    | 26.733 |  2.673 |  2.136 |  3.206 |  0.405 |    9.7 |
| >>> |_nested/0                                                   |      2 |      1 | monotonic_clock | sec    |  5.276 |  2.638 |  2.438 |  2.838 |  0.283 |   20.9 |
| >>>   |_inst_fib                                                 |      2 |      2 | monotonic_clock | sec    |  3.406 |  1.703 |  1.402 |  2.004 |  0.426 |    5.9 |
| >>>     |_inst_fib                                               |      2 |      3 | monotonic_clock | sec    |  3.206 |  1.603 |  1.302 |  1.904 |  0.426 |    6.3 |
| >>>       |_inst_fib                                             |      2 |      4 | monotonic_clock | sec    |  3.006 |  1.503 |  1.202 |  1.804 |  0.426 |    6.7 |
| >>>         |_inst_fib                                           |      2 |      5 | monotonic_clock | sec    |  2.805 |  1.403 |  1.101 |  1.704 |  0.426 |    7.1 |
| >>>           |_inst_fib                                         |      2 |      6 | monotonic_clock | sec    |  2.605 |  1.302 |  1.001 |  1.603 |  0.426 |    7.7 |
| >>>             |_inst_fib                                       |      2 |      7 | monotonic_clock | sec    |  2.404 |  1.202 |  0.901 |  1.503 |  0.426 |    8.3 |
| >>>               |_inst_fib                                     |      2 |      8 | monotonic_clock | sec    |  2.204 |  1.102 |  0.801 |  1.403 |  0.426 |    9.1 |
| >>>                 |_inst_fib                                   |      2 |      9 | monotonic_clock | sec    |  2.004 |  1.002 |  0.701 |  1.303 |  0.426 |   10.0 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                   |_inst_fib                                 |      2 |     10 | monotonic_clock | sec    |  1.803 |  0.902 |  0.601 |  1.203 |  0.426 |   11.1 |
| >>>                     |_inst_fib                               |      2 |     11 | monotonic_clock | sec    |  1.603 |  0.802 |  0.501 |  1.102 |  0.425 |   12.5 |
| >>>                       |_inst_fib                             |      2 |     12 | monotonic_clock | sec    |  1.403 |  0.701 |  0.401 |  1.002 |  0.425 |   14.3 |
| >>>                         |_inst_fib                           |      2 |     13 | monotonic_clock | sec    |  1.202 |  0.601 |  0.300 |  0.902 |  0.425 |   16.7 |
| >>>                           |_inst_fib                         |      2 |     14 | monotonic_clock | sec    |  1.002 |  0.501 |  0.200 |  0.802 |  0.425 |   20.0 |
| >>>                             |_inst_fib                       |      2 |     15 | monotonic_clock | sec    |  0.802 |  0.401 |  0.100 |  0.701 |  0.425 |   25.0 |
| >>>                               |_inst_fib                     |      1 |     16 | monotonic_clock | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
| >>>                                 |_inst_fib                   |      1 |     17 | monotonic_clock | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | monotonic_clock | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>                                     |_inst_fib               |      1 |     19 | monotonic_clock | sec    |  0.301 |  0.301 |  0.301 |  0.301 |  0.000 |   33.3 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                                       |_inst_fib             |      1 |     20 | monotonic_clock | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                                         |_inst_fib           |      1 |     21 | monotonic_clock | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | monotonic_clock | sec    |  0.766 |  0.766 |  0.766 |  0.766 |  0.000 |   21.6 |
| >>>     |_inst_fib/6                                             |      1 |      3 | monotonic_clock | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
| >>>       |_inst_fib/5                                           |      1 |      4 | monotonic_clock | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>         |_inst_fib/4                                         |      1 |      5 | monotonic_clock | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>           |_inst_fib/3                                       |      1 |      6 | monotonic_clock | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>             |_inst_fib/2                                     |      1 |      7 | monotonic_clock | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>               |_inst_fib/1                                   |      1 |      8 | monotonic_clock | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>> |_nested/1                                                   |      2 |      1 | monotonic_clock | sec    |  4.825 |  2.412 |  1.883 |  2.942 |  0.749 |   21.1 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>   |_inst_fib                                                 |      2 |      2 | monotonic_clock | sec    |  3.807 |  1.903 |  1.303 |  2.504 |  0.849 |    5.3 |
| >>>     |_inst_fib                                               |      2 |      3 | monotonic_clock | sec    |  3.606 |  1.803 |  1.203 |  2.404 |  0.849 |    5.6 |
| >>>       |_inst_fib                                             |      2 |      4 | monotonic_clock | sec    |  3.406 |  1.703 |  1.102 |  2.303 |  0.849 |    5.9 |
| >>>         |_inst_fib                                           |      2 |      5 | monotonic_clock | sec    |  3.205 |  1.603 |  1.002 |  2.203 |  0.849 |    6.3 |
| >>>           |_inst_fib                                         |      2 |      6 | monotonic_clock | sec    |  3.005 |  1.502 |  0.902 |  2.103 |  0.849 |    6.7 |
| >>>             |_inst_fib                                       |      2 |      7 | monotonic_clock | sec    |  2.804 |  1.402 |  0.802 |  2.003 |  0.849 |    7.1 |
| >>>               |_inst_fib                                     |      2 |      8 | monotonic_clock | sec    |  2.604 |  1.302 |  0.701 |  1.903 |  0.849 |    7.7 |
| >>>                 |_inst_fib                                   |      2 |      9 | monotonic_clock | sec    |  2.404 |  1.202 |  0.601 |  1.802 |  0.849 |    8.3 |
| >>>                   |_inst_fib                                 |      2 |     10 | monotonic_clock | sec    |  2.203 |  1.102 |  0.501 |  1.702 |  0.849 |    9.1 |
| >>>                     |_inst_fib                               |      2 |     11 | monotonic_clock | sec    |  2.003 |  1.001 |  0.401 |  1.602 |  0.850 |   10.0 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                       |_inst_fib                             |      2 |     12 | monotonic_clock | sec    |  1.803 |  0.901 |  0.301 |  1.502 |  0.850 |   11.1 |
| >>>                         |_inst_fib                           |      2 |     13 | monotonic_clock | sec    |  1.602 |  0.801 |  0.200 |  1.402 |  0.850 |   12.5 |
| >>>                           |_inst_fib                         |      2 |     14 | monotonic_clock | sec    |  1.402 |  0.701 |  0.100 |  1.302 |  0.850 |   14.3 |
| >>>                             |_inst_fib                       |      1 |     15 | monotonic_clock | sec    |  1.202 |  1.202 |  1.202 |  1.202 |  0.000 |    8.3 |
| >>>                               |_inst_fib                     |      1 |     16 | monotonic_clock | sec    |  1.102 |  1.102 |  1.102 |  1.102 |  0.000 |    9.1 |
| >>>                                 |_inst_fib                   |      1 |     17 | monotonic_clock | sec    |  1.001 |  1.001 |  1.001 |  1.001 |  0.000 |   10.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | monotonic_clock | sec    |  0.901 |  0.901 |  0.901 |  0.901 |  0.000 |   11.1 |
| >>>                                     |_inst_fib               |      1 |     19 | monotonic_clock | sec    |  0.801 |  0.801 |  0.801 |  0.801 |  0.000 |   12.5 |
| >>>                                       |_inst_fib             |      1 |     20 | monotonic_clock | sec    |  0.701 |  0.701 |  0.701 |  0.701 |  0.000 |   14.3 |
| >>>                                         |_inst_fib           |      1 |     21 | monotonic_clock | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                                           |_inst_fib         |      1 |     22 | monotonic_clock | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>                                             |_inst_fib       |      1 |     23 | monotonic_clock | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>                                               |_inst_fib     |      1 |     24 | monotonic_clock | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>                                                 |_inst_fib   |      1 |     25 | monotonic_clock | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                                                   |_inst_fib |      1 |     26 | monotonic_clock | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>> |_nested/2                                                   |      2 |      1 | monotonic_clock | sec    |  4.521 |  2.261 |  1.881 |  2.640 |  0.537 |   27.6 |
| >>>   |_inst_fib                                                 |      2 |      2 | monotonic_clock | sec    |  2.403 |  1.202 |  1.101 |  1.302 |  0.142 |    8.3 |
| >>>     |_inst_fib                                               |      2 |      3 | monotonic_clock | sec    |  2.203 |  1.102 |  1.001 |  1.202 |  0.142 |    9.1 |
| >>>       |_inst_fib                                             |      2 |      4 | monotonic_clock | sec    |  2.003 |  1.001 |  0.901 |  1.102 |  0.142 |   10.0 |
| >>>         |_inst_fib                                           |      2 |      5 | monotonic_clock | sec    |  1.802 |  0.901 |  0.801 |  1.001 |  0.142 |   11.1 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>           |_inst_fib                                         |      2 |      6 | monotonic_clock | sec    |  1.602 |  0.801 |  0.701 |  0.901 |  0.142 |   12.5 |
| >>>             |_inst_fib                                       |      2 |      7 | monotonic_clock | sec    |  1.402 |  0.701 |  0.601 |  0.801 |  0.142 |   14.3 |
| >>>               |_inst_fib                                     |      2 |      8 | monotonic_clock | sec    |  1.202 |  0.601 |  0.501 |  0.701 |  0.142 |   16.7 |
| >>>                 |_inst_fib                                   |      2 |      9 | monotonic_clock | sec    |  1.001 |  0.501 |  0.401 |  0.601 |  0.142 |   20.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | monotonic_clock | sec    |  0.801 |  0.401 |  0.300 |  0.501 |  0.142 |   25.0 |
| >>>                     |_inst_fib                               |      2 |     11 | monotonic_clock | sec    |  0.601 |  0.300 |  0.200 |  0.401 |  0.142 |   33.3 |
| >>>                       |_inst_fib                             |      2 |     12 | monotonic_clock | sec    |  0.401 |  0.200 |  0.100 |  0.300 |  0.142 |   50.0 |
| >>>                         |_inst_fib                           |      1 |     13 | monotonic_clock | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                           |_inst_fib                         |      1 |     14 | monotonic_clock | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | monotonic_clock | sec    |  0.869 |  0.869 |  0.869 |  0.869 |  0.000 |   19.3 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>     |_inst_fib/7                                             |      1 |      3 | monotonic_clock | sec    |  0.701 |  0.701 |  0.701 |  0.701 |  0.000 |   14.3 |
| >>>       |_inst_fib/6                                           |      1 |      4 | monotonic_clock | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
| >>>         |_inst_fib/5                                         |      1 |      5 | monotonic_clock | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>           |_inst_fib/4                                       |      1 |      6 | monotonic_clock | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>             |_inst_fib/3                                     |      1 |      7 | monotonic_clock | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>               |_inst_fib/2                                   |      1 |      8 | monotonic_clock | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                 |_inst_fib/1                                 |      1 |      9 | monotonic_clock | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>> |_nested/3                                                   |      2 |      1 | monotonic_clock | sec    |  5.182 |  2.591 |  2.445 |  2.737 |  0.207 |   21.4 |
| >>>   |_inst_fib                                                 |      2 |      2 | monotonic_clock | sec    |  3.405 |  1.702 |  1.402 |  2.003 |  0.425 |    5.9 |
| >>>     |_inst_fib                                               |      2 |      3 | monotonic_clock | sec    |  3.204 |  1.602 |  1.302 |  1.903 |  0.425 |    6.3 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>       |_inst_fib                                             |      2 |      4 | monotonic_clock | sec    |  3.004 |  1.502 |  1.202 |  1.803 |  0.425 |    6.7 |
| >>>         |_inst_fib                                           |      2 |      5 | monotonic_clock | sec    |  2.804 |  1.402 |  1.101 |  1.702 |  0.425 |    7.1 |
| >>>           |_inst_fib                                         |      2 |      6 | monotonic_clock | sec    |  2.604 |  1.302 |  1.001 |  1.602 |  0.425 |    7.7 |
| >>>             |_inst_fib                                       |      2 |      7 | monotonic_clock | sec    |  2.403 |  1.202 |  0.901 |  1.502 |  0.425 |    8.3 |
| >>>               |_inst_fib                                     |      2 |      8 | monotonic_clock | sec    |  2.203 |  1.102 |  0.801 |  1.402 |  0.425 |    9.1 |
| >>>                 |_inst_fib                                   |      2 |      9 | monotonic_clock | sec    |  2.003 |  1.001 |  0.701 |  1.302 |  0.425 |   10.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | monotonic_clock | sec    |  1.803 |  0.901 |  0.601 |  1.202 |  0.425 |   11.1 |
| >>>                     |_inst_fib                               |      2 |     11 | monotonic_clock | sec    |  1.602 |  0.801 |  0.501 |  1.102 |  0.425 |   12.5 |
| >>>                       |_inst_fib                             |      2 |     12 | monotonic_clock | sec    |  1.402 |  0.701 |  0.401 |  1.001 |  0.425 |   14.3 |
| >>>                         |_inst_fib                           |      2 |     13 | monotonic_clock | sec    |  1.202 |  0.601 |  0.300 |  0.901 |  0.425 |   16.7 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                           |_inst_fib                         |      2 |     14 | monotonic_clock | sec    |  1.001 |  0.501 |  0.200 |  0.801 |  0.425 |   20.0 |
| >>>                             |_inst_fib                       |      2 |     15 | monotonic_clock | sec    |  0.801 |  0.401 |  0.100 |  0.701 |  0.425 |   25.0 |
| >>>                               |_inst_fib                     |      1 |     16 | monotonic_clock | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
| >>>                                 |_inst_fib                   |      1 |     17 | monotonic_clock | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | monotonic_clock | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>                                     |_inst_fib               |      1 |     19 | monotonic_clock | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>                                       |_inst_fib             |      1 |     20 | monotonic_clock | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                                         |_inst_fib           |      1 |     21 | monotonic_clock | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | monotonic_clock | sec    |  0.667 |  0.667 |  0.667 |  0.667 |  0.000 |   24.9 |
| >>>     |_inst_fib/5                                             |      1 |      3 | monotonic_clock | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>       |_inst_fib/4                                           |      1 |      4 | monotonic_clock | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>         |_inst_fib/3                                         |      1 |      5 | monotonic_clock | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>           |_inst_fib/2                                       |      1 |      6 | monotonic_clock | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>             |_inst_fib/1                                     |      1 |      7 | monotonic_clock | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>> |_nested/4                                                   |      2 |      1 | monotonic_clock | sec    |  4.326 |  2.163 |  1.883 |  2.443 |  0.396 |   23.6 |
| >>>   |_inst_fib                                                 |      2 |      2 | monotonic_clock | sec    |  3.305 |  1.652 |  1.302 |  2.003 |  0.496 |    6.1 |
| >>>     |_inst_fib                                               |      2 |      3 | monotonic_clock | sec    |  3.104 |  1.552 |  1.202 |  1.903 |  0.496 |    6.5 |
| >>>       |_inst_fib                                             |      2 |      4 | monotonic_clock | sec    |  2.904 |  1.452 |  1.101 |  1.802 |  0.496 |    6.9 |
| >>>         |_inst_fib                                           |      2 |      5 | monotonic_clock | sec    |  2.704 |  1.352 |  1.001 |  1.702 |  0.496 |    7.4 |
| >>>           |_inst_fib                                         |      2 |      6 | monotonic_clock | sec    |  2.503 |  1.252 |  0.901 |  1.602 |  0.496 |    8.0 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>             |_inst_fib                                       |      2 |      7 | monotonic_clock | sec    |  2.303 |  1.152 |  0.801 |  1.502 |  0.496 |    8.7 |
| >>>               |_inst_fib                                     |      2 |      8 | monotonic_clock | sec    |  2.103 |  1.051 |  0.701 |  1.402 |  0.496 |    9.5 |
| >>>                 |_inst_fib                                   |      2 |      9 | monotonic_clock | sec    |  1.903 |  0.951 |  0.601 |  1.302 |  0.496 |   10.5 |
| >>>                   |_inst_fib                                 |      2 |     10 | monotonic_clock | sec    |  1.702 |  0.851 |  0.501 |  1.202 |  0.496 |   11.8 |
| >>>                     |_inst_fib                               |      2 |     11 | monotonic_clock | sec    |  1.502 |  0.751 |  0.401 |  1.101 |  0.496 |   13.3 |
| >>>                       |_inst_fib                             |      2 |     12 | monotonic_clock | sec    |  1.302 |  0.651 |  0.300 |  1.001 |  0.496 |   15.4 |
| >>>                         |_inst_fib                           |      2 |     13 | monotonic_clock | sec    |  1.101 |  0.551 |  0.200 |  0.901 |  0.496 |   18.2 |
| >>>                           |_inst_fib                         |      2 |     14 | monotonic_clock | sec    |  0.901 |  0.451 |  0.100 |  0.801 |  0.496 |   22.2 |
| >>>                             |_inst_fib                       |      1 |     15 | monotonic_clock | sec    |  0.701 |  0.701 |  0.701 |  0.701 |  0.000 |   14.3 |
| >>>                               |_inst_fib                     |      1 |     16 | monotonic_clock | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
|------------------------------------------------------------------|--------|--------|-----------------|--------|--------|--------|--------|--------|--------|--------|
| >>>                                 |_inst_fib                   |      1 |     17 | monotonic_clock | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | monotonic_clock | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>                                     |_inst_fib               |      1 |     19 | monotonic_clock | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>                                       |_inst_fib             |      1 |     20 | monotonic_clock | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                                         |_inst_fib           |      1 |     21 | monotonic_clock | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-cxx-statistics-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-cxx-statistics-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-cxx-statistics-output/wall.txt'...
Opening 'timemory-ex-cxx-statistics-output/wall.jpeg' for output...
Closed 'timemory-ex-cxx-statistics-output/wall.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                          REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                          |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                              LABEL                               | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> main@ex_statistics.cpp:103/total                             |     10 |      0 | wall   | sec    | 26.733 |  2.673 |  2.136 |  3.206 |  0.405 |    9.7 |
| >>> |_nested/0                                                   |      2 |      1 | wall   | sec    |  5.276 |  2.638 |  2.438 |  2.838 |  0.283 |   20.9 |
| >>>   |_inst_fib                                                 |      2 |      2 | wall   | sec    |  3.406 |  1.703 |  1.402 |  2.004 |  0.426 |    5.9 |
| >>>     |_inst_fib                                               |      2 |      3 | wall   | sec    |  3.206 |  1.603 |  1.302 |  1.904 |  0.426 |    6.3 |
| >>>       |_inst_fib                                             |      2 |      4 | wall   | sec    |  3.006 |  1.503 |  1.202 |  1.804 |  0.426 |    6.7 |
| >>>         |_inst_fib                                           |      2 |      5 | wall   | sec    |  2.805 |  1.403 |  1.101 |  1.704 |  0.426 |    7.1 |
| >>>           |_inst_fib                                         |      2 |      6 | wall   | sec    |  2.605 |  1.302 |  1.001 |  1.603 |  0.426 |    7.7 |
| >>>             |_inst_fib                                       |      2 |      7 | wall   | sec    |  2.404 |  1.202 |  0.901 |  1.503 |  0.426 |    8.3 |
| >>>               |_inst_fib                                     |      2 |      8 | wall   | sec    |  2.204 |  1.102 |  0.801 |  1.403 |  0.426 |    9.1 |
| >>>                 |_inst_fib                                   |      2 |      9 | wall   | sec    |  2.004 |  1.002 |  0.701 |  1.303 |  0.426 |   10.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>                   |_inst_fib                                 |      2 |     10 | wall   | sec    |  1.803 |  0.902 |  0.601 |  1.203 |  0.426 |   11.1 |
| >>>                     |_inst_fib                               |      2 |     11 | wall   | sec    |  1.603 |  0.802 |  0.501 |  1.102 |  0.425 |   12.5 |
| >>>                       |_inst_fib                             |      2 |     12 | wall   | sec    |  1.403 |  0.701 |  0.401 |  1.002 |  0.425 |   14.3 |
| >>>                         |_inst_fib                           |      2 |     13 | wall   | sec    |  1.202 |  0.601 |  0.300 |  0.902 |  0.425 |   16.7 |
| >>>                           |_inst_fib                         |      2 |     14 | wall   | sec    |  1.002 |  0.501 |  0.200 |  0.802 |  0.425 |   20.0 |
| >>>                             |_inst_fib                       |      2 |     15 | wall   | sec    |  0.802 |  0.401 |  0.100 |  0.701 |  0.425 |   25.0 |
| >>>                               |_inst_fib                     |      1 |     16 | wall   | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
| >>>                                 |_inst_fib                   |      1 |     17 | wall   | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | wall   | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>                                     |_inst_fib               |      1 |     19 | wall   | sec    |  0.301 |  0.301 |  0.301 |  0.301 |  0.000 |   33.3 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>                                       |_inst_fib             |      1 |     20 | wall   | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                                         |_inst_fib           |      1 |     21 | wall   | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | wall   | sec    |  0.766 |  0.766 |  0.766 |  0.766 |  0.000 |   21.6 |
| >>>     |_inst_fib/6                                             |      1 |      3 | wall   | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
| >>>       |_inst_fib/5                                           |      1 |      4 | wall   | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>         |_inst_fib/4                                         |      1 |      5 | wall   | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>           |_inst_fib/3                                       |      1 |      6 | wall   | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>             |_inst_fib/2                                     |      1 |      7 | wall   | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>               |_inst_fib/1                                   |      1 |      8 | wall   | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>> |_nested/1                                                   |      2 |      1 | wall   | sec    |  4.825 |  2.412 |  1.883 |  2.942 |  0.749 |   21.1 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>   |_inst_fib                                                 |      2 |      2 | wall   | sec    |  3.807 |  1.903 |  1.303 |  2.504 |  0.849 |    5.3 |
| >>>     |_inst_fib                                               |      2 |      3 | wall   | sec    |  3.606 |  1.803 |  1.203 |  2.404 |  0.849 |    5.6 |
| >>>       |_inst_fib                                             |      2 |      4 | wall   | sec    |  3.406 |  1.703 |  1.102 |  2.303 |  0.849 |    5.9 |
| >>>         |_inst_fib                                           |      2 |      5 | wall   | sec    |  3.205 |  1.603 |  1.002 |  2.203 |  0.849 |    6.3 |
| >>>           |_inst_fib                                         |      2 |      6 | wall   | sec    |  3.005 |  1.502 |  0.902 |  2.103 |  0.849 |    6.7 |
| >>>             |_inst_fib                                       |      2 |      7 | wall   | sec    |  2.804 |  1.402 |  0.802 |  2.003 |  0.849 |    7.1 |
| >>>               |_inst_fib                                     |      2 |      8 | wall   | sec    |  2.604 |  1.302 |  0.701 |  1.903 |  0.849 |    7.7 |
| >>>                 |_inst_fib                                   |      2 |      9 | wall   | sec    |  2.404 |  1.202 |  0.601 |  1.802 |  0.849 |    8.3 |
| >>>                   |_inst_fib                                 |      2 |     10 | wall   | sec    |  2.203 |  1.102 |  0.501 |  1.702 |  0.849 |    9.1 |
| >>>                     |_inst_fib                               |      2 |     11 | wall   | sec    |  2.003 |  1.001 |  0.401 |  1.602 |  0.850 |   10.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>                       |_inst_fib                             |      2 |     12 | wall   | sec    |  1.803 |  0.901 |  0.301 |  1.502 |  0.850 |   11.1 |
| >>>                         |_inst_fib                           |      2 |     13 | wall   | sec    |  1.602 |  0.801 |  0.200 |  1.402 |  0.850 |   12.5 |
| >>>                           |_inst_fib                         |      2 |     14 | wall   | sec    |  1.402 |  0.701 |  0.100 |  1.302 |  0.850 |   14.3 |
| >>>                             |_inst_fib                       |      1 |     15 | wall   | sec    |  1.202 |  1.202 |  1.202 |  1.202 |  0.000 |    8.3 |
| >>>                               |_inst_fib                     |      1 |     16 | wall   | sec    |  1.102 |  1.102 |  1.102 |  1.102 |  0.000 |    9.1 |
| >>>                                 |_inst_fib                   |      1 |     17 | wall   | sec    |  1.001 |  1.001 |  1.001 |  1.001 |  0.000 |   10.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | wall   | sec    |  0.901 |  0.901 |  0.901 |  0.901 |  0.000 |   11.1 |
| >>>                                     |_inst_fib               |      1 |     19 | wall   | sec    |  0.801 |  0.801 |  0.801 |  0.801 |  0.000 |   12.5 |
| >>>                                       |_inst_fib             |      1 |     20 | wall   | sec    |  0.701 |  0.701 |  0.701 |  0.701 |  0.000 |   14.3 |
| >>>                                         |_inst_fib           |      1 |     21 | wall   | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>                                           |_inst_fib         |      1 |     22 | wall   | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>                                             |_inst_fib       |      1 |     23 | wall   | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>                                               |_inst_fib     |      1 |     24 | wall   | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>                                                 |_inst_fib   |      1 |     25 | wall   | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                                                   |_inst_fib |      1 |     26 | wall   | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>> |_nested/2                                                   |      2 |      1 | wall   | sec    |  4.521 |  2.261 |  1.881 |  2.640 |  0.537 |   27.6 |
| >>>   |_inst_fib                                                 |      2 |      2 | wall   | sec    |  2.403 |  1.202 |  1.101 |  1.302 |  0.142 |    8.3 |
| >>>     |_inst_fib                                               |      2 |      3 | wall   | sec    |  2.203 |  1.102 |  1.001 |  1.202 |  0.142 |    9.1 |
| >>>       |_inst_fib                                             |      2 |      4 | wall   | sec    |  2.003 |  1.001 |  0.901 |  1.102 |  0.142 |   10.0 |
| >>>         |_inst_fib                                           |      2 |      5 | wall   | sec    |  1.802 |  0.901 |  0.801 |  1.001 |  0.142 |   11.1 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>           |_inst_fib                                         |      2 |      6 | wall   | sec    |  1.602 |  0.801 |  0.701 |  0.901 |  0.142 |   12.5 |
| >>>             |_inst_fib                                       |      2 |      7 | wall   | sec    |  1.402 |  0.701 |  0.601 |  0.801 |  0.142 |   14.3 |
| >>>               |_inst_fib                                     |      2 |      8 | wall   | sec    |  1.202 |  0.601 |  0.501 |  0.701 |  0.142 |   16.7 |
| >>>                 |_inst_fib                                   |      2 |      9 | wall   | sec    |  1.001 |  0.501 |  0.401 |  0.601 |  0.142 |   20.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | wall   | sec    |  0.801 |  0.401 |  0.300 |  0.501 |  0.142 |   25.0 |
| >>>                     |_inst_fib                               |      2 |     11 | wall   | sec    |  0.601 |  0.300 |  0.200 |  0.401 |  0.142 |   33.3 |
| >>>                       |_inst_fib                             |      2 |     12 | wall   | sec    |  0.401 |  0.200 |  0.100 |  0.300 |  0.142 |   50.0 |
| >>>                         |_inst_fib                           |      1 |     13 | wall   | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                           |_inst_fib                         |      1 |     14 | wall   | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | wall   | sec    |  0.869 |  0.869 |  0.869 |  0.869 |  0.000 |   19.3 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>     |_inst_fib/7                                             |      1 |      3 | wall   | sec    |  0.701 |  0.701 |  0.701 |  0.701 |  0.000 |   14.3 |
| >>>       |_inst_fib/6                                           |      1 |      4 | wall   | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
| >>>         |_inst_fib/5                                         |      1 |      5 | wall   | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>           |_inst_fib/4                                       |      1 |      6 | wall   | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>             |_inst_fib/3                                     |      1 |      7 | wall   | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>               |_inst_fib/2                                   |      1 |      8 | wall   | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                 |_inst_fib/1                                 |      1 |      9 | wall   | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>> |_nested/3                                                   |      2 |      1 | wall   | sec    |  5.182 |  2.591 |  2.445 |  2.737 |  0.207 |   21.4 |
| >>>   |_inst_fib                                                 |      2 |      2 | wall   | sec    |  3.405 |  1.702 |  1.402 |  2.003 |  0.425 |    5.9 |
| >>>     |_inst_fib                                               |      2 |      3 | wall   | sec    |  3.204 |  1.602 |  1.302 |  1.903 |  0.425 |    6.3 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>       |_inst_fib                                             |      2 |      4 | wall   | sec    |  3.004 |  1.502 |  1.202 |  1.803 |  0.425 |    6.7 |
| >>>         |_inst_fib                                           |      2 |      5 | wall   | sec    |  2.804 |  1.402 |  1.101 |  1.702 |  0.425 |    7.1 |
| >>>           |_inst_fib                                         |      2 |      6 | wall   | sec    |  2.604 |  1.302 |  1.001 |  1.602 |  0.425 |    7.7 |
| >>>             |_inst_fib                                       |      2 |      7 | wall   | sec    |  2.403 |  1.202 |  0.901 |  1.502 |  0.425 |    8.3 |
| >>>               |_inst_fib                                     |      2 |      8 | wall   | sec    |  2.203 |  1.102 |  0.801 |  1.402 |  0.425 |    9.1 |
| >>>                 |_inst_fib                                   |      2 |      9 | wall   | sec    |  2.003 |  1.001 |  0.701 |  1.302 |  0.425 |   10.0 |
| >>>                   |_inst_fib                                 |      2 |     10 | wall   | sec    |  1.803 |  0.901 |  0.601 |  1.202 |  0.425 |   11.1 |
| >>>                     |_inst_fib                               |      2 |     11 | wall   | sec    |  1.602 |  0.801 |  0.501 |  1.102 |  0.425 |   12.5 |
| >>>                       |_inst_fib                             |      2 |     12 | wall   | sec    |  1.402 |  0.701 |  0.401 |  1.001 |  0.425 |   14.3 |
| >>>                         |_inst_fib                           |      2 |     13 | wall   | sec    |  1.202 |  0.601 |  0.300 |  0.901 |  0.425 |   16.7 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>                           |_inst_fib                         |      2 |     14 | wall   | sec    |  1.001 |  0.501 |  0.200 |  0.801 |  0.425 |   20.0 |
| >>>                             |_inst_fib                       |      2 |     15 | wall   | sec    |  0.801 |  0.401 |  0.100 |  0.701 |  0.425 |   25.0 |
| >>>                               |_inst_fib                     |      1 |     16 | wall   | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
| >>>                                 |_inst_fib                   |      1 |     17 | wall   | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | wall   | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>                                     |_inst_fib               |      1 |     19 | wall   | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>                                       |_inst_fib             |      1 |     20 | wall   | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                                         |_inst_fib           |      1 |     21 | wall   | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>>   |_main/occasional/2                                        |      1 |      2 | wall   | sec    |  0.667 |  0.667 |  0.667 |  0.667 |  0.000 |   24.9 |
| >>>     |_inst_fib/5                                             |      1 |      3 | wall   | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>       |_inst_fib/4                                           |      1 |      4 | wall   | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>         |_inst_fib/3                                         |      1 |      5 | wall   | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>           |_inst_fib/2                                       |      1 |      6 | wall   | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>             |_inst_fib/1                                     |      1 |      7 | wall   | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
| >>> |_nested/4                                                   |      2 |      1 | wall   | sec    |  4.326 |  2.163 |  1.883 |  2.443 |  0.396 |   23.6 |
| >>>   |_inst_fib                                                 |      2 |      2 | wall   | sec    |  3.305 |  1.652 |  1.302 |  2.003 |  0.496 |    6.1 |
| >>>     |_inst_fib                                               |      2 |      3 | wall   | sec    |  3.104 |  1.552 |  1.202 |  1.903 |  0.496 |    6.5 |
| >>>       |_inst_fib                                             |      2 |      4 | wall   | sec    |  2.904 |  1.452 |  1.101 |  1.802 |  0.496 |    6.9 |
| >>>         |_inst_fib                                           |      2 |      5 | wall   | sec    |  2.704 |  1.352 |  1.001 |  1.702 |  0.496 |    7.4 |
| >>>           |_inst_fib                                         |      2 |      6 | wall   | sec    |  2.503 |  1.252 |  0.901 |  1.602 |  0.496 |    8.0 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>             |_inst_fib                                       |      2 |      7 | wall   | sec    |  2.303 |  1.152 |  0.801 |  1.502 |  0.496 |    8.7 |
| >>>               |_inst_fib                                     |      2 |      8 | wall   | sec    |  2.103 |  1.051 |  0.701 |  1.402 |  0.496 |    9.5 |
| >>>                 |_inst_fib                                   |      2 |      9 | wall   | sec    |  1.903 |  0.951 |  0.601 |  1.302 |  0.496 |   10.5 |
| >>>                   |_inst_fib                                 |      2 |     10 | wall   | sec    |  1.702 |  0.851 |  0.501 |  1.202 |  0.496 |   11.8 |
| >>>                     |_inst_fib                               |      2 |     11 | wall   | sec    |  1.502 |  0.751 |  0.401 |  1.101 |  0.496 |   13.3 |
| >>>                       |_inst_fib                             |      2 |     12 | wall   | sec    |  1.302 |  0.651 |  0.300 |  1.001 |  0.496 |   15.4 |
| >>>                         |_inst_fib                           |      2 |     13 | wall   | sec    |  1.101 |  0.551 |  0.200 |  0.901 |  0.496 |   18.2 |
| >>>                           |_inst_fib                         |      2 |     14 | wall   | sec    |  0.901 |  0.451 |  0.100 |  0.801 |  0.496 |   22.2 |
| >>>                             |_inst_fib                       |      1 |     15 | wall   | sec    |  0.701 |  0.701 |  0.701 |  0.701 |  0.000 |   14.3 |
| >>>                               |_inst_fib                     |      1 |     16 | wall   | sec    |  0.601 |  0.601 |  0.601 |  0.601 |  0.000 |   16.7 |
|------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>>                                 |_inst_fib                   |      1 |     17 | wall   | sec    |  0.501 |  0.501 |  0.501 |  0.501 |  0.000 |   20.0 |
| >>>                                   |_inst_fib                 |      1 |     18 | wall   | sec    |  0.401 |  0.401 |  0.401 |  0.401 |  0.000 |   25.0 |
| >>>                                     |_inst_fib               |      1 |     19 | wall   | sec    |  0.300 |  0.300 |  0.300 |  0.300 |  0.000 |   33.3 |
| >>>                                       |_inst_fib             |      1 |     20 | wall   | sec    |  0.200 |  0.200 |  0.200 |  0.200 |  0.000 |   50.0 |
| >>>                                         |_inst_fib           |      1 |     21 | wall   | sec    |  0.100 |  0.100 |  0.100 |  0.100 |  0.000 |  100.0 |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|


[metadata::manager::finalize]> Outputting 'timemory-ex-cxx-statistics-output/metadata.json'...


#---------------------- tim::manager destroyed [rank=0][id=0][pid=13711] ----------------------#
```