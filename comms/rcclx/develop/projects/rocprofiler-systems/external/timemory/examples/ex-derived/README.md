# ex-derived

This example demonstrates the construction of a custom (derived) component from existing timemory components and then using the derived component for performance measurement. In this example code, we construct a derived component called `derived_cpu_util` which combines measurements from `cpu clock and system clock` or `user clock, system clock and wall clock` to derive cpu utilization as `100 x cpu_clock/wall_clock` or `100 x ((user_clock + system_clock) / wall_clock)` respectively.

## Build

See [examples](../README.md##Build).

## Expected Output
```bash
$ ./ex_derived
#------------------------- tim::manager initialized [id=0][pid=6018] -------------------------#

>>>  triplet_tuple/0 :        103.3 % derived_cpu_util,        0.165 sec wall,        0.170 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/0  :          0.0 % derived_cpu_util,        0.658 sec wall,        0.160 sec user,  [laps: 1]

Answer = 331160282

>>>  pair_list0      :         43.7 % derived_cpu_util,        1.328 sec wall,        0.580 sec cpu,  [laps: 1]
>>>  pair_tuple      :         53.2 % derived_cpu_util,        1.599 sec wall,        0.850 sec cpu [laps: 1]
>>>  triplet_tuple/1 :         98.4 % derived_cpu_util,        0.163 sec wall,        0.160 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/1  :          0.0 % derived_cpu_util,        0.658 sec wall,        0.160 sec user,  [laps: 1]

Answer = 394406268

>>>  pair_list1      :         49.2 % derived_cpu_util,        1.484 sec wall,        0.730 sec cpu,  [laps: 1]
>>>  pair_tuple      :         56.7 % derived_cpu_util,        1.745 sec wall,        0.990 sec cpu [laps: 1]
>>>  triplet_tuple/2 :         98.2 % derived_cpu_util,        0.163 sec wall,        0.160 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/2  :          0.0 % derived_cpu_util,        0.658 sec wall,        0.160 sec user,  [laps: 1]

Answer = 496740423

>>>  pair_list2      :         56.9 % derived_cpu_util,        1.740 sec wall,        0.990 sec cpu,  [laps: 1]
>>>  pair_tuple      :         62.5 % derived_cpu_util,        1.999 sec wall,        1.250 sec cpu [laps: 1]
>>>  triplet_tuple/3 :         98.8 % derived_cpu_util,        0.162 sec wall,        0.160 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/3  :          0.0 % derived_cpu_util,        0.658 sec wall,        0.160 sec user,  [laps: 1]

Answer = 331160282

>>>  pair_list3      :         44.5 % derived_cpu_util,        1.326 sec wall,        0.590 sec cpu,  [laps: 1]
>>>  pair_tuple      :         53.5 % derived_cpu_util,        1.590 sec wall,        0.850 sec cpu [laps: 1]
>>>  triplet_tuple/4 :         98.6 % derived_cpu_util,        0.162 sec wall,        0.160 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/4  :          0.0 % derived_cpu_util,        0.658 sec wall,        0.160 sec user,  [laps: 1]

Answer = 394406268

>>>  pair_list4      :         49.2 % derived_cpu_util,        1.484 sec wall,        0.730 sec cpu,  [laps: 1]
>>>  pair_tuple      :         56.8 % derived_cpu_util,        1.743 sec wall,        0.990 sec cpu [laps: 1]
>>>  triplet_tuple/5 :        101.9 % derived_cpu_util,        0.167 sec wall,        0.170 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/5  :         22.8 % derived_cpu_util,        0.658 sec wall,        0.150 sec user,        0.000 sec sys [laps: 1]

Answer = 496740423

>>>  pair_list5      :        1.744 sec wall,        0.990 sec cpu,  [laps: 1]
>>>  pair_tuple      :         62.4 % derived_cpu_util,        2.003 sec wall,        1.250 sec cpu [laps: 1]
>>>  triplet_tuple/6 :         98.0 % derived_cpu_util,        0.163 sec wall,        0.160 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/6  :         24.3 % derived_cpu_util,        0.658 sec wall,        0.160 sec user,        0.000 sec sys [laps: 1]

Answer = 331160282

>>>  pair_list6      :        1.327 sec wall,        0.580 sec cpu,  [laps: 1]
>>>  pair_tuple      :         52.9 % derived_cpu_util,        1.587 sec wall,        0.840 sec cpu [laps: 1]
>>>  triplet_tuple/7 :        104.4 % derived_cpu_util,        0.163 sec wall,        0.170 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/7  :         22.8 % derived_cpu_util,        0.658 sec wall,        0.150 sec user,        0.000 sec sys [laps: 1]

Answer = 394406268

>>>  pair_list7      :        1.484 sec wall,        0.730 sec cpu,  [laps: 1]
>>>  pair_tuple      :         56.7 % derived_cpu_util,        1.745 sec wall,        0.990 sec cpu [laps: 1]
>>>  triplet_tuple/8 :         98.8 % derived_cpu_util,        0.162 sec wall,        0.160 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/8  :         24.3 % derived_cpu_util,        0.658 sec wall,        0.160 sec user,        0.000 sec sys [laps: 1]

Answer = 496740423

>>>  pair_list8      :        1.739 sec wall,        0.990 sec cpu,  [laps: 1]
>>>  pair_tuple      :         63.0 % derived_cpu_util,        2.001 sec wall,        1.260 sec cpu [laps: 1]
>>>  triplet_tuple/9 :        101.5 % derived_cpu_util,        0.167 sec wall,        0.170 sec user,        0.000 sec sys,        0.000 MB peak_rss [laps: 1]
>>>  triplet_list/9  :         24.3 % derived_cpu_util,        0.658 sec wall,        0.160 sec user,        0.000 sec sys [laps: 1]

Answer = 331160282

>>>  pair_list9      :        1.331 sec wall,        0.590 sec cpu,  [laps: 1]
>>>  pair_tuple      :         52.8 % derived_cpu_util,        1.591 sec wall,        0.840 sec cpu [laps: 1]
[derived_cpu_util]|0> Outputting 'timemory-ex-derived-output/derived_cpu_util.json'...
[derived_cpu_util]|0> Outputting 'timemory-ex-derived-output/derived_cpu_util.txt'...
Opening 'timemory-ex-derived-output/derived_cpu_util.jpeg' for output...
Closed 'timemory-ex-derived-output/derived_cpu_util.jpeg'...

|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                     CPU UTILIZATION (DERIVED)                                                                   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|         LABEL           |   COUNT    |   DEPTH    |      METRIC      |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|-------------------------|------------|------------|------------------|------------|------------|------------|------------|------------|------------|------------|
| >>> pair_tuple          |         10 |          0 | derived_cpu_util | %          |      570.5 |       57.1 |       52.8 |       63.0 |        4.2 |        0.0 |
| >>> |_pair_list0        |          1 |          1 | derived_cpu_util | %          |       43.7 |       43.7 |       43.7 |       43.7 |        0.0 |        0.0 |
| >>>   |_triplet_tuple/0 |          1 |          2 | derived_cpu_util | %          |      103.3 |      103.3 |      103.3 |      103.3 |        0.0 |      100.0 |
| >>>   |_triplet_list/0  |          1 |          2 | derived_cpu_util | %          |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |
| >>> |_pair_list1        |          1 |          1 | derived_cpu_util | %          |       49.2 |       49.2 |       49.2 |       49.2 |        0.0 |        0.0 |
| >>>   |_triplet_tuple/1 |          1 |          2 | derived_cpu_util | %          |       98.4 |       98.4 |       98.4 |       98.4 |        0.0 |      100.0 |
| >>>   |_triplet_list/1  |          1 |          2 | derived_cpu_util | %          |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |
| >>> |_pair_list2        |          1 |          1 | derived_cpu_util | %          |       56.9 |       56.9 |       56.9 |       56.9 |        0.0 |        0.0 |
| >>>   |_triplet_tuple/2 |          1 |          2 | derived_cpu_util | %          |       98.2 |       98.2 |       98.2 |       98.2 |        0.0 |      100.0 |
| >>>   |_triplet_list/2  |          1 |          2 | derived_cpu_util | %          |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |
|-------------------------|------------|------------|------------------|------------|------------|------------|------------|------------|------------|------------|
| >>> |_pair_list3        |          1 |          1 | derived_cpu_util | %          |       44.5 |       44.5 |       44.5 |       44.5 |        0.0 |        0.0 |
| >>>   |_triplet_tuple/3 |          1 |          2 | derived_cpu_util | %          |       98.8 |       98.8 |       98.8 |       98.8 |        0.0 |      100.0 |
| >>>   |_triplet_list/3  |          1 |          2 | derived_cpu_util | %          |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |
| >>> |_pair_list4        |          1 |          1 | derived_cpu_util | %          |       49.2 |       49.2 |       49.2 |       49.2 |        0.0 |        0.0 |
| >>>   |_triplet_tuple/4 |          1 |          2 | derived_cpu_util | %          |       98.6 |       98.6 |       98.6 |       98.6 |        0.0 |      100.0 |
| >>>   |_triplet_list/4  |          1 |          2 | derived_cpu_util | %          |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |        0.0 |
| >>> |_triplet_tuple/5   |          1 |          1 | derived_cpu_util | %          |      101.9 |      101.9 |      101.9 |      101.9 |        0.0 |      100.0 |
| >>> |_triplet_list/5    |          1 |          1 | derived_cpu_util | %          |       22.8 |       22.8 |       22.8 |       22.8 |        0.0 |      100.0 |
| >>> |_triplet_tuple/6   |          1 |          1 | derived_cpu_util | %          |       98.0 |       98.0 |       98.0 |       98.0 |        0.0 |      100.0 |
| >>> |_triplet_list/6    |          1 |          1 | derived_cpu_util | %          |       24.3 |       24.3 |       24.3 |       24.3 |        0.0 |      100.0 |
|-------------------------|------------|------------|------------------|------------|------------|------------|------------|------------|------------|------------|
| >>> |_triplet_tuple/7   |          1 |          1 | derived_cpu_util | %          |      104.4 |      104.4 |      104.4 |      104.4 |        0.0 |      100.0 |
| >>> |_triplet_list/7    |          1 |          1 | derived_cpu_util | %          |       22.8 |       22.8 |       22.8 |       22.8 |        0.0 |      100.0 |
| >>> |_triplet_tuple/8   |          1 |          1 | derived_cpu_util | %          |       98.8 |       98.8 |       98.8 |       98.8 |        0.0 |      100.0 |
| >>> |_triplet_list/8    |          1 |          1 | derived_cpu_util | %          |       24.3 |       24.3 |       24.3 |       24.3 |        0.0 |      100.0 |
| >>> |_triplet_tuple/9   |          1 |          1 | derived_cpu_util | %          |      101.5 |      101.5 |      101.5 |      101.5 |        0.0 |      100.0 |
| >>> |_triplet_list/9    |          1 |          1 | derived_cpu_util | %          |       24.3 |       24.3 |       24.3 |       24.3 |        0.0 |      100.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|

[cpu]|0> Outputting 'timemory-ex-derived-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-ex-derived-output/cpu.json'...
[cpu]|0> Outputting 'timemory-ex-derived-output/cpu.txt'...
Opening 'timemory-ex-derived-output/cpu.jpeg' for output...
Closed 'timemory-ex-derived-output/cpu.jpeg'...

|---------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                    TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                                                   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|
|        LABEL          |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|-----------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> pair_tuple        |         10 |          0 | cpu        | sec        |     10.110 |      1.011 |      0.840 |      1.260 |      0.179 |       25.8 |
| >>> |_pair_list0      |          1 |          1 | cpu        | sec        |      0.580 |      0.580 |      0.580 |      0.580 |      0.000 |      100.0 |
| >>> |_pair_list1      |          1 |          1 | cpu        | sec        |      0.730 |      0.730 |      0.730 |      0.730 |      0.000 |      100.0 |
| >>> |_pair_list2      |          1 |          1 | cpu        | sec        |      0.990 |      0.990 |      0.990 |      0.990 |      0.000 |      100.0 |
| >>> |_pair_list3      |          1 |          1 | cpu        | sec        |      0.590 |      0.590 |      0.590 |      0.590 |      0.000 |      100.0 |
| >>> |_pair_list4      |          1 |          1 | cpu        | sec        |      0.730 |      0.730 |      0.730 |      0.730 |      0.000 |      100.0 |
| >>> |_pair_list5      |          1 |          1 | cpu        | sec        |      0.990 |      0.990 |      0.990 |      0.990 |      0.000 |      100.0 |
| >>> |_pair_list6      |          1 |          1 | cpu        | sec        |      0.580 |      0.580 |      0.580 |      0.580 |      0.000 |      100.0 |
| >>> |_pair_list7      |          1 |          1 | cpu        | sec        |      0.730 |      0.730 |      0.730 |      0.730 |      0.000 |      100.0 |
| >>> |_pair_list8      |          1 |          1 | cpu        | sec        |      0.990 |      0.990 |      0.990 |      0.990 |      0.000 |      100.0 |
|-----------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> |_pair_list9      |          1 |          1 | cpu        | sec        |      0.590 |      0.590 |      0.590 |      0.590 |      0.000 |      100.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|

[user]|0> Outputting 'timemory-ex-derived-output/user.flamegraph.json'...
[user]|0> Outputting 'timemory-ex-derived-output/user.json'...
[user]|0> Outputting 'timemory-ex-derived-output/user.txt'...
Opening 'timemory-ex-derived-output/user.jpeg' for output...
Closed 'timemory-ex-derived-output/user.jpeg'...

|-------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                               CPU TIME SPENT IN USER-MODE                                                             |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|
|       LABEL         |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|---------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> triplet_tuple/0 |          1 |          0 | user       | sec        |      0.170 |      0.170 |      0.170 |      0.170 |      0.000 |      100.0 |
| >>> triplet_list/0  |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_tuple/1 |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_list/1  |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_tuple/2 |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_list/2  |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_tuple/3 |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_list/3  |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_tuple/4 |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_list/4  |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
|---------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> triplet_tuple/5 |          1 |          0 | user       | sec        |      0.170 |      0.170 |      0.170 |      0.170 |      0.000 |      100.0 |
| >>> triplet_list/5  |          1 |          0 | user       | sec        |      0.150 |      0.150 |      0.150 |      0.150 |      0.000 |      100.0 |
| >>> triplet_tuple/6 |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_list/6  |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_tuple/7 |          1 |          0 | user       | sec        |      0.170 |      0.170 |      0.170 |      0.170 |      0.000 |      100.0 |
| >>> triplet_list/7  |          1 |          0 | user       | sec        |      0.150 |      0.150 |      0.150 |      0.150 |      0.000 |      100.0 |
| >>> triplet_tuple/8 |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_list/8  |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
| >>> triplet_tuple/9 |          1 |          0 | user       | sec        |      0.170 |      0.170 |      0.170 |      0.170 |      0.000 |      100.0 |
| >>> triplet_list/9  |          1 |          0 | user       | sec        |      0.160 |      0.160 |      0.160 |      0.160 |      0.000 |      100.0 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|

[sys]|0> Outputting 'timemory-ex-derived-output/sys.flamegraph.json'...
[sys]|0> Outputting 'timemory-ex-derived-output/sys.json'...
[sys]|0> Outputting 'timemory-ex-derived-output/sys.txt'...
Opening 'timemory-ex-derived-output/sys.jpeg' for output...
Closed 'timemory-ex-derived-output/sys.jpeg'...

|-------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                              CPU TIME SPENT IN KERNEL-MODE                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|
|       LABEL         |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|---------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> triplet_tuple/0 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/1 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/2 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/3 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/4 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/5 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_list/5  |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/6 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_list/6  |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/7 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|---------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> triplet_list/7  |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/8 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_list/8  |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/9 |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_list/9  |          1 |          0 | sys        | sec        |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-derived-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-derived-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-derived-output/wall.txt'...
Opening 'timemory-ex-derived-output/wall.jpeg' for output...
Closed 'timemory-ex-derived-output/wall.jpeg'...

|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                          REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                         |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
|         LABEL           |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|-------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> pair_tuple          |         10 |          0 | wall       | sec        |     17.603 |      1.760 |      1.587 |      2.003 |      0.179 |       14.9 |
| >>> |_pair_list0        |          1 |          1 | wall       | sec        |      1.328 |      1.328 |      1.328 |      1.328 |      0.000 |       38.1 |
| >>>   |_triplet_tuple/0 |          1 |          2 | wall       | sec        |      0.165 |      0.165 |      0.165 |      0.165 |      0.000 |      100.0 |
| >>>   |_triplet_list/0  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
| >>> |_pair_list1        |          1 |          1 | wall       | sec        |      1.484 |      1.484 |      1.484 |      1.484 |      0.000 |       44.7 |
| >>>   |_triplet_tuple/1 |          1 |          2 | wall       | sec        |      0.163 |      0.163 |      0.163 |      0.163 |      0.000 |      100.0 |
| >>>   |_triplet_list/1  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
| >>> |_pair_list2        |          1 |          1 | wall       | sec        |      1.740 |      1.740 |      1.740 |      1.740 |      0.000 |       52.8 |
| >>>   |_triplet_tuple/2 |          1 |          2 | wall       | sec        |      0.163 |      0.163 |      0.163 |      0.163 |      0.000 |      100.0 |
| >>>   |_triplet_list/2  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
|-------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> |_pair_list3        |          1 |          1 | wall       | sec        |      1.326 |      1.326 |      1.326 |      1.326 |      0.000 |       38.1 |
| >>>   |_triplet_tuple/3 |          1 |          2 | wall       | sec        |      0.162 |      0.162 |      0.162 |      0.162 |      0.000 |      100.0 |
| >>>   |_triplet_list/3  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
| >>> |_pair_list4        |          1 |          1 | wall       | sec        |      1.484 |      1.484 |      1.484 |      1.484 |      0.000 |       44.7 |
| >>>   |_triplet_tuple/4 |          1 |          2 | wall       | sec        |      0.162 |      0.162 |      0.162 |      0.162 |      0.000 |      100.0 |
| >>>   |_triplet_list/4  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
| >>> |_pair_list5        |          1 |          1 | wall       | sec        |      1.744 |      1.744 |      1.744 |      1.744 |      0.000 |       52.7 |
| >>>   |_triplet_tuple/5 |          1 |          2 | wall       | sec        |      0.167 |      0.167 |      0.167 |      0.167 |      0.000 |      100.0 |
| >>>   |_triplet_list/5  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
| >>> |_pair_list6        |          1 |          1 | wall       | sec        |      1.327 |      1.327 |      1.327 |      1.327 |      0.000 |       38.1 |
|-------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_triplet_tuple/6 |          1 |          2 | wall       | sec        |      0.163 |      0.163 |      0.163 |      0.163 |      0.000 |      100.0 |
| >>>   |_triplet_list/6  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
| >>> |_pair_list7        |          1 |          1 | wall       | sec        |      1.484 |      1.484 |      1.484 |      1.484 |      0.000 |       44.7 |
| >>>   |_triplet_tuple/7 |          1 |          2 | wall       | sec        |      0.163 |      0.163 |      0.163 |      0.163 |      0.000 |      100.0 |
| >>>   |_triplet_list/7  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
| >>> |_pair_list8        |          1 |          1 | wall       | sec        |      1.739 |      1.739 |      1.739 |      1.739 |      0.000 |       52.8 |
| >>>   |_triplet_tuple/8 |          1 |          2 | wall       | sec        |      0.162 |      0.162 |      0.162 |      0.162 |      0.000 |      100.0 |
| >>>   |_triplet_list/8  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
| >>> |_pair_list9        |          1 |          1 | wall       | sec        |      1.331 |      1.331 |      1.331 |      1.331 |      0.000 |       38.0 |
| >>>   |_triplet_tuple/9 |          1 |          2 | wall       | sec        |      0.167 |      0.167 |      0.167 |      0.167 |      0.000 |      100.0 |
|-------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_triplet_list/9  |          1 |          2 | wall       | sec        |      0.658 |      0.658 |      0.658 |      0.658 |      0.000 |      100.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|

[peak_rss]|0> Outputting 'timemory-ex-derived-output/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-ex-derived-output/peak_rss.txt'...
Opening 'timemory-ex-derived-output/peak_rss.jpeg' for output...
Closed 'timemory-ex-derived-output/peak_rss.jpeg'...

|-------------------------------------------------------------------------------------------------------------------------------------------------------|
|                   MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF SWAP IS ENABLED                 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|
|       LABEL         |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|---------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> triplet_tuple/0 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/1 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/2 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/3 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/4 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/5 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/6 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/7 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/8 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>> triplet_tuple/9 |          1 |          0 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|


[metadata::manager::finalize]> Outputting 'timemory-ex-derived-output/metadata.json'...


#---------------------- tim::manager destroyed [rank=0][id=0][pid=6018] ----------------------#
```
