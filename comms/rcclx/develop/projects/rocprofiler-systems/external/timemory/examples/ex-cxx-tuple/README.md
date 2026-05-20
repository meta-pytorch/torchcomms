# ex-cxx-tuple

This example demonstrates the usage of timemory `auto_tuple` and `component_tuple` for bundling various timemory components into tuples for combines performance measurements.

## Build

See [examples](../README.md##Build).

## Expected Output
```bash

#------------------------- tim::manager initialized [id=0][pid=12303] -------------------------#

# tests: 3

[0] TESTING [test_1_usage]...

usage (begin): |0>>>  test_1_usage_begin :   48.344 MB peak_rss,   49.504 MB page_rss,  166.957 MB virtual_memory,   0 major_page_flts, 6131 minor_page_flts,  14 prio_cxt_swch, 100 vol_cxt_swch [laps: 1]
usage (delta): |0>>>  test_1_usage_delta :   39.248 MB peak_rss,   40.190 MB page_rss,   40.002 MB virtual_memory,   0 major_page_flts, 9769 minor_page_flts,   0 prio_cxt_swch,   0 vol_cxt_swch [laps: 1]
usage (end):   |0>>>  test_1_usage_end   :   87.592 MB peak_rss,   89.694 MB page_rss,  206.959 MB virtual_memory,   0 major_page_flts, 15903 minor_page_flts,  14 prio_cxt_swch, 100 vol_cxt_swch [laps: 1]

[0] TESTING [test_2_timing]...

[0] test_2_timing/250
total runtime: |0>>>  test_2_timing_runtime :    1.139 sec wall,    0.010 sec sys,    0.292 sec thread_cpu,   25.7 % thread_cpu_util,    1.746 sec process_cpu,  153.3 % proc_cpu_util,   1106329368 Total cycles,   3350858087 Instr completed,   1324720600 L/S completed papi-1 [laps: 1]
std::get:    1.139 sec wall
fibonacci total: 600420847

runtime process cpu time: 0x7fff5a25e488
[0] test_2_timing/255

[0] TESTING [test_3_measure]...

  Current rss: |0>>>  test_3_measure/ex_cxx_tuple.cpp:265 :   51.384 MB page_rss,   87.592 MB peak_rss
Change in rss: |0>>>  test_3_measure/ex_cxx_tuple.cpp:265 :   51.384 MB page_rss,   87.592 MB peak_rss [laps: 1]
  Current rss: |0>>>  test_3_measure/ex_cxx_tuple.cpp:265 :   51.388 MB page_rss,  128.056 MB peak_rss [laps: 1]

|0>>>  PAPI measurements :   1153732781 Total cycles,   3440079475 Instr completed,   1351354816 L/S completed papi-1 [laps: 1]
[0]
... [TESTING COMPLETED] ...

#==============================================================================#
#
#       [./ex_cxx_tuple] TESTS PASSED: 3/3
#
#==============================================================================#

[RANK: 0]
        >>> test_1_usage@ex_cxx_tuple.cpp:192            :    0.018 sec wall
        >>> |_time_fibonacci@ex_cxx_tuple.cpp:69         :    0.002 sec wall
        >>> test_2_timing_runtime                        :    1.139 sec wall
        >>> |_test_2_timing@ex_cxx_tuple.cpp:228         :    1.139 sec wall
        >>>   |_run_fib/40                               :    0.292 sec wall
        >>>     |_time_fibonacci@ex_cxx_tuple.cpp:69     :    0.292 sec wall
        >>>       |_run_fib/35                           :    0.309 sec wall
        >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 :    0.308 sec wall
        >>>       |_run_fib/43                           :    1.137 sec wall
        >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 :    1.137 sec wall
[papi-1]|0> Outputting 'timemory-ex-cxx-tuple-output/papi-1.xml'...
[papi-1]|0> Outputting 'timemory-ex-cxx-tuple-output/papi-1.txt'...
Traceback (most recent call last):
  File "/home/mhaseeb/repos/haseeb/timemory/build/timemory/plotting/__main__.py", line 112, in try_plot
    _jdata = json.load(f)
  File "/home/mhaseeb/repos/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/python-3.7.7-2dybrjceqs3qc4k7ci56t56bvzb4csxc/lib/python3.7/json/__init__.py", line 296, in load
    parse_constant=parse_constant, object_pairs_hook=object_pairs_hook, **kw)
  File "/home/mhaseeb/repos/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/python-3.7.7-2dybrjceqs3qc4k7ci56t56bvzb4csxc/lib/python3.7/json/__init__.py", line 348, in loads
    return _default_decoder.decode(s)
  File "/home/mhaseeb/repos/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/python-3.7.7-2dybrjceqs3qc4k7ci56t56bvzb4csxc/lib/python3.7/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/home/mhaseeb/repos/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/python-3.7.7-2dybrjceqs3qc4k7ci56t56bvzb4csxc/lib/python3.7/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
Exception - Expecting value: line 1 column 1 (char 0)
[timemory]> Command: '/home/mhaseeb/repos/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/python-3.7.7-2dybrjceqs3qc4k7ci56t56bvzb4csxc/bin/python -m timemory.plotting -f timemory-ex-cxx-tuple-output/papi-1.xml -t "papi-1 " -o timemory-ex-cxx-tuple-output' returned a non-zero exit code: 256... plot/definition.hpp:77 plot generation failed

|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                       LABEL                        |   COUNT    |   DEPTH    |     METRIC      |   UNITS    |      SUM        |      MEAN       |      MIN        |      MAX        |    STDDEV    |   % SELF   |
|----------------------------------------------------|------------|------------|-----------------|------------|-----------------|-----------------|-----------------|-----------------|--------------|------------|
| >>> PAPI measurements                              |          1 |          0 | Total cycles    |            |  1153732781.000 |  1153732781.000 |  1153732781.000 |  1153732781.000 |        0.000 |        2.2 |
|                                                    |            |            | Instr completed |            |  3440079475.000 |  3440079475.000 |  3440079475.000 |  3440079475.000 |        0.000 |        1.2 |
|                                                    |            |            | L/S completed   |            |  1351354816.000 |  1351354816.000 |  1351354816.000 |  1351354816.000 |        0.000 |        0.8 |
| >>> |_test_1_usage@ex_cxx_tuple.cpp:192            |          1 |          1 | Total cycles    |            |    22026299.000 |    22026299.000 |    22026299.000 |    22026299.000 |        0.000 |       62.1 |
|                                                    |            |            | Instr completed |            |    48265123.000 |    48265123.000 |    48265123.000 |    48265123.000 |        0.000 |       43.5 |
|                                                    |            |            | L/S completed   |            |    16200389.000 |    16200389.000 |    16200389.000 |    16200389.000 |        0.000 |       33.5 |
| >>>   |_time_fibonacci@ex_cxx_tuple.cpp:69         |          1 |          2 | Total cycles    |            |     8348996.000 |     8348996.000 |     8348996.000 |     8348996.000 |        0.000 |      100.0 |
|                                                    |            |            | Instr completed |            |    27251447.000 |    27251447.000 |    27251447.000 |    27251447.000 |        0.000 |      100.0 |
|                                                    |            |            | L/S completed   |            |    10773368.000 |    10773368.000 |    10773368.000 |    10773368.000 |        0.000 |      100.0 |
| >>> |_test_2_timing_runtime                        |          1 |          1 | Total cycles    |            |  1106329368.000 |  1106329368.000 |  1106329368.000 |  1106329368.000 |        0.000 |        0.0 |
|                                                    |            |            | Instr completed |            |  3350858087.000 |  3350858087.000 |  3350858087.000 |  3350858087.000 |        0.000 |        0.0 |
|                                                    |            |            | L/S completed   |            |  1324720600.000 |  1324720600.000 |  1324720600.000 |  1324720600.000 |        0.000 |        0.0 |
| >>>   |_test_2_timing@ex_cxx_tuple.cpp:228         |          1 |          2 | Total cycles    |            |  1106316448.000 |  1106316448.000 |  1106316448.000 |  1106316448.000 |        0.000 |        0.0 |
|                                                    |            |            | Instr completed |            |  3350840602.000 |  3350840602.000 |  3350840602.000 |  3350840602.000 |        0.000 |        0.0 |
|                                                    |            |            | L/S completed   |            |  1324712097.000 |  1324712097.000 |  1324712097.000 |  1324712097.000 |        0.000 |        0.0 |
| >>>     |_run_fib/40                               |          1 |          3 | Total cycles    |            |  1106022545.000 |  1106022545.000 |  1106022545.000 |  1106022545.000 |        0.000 |        0.0 |
|                                                    |            |            | Instr completed |            |  3350722429.000 |  3350722429.000 |  3350722429.000 |  3350722429.000 |        0.000 |        0.0 |
|                                                    |            |            | L/S completed   |            |  1324657228.000 |  1324657228.000 |  1324657228.000 |  1324657228.000 |        0.000 |        0.0 |
| >>>       |_time_fibonacci@ex_cxx_tuple.cpp:69     |          1 |          4 | Total cycles    |            |  1105972354.000 |  1105972354.000 |  1105972354.000 |  1105972354.000 |        0.000 |        0.0 |
|                                                    |            |            | Instr completed |            |  3350692563.000 |  3350692563.000 |  3350692563.000 |  3350692563.000 |        0.000 |        0.0 |
|                                                    |            |            | L/S completed   |            |  1324642414.000 |  1324642414.000 |  1324642414.000 |  1324642414.000 |        0.000 |        0.0 |
| >>>         |_run_fib/35                           |          7 |          5 | Total cycles    |            |   963952259.000 |   137707465.571 |    92187184.000 |   172834169.000 | 42501983.949 |        0.1 |
|                                                    |            |            | Instr completed |            |  2115196983.000 |   302170997.571 |   302170933.000 |   302171012.000 |       28.844 |        0.0 |
|                                                    |            |            | L/S completed   |            |   836235936.000 |   119462276.571 |   119462201.000 |   119462314.000 |       42.016 |        0.0 |
| >>>           |_time_fibonacci@ex_cxx_tuple.cpp:69 |          7 |          6 | Total cycles    |            |   963336278.000 |   137619468.286 |    92116775.000 |   172741354.000 | 42490426.970 |      100.0 |
|                                                    |            |            | Instr completed |            |  2114931893.000 |   302133127.571 |   302133119.000 |   302133138.000 |        8.000 |      100.0 |
|                                                    |            |            | L/S completed   |            |   836105354.000 |   119443622.000 |   119443565.000 |   119443657.000 |       36.258 |      100.0 |
| >>>         |_run_fib/43                           |          1 |          5 | Total cycles    |            |  4407531439.000 |  4407531439.000 |  4407531439.000 |  4407531439.000 |        0.000 |        0.0 |
|                                                    |            |            | Instr completed |            | 14193794478.000 | 14193794478.000 | 14193794478.000 | 14193794478.000 |        0.000 |        0.0 |
|                                                    |            |            | L/S completed   |            |  5611290601.000 |  5611290601.000 |  5611290601.000 |  5611290601.000 |        0.000 |        0.0 |
|----------------------------------------------------|------------|------------|-----------------|------------|-----------------|-----------------|-----------------|-----------------|--------------|------------|
| >>>           |_time_fibonacci@ex_cxx_tuple.cpp:69 |          1 |          6 | Total cycles    |            |  4407429762.000 |  4407429762.000 |  4407429762.000 |  4407429762.000 |        0.000 |      100.0 |
|                                                    |            |            | Instr completed |            | 14193756600.000 | 14193756600.000 | 14193756600.000 | 14193756600.000 |        0.000 |      100.0 |
|                                                    |            |            | L/S completed   |            |  5611271955.000 |  5611271955.000 |  5611271955.000 |  5611271955.000 |        0.000 |      100.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[proc_cpu_util]|0> Outputting 'timemory-ex-cxx-tuple-output/proc_cpu_util.json'...
[proc_cpu_util]|0> Outputting 'timemory-ex-cxx-tuple-output/proc_cpu_util.txt'...
Opening 'timemory-ex-cxx-tuple-output/proc_cpu_util.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/proc_cpu_util.jpeg'...

|--------------------------------------------------------------------------------------------------------------------------------------------|
|                          PERCENTAGE OF CPU-CLOCK TIME DIVIDED BY WALL-CLOCK TIME FOR CALLING PROCESS (ALL THREADS)                         |
|--------------------------------------------------------------------------------------------------------------------------------------------|
|                      LABEL                       | COUNT | DEPTH |    METRIC     | UNITS | SUM   | MEAN  | MIN   | MAX   | STDDEV | % SELF |
|--------------------------------------------------|-------|-------|---------------|-------|-------|-------|-------|-------|--------|--------|
| >>> test_1_usage@ex_cxx_tuple.cpp:192            |     1 |     0 | proc_cpu_util | %     |  99.9 |  99.9 |  99.9 |  99.9 |    0.0 |    0.0 |
| >>> |_time_fibonacci@ex_cxx_tuple.cpp:69         |     1 |     1 | proc_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
| >>> test_2_timing_runtime                        |     1 |     0 | proc_cpu_util | %     | 153.3 | 153.3 | 153.3 | 153.3 |    0.0 |    0.0 |
| >>> |_test_2_timing@ex_cxx_tuple.cpp:228         |     1 |     1 | proc_cpu_util | %     | 153.3 | 153.3 | 153.3 | 153.3 |    0.0 |    0.0 |
| >>>   |_run_fib/40                               |     1 |     2 | proc_cpu_util | %     | 306.8 | 306.8 | 306.8 | 306.8 |    0.0 |    0.0 |
| >>>     |_time_fibonacci@ex_cxx_tuple.cpp:69     |     1 |     3 | proc_cpu_util | %     | 306.8 | 306.8 | 306.8 | 306.8 |    0.0 |    0.0 |
| >>>       |_run_fib/35                           |     7 |     4 | proc_cpu_util | %     | 800.5 | 114.4 | 754.0 | 844.8 |   29.0 |    0.0 |
| >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     7 |     5 | proc_cpu_util | %     | 802.2 | 114.6 | 756.6 | 848.7 |   29.3 |  100.0 |
| >>>       |_run_fib/43                           |     1 |     4 | proc_cpu_util | %     | 153.1 | 153.1 | 153.1 | 153.1 |    0.0 |    0.0 |
| >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     1 |     5 | proc_cpu_util | %     | 153.0 | 153.0 | 153.0 | 153.0 |    0.0 |  100.0 |
|--------------------------------------------------------------------------------------------------------------------------------------------|

[process_cpu]|0> Outputting 'timemory-ex-cxx-tuple-output/process_cpu.flamegraph.json'...
[process_cpu]|0> Outputting 'timemory-ex-cxx-tuple-output/process_cpu.json'...
[process_cpu]|0> Outputting 'timemory-ex-cxx-tuple-output/process_cpu.txt'...
Opening 'timemory-ex-cxx-tuple-output/process_cpu.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/process_cpu.jpeg'...

|-------------------------------------------------------------------------------------------------------------------------------------------------|
|                                               CPU-CLOCK TIMER FOR THE CALLING PROCESS (ALL THREADS)                                             |
|-------------------------------------------------------------------------------------------------------------------------------------------------|
|                      LABEL                       | COUNT  | DEPTH  |   METRIC    | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|--------------------------------------------------|--------|--------|-------------|--------|--------|--------|--------|--------|--------|--------|
| >>> test_1_usage@ex_cxx_tuple.cpp:192            |      1 |      0 | process_cpu | sec    |  0.018 |  0.018 |  0.018 |  0.018 |  0.000 |   86.3 |
| >>> |_time_fibonacci@ex_cxx_tuple.cpp:69         |      1 |      1 | process_cpu | sec    |  0.002 |  0.002 |  0.002 |  0.002 |  0.000 |  100.0 |
| >>> test_2_timing_runtime                        |      1 |      0 | process_cpu | sec    |  1.746 |  1.746 |  1.746 |  1.746 |  0.000 |    0.0 |
| >>> |_test_2_timing@ex_cxx_tuple.cpp:228         |      1 |      1 | process_cpu | sec    |  1.746 |  1.746 |  1.746 |  1.746 |  0.000 |   48.7 |
| >>>   |_run_fib/40                               |      1 |      2 | process_cpu | sec    |  0.895 |  0.895 |  0.895 |  0.895 |  0.000 |    0.0 |
| >>>     |_time_fibonacci@ex_cxx_tuple.cpp:69     |      1 |      3 | process_cpu | sec    |  0.895 |  0.895 |  0.895 |  0.895 |  0.000 |    0.0 |
| >>>       |_run_fib/35                           |      7 |      4 | process_cpu | sec    |  2.477 |  0.354 |  0.287 |  0.421 |  0.062 |    0.1 |
| >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      7 |      5 | process_cpu | sec    |  2.475 |  0.354 |  0.287 |  0.420 |  0.062 |  100.0 |
| >>>       |_run_fib/43                           |      1 |      4 | process_cpu | sec    |  1.740 |  1.740 |  1.740 |  1.740 |  0.000 |    0.0 |
| >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | process_cpu | sec    |  1.739 |  1.739 |  1.739 |  1.739 |  0.000 |  100.0 |
|-------------------------------------------------------------------------------------------------------------------------------------------------|

[thread_cpu_util]|0> Outputting 'timemory-ex-cxx-tuple-output/thread_cpu_util.json'...
[thread_cpu_util]|0> Outputting 'timemory-ex-cxx-tuple-output/thread_cpu_util.txt'...
Opening 'timemory-ex-cxx-tuple-output/thread_cpu_util.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/thread_cpu_util.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------------|
|                                   PERCENTAGE OF CPU-CLOCK TIME DIVIDED BY WALL-CLOCK TIME FOR CALLING THREAD                                   |
|------------------------------------------------------------------------------------------------------------------------------------------------|
|                       LABEL                        | COUNT | DEPTH |     METRIC      | UNITS | SUM   | MEAN  | MIN   | MAX   | STDDEV | % SELF |
|----------------------------------------------------|-------|-------|-----------------|-------|-------|-------|-------|-------|--------|--------|
| |0>>> test_1_usage@ex_cxx_tuple.cpp:192            |     1 |     0 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |    0.0 |
| |0>>> |_time_fibonacci@ex_cxx_tuple.cpp:69         |     1 |     1 | thread_cpu_util | %     |  99.9 |  99.9 |  99.9 |  99.9 |    0.0 |  100.0 |
| |0>>> test_2_timing_runtime                        |     1 |     0 | thread_cpu_util | %     |  25.7 |  25.7 |  25.7 |  25.7 |    0.0 |    0.0 |
| |0>>> |_test_2_timing@ex_cxx_tuple.cpp:228         |     1 |     1 | thread_cpu_util | %     |  25.7 |  25.7 |  25.7 |  25.7 |    0.0 |    0.0 |
| |0>>>   |_run_fib/40                               |     1 |     2 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |    0.0 |
| |0>>>     |_time_fibonacci@ex_cxx_tuple.cpp:69     |     1 |     3 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |    0.0 |
| |1>>>       |_run_fib/35                           |     1 |     4 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |    0.0 |
| |1>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     1 |     5 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
| |3>>>       |_run_fib/35                           |     1 |     4 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |    0.0 |
| |3>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     1 |     5 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
|----------------------------------------------------|-------|-------|-----------------|-------|-------|-------|-------|-------|--------|--------|
| |6>>>       |_run_fib/35                           |     1 |     4 | thread_cpu_util | %     |  99.8 |  99.8 |  99.8 |  99.8 |    0.0 |    0.0 |
| |6>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     1 |     5 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
| |2>>>       |_run_fib/35                           |     1 |     4 | thread_cpu_util | %     |  99.9 |  99.9 |  99.9 |  99.9 |    0.0 |    0.0 |
| |2>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     1 |     5 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
| |7>>>       |_run_fib/35                           |     1 |     4 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |    0.0 |
| |7>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     1 |     5 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
| |8>>>       |_run_fib/35                           |     1 |     4 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |    0.0 |
| |8>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     1 |     5 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
| |5>>>       |_run_fib/35                           |     1 |     4 | thread_cpu_util | %     |  99.9 |  99.9 |  99.9 |  99.9 |    0.0 |    0.0 |
| |5>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     1 |     5 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
|----------------------------------------------------|-------|-------|-----------------|-------|-------|-------|-------|-------|--------|--------|
| |4>>>       |_run_fib/43                           |     1 |     4 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |    0.0 |
| |4>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |     1 |     5 | thread_cpu_util | %     | 100.0 | 100.0 | 100.0 | 100.0 |    0.0 |  100.0 |
|------------------------------------------------------------------------------------------------------------------------------------------------|

[thread_cpu]|0> Outputting 'timemory-ex-cxx-tuple-output/thread_cpu.flamegraph.json'...
[thread_cpu]|0> Outputting 'timemory-ex-cxx-tuple-output/thread_cpu.json'...
[thread_cpu]|0> Outputting 'timemory-ex-cxx-tuple-output/thread_cpu.txt'...
Opening 'timemory-ex-cxx-tuple-output/thread_cpu.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/thread_cpu.jpeg'...

|--------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                      CPU-CLOCK TIMER FOR THE CALLING THREAD                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------|
|                       LABEL                        | COUNT  | DEPTH  |   METRIC   | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|----------------------------------------------------|--------|--------|------------|--------|--------|--------|--------|--------|--------|--------|
| |0>>> test_1_usage@ex_cxx_tuple.cpp:192            |      1 |      0 | thread_cpu | sec    |  0.018 |  0.018 |  0.018 |  0.018 |  0.000 |   86.3 |
| |0>>> |_time_fibonacci@ex_cxx_tuple.cpp:69         |      1 |      1 | thread_cpu | sec    |  0.002 |  0.002 |  0.002 |  0.002 |  0.000 |  100.0 |
| |0>>> test_2_timing_runtime                        |      1 |      0 | thread_cpu | sec    |  0.292 |  0.292 |  0.292 |  0.292 |  0.000 |    0.0 |
| |0>>> |_test_2_timing@ex_cxx_tuple.cpp:228         |      1 |      1 | thread_cpu | sec    |  0.292 |  0.292 |  0.292 |  0.292 |  0.000 |    0.2 |
| |0>>>   |_run_fib/40                               |      1 |      2 | thread_cpu | sec    |  0.292 |  0.292 |  0.292 |  0.292 |  0.000 |    0.0 |
| |0>>>     |_time_fibonacci@ex_cxx_tuple.cpp:69     |      1 |      3 | thread_cpu | sec    |  0.292 |  0.292 |  0.292 |  0.292 |  0.000 |    0.0 |
| |1>>>       |_run_fib/35                           |      1 |      4 | thread_cpu | sec    |  0.035 |  0.035 |  0.035 |  0.035 |  0.000 |    0.2 |
| |1>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | thread_cpu | sec    |  0.035 |  0.035 |  0.035 |  0.035 |  0.000 |  100.0 |
| |3>>>       |_run_fib/35                           |      1 |      4 | thread_cpu | sec    |  0.035 |  0.035 |  0.035 |  0.035 |  0.000 |    0.2 |
| |3>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | thread_cpu | sec    |  0.035 |  0.035 |  0.035 |  0.035 |  0.000 |  100.0 |
|----------------------------------------------------|--------|--------|------------|--------|--------|--------|--------|--------|--------|--------|
| |6>>>       |_run_fib/35                           |      1 |      4 | thread_cpu | sec    |  0.034 |  0.034 |  0.034 |  0.034 |  0.000 |    0.4 |
| |6>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | thread_cpu | sec    |  0.034 |  0.034 |  0.034 |  0.034 |  0.000 |  100.0 |
| |2>>>       |_run_fib/35                           |      1 |      4 | thread_cpu | sec    |  0.049 |  0.049 |  0.049 |  0.049 |  0.000 |    0.2 |
| |2>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | thread_cpu | sec    |  0.049 |  0.049 |  0.049 |  0.049 |  0.000 |  100.0 |
| |7>>>       |_run_fib/35                           |      1 |      4 | thread_cpu | sec    |  0.049 |  0.049 |  0.049 |  0.049 |  0.000 |    0.1 |
| |7>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | thread_cpu | sec    |  0.048 |  0.048 |  0.048 |  0.048 |  0.000 |  100.0 |
| |8>>>       |_run_fib/35                           |      1 |      4 | thread_cpu | sec    |  0.051 |  0.051 |  0.051 |  0.051 |  0.000 |    0.1 |
| |8>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | thread_cpu | sec    |  0.051 |  0.051 |  0.051 |  0.051 |  0.000 |  100.0 |
| |5>>>       |_run_fib/35                           |      1 |      4 | thread_cpu | sec    |  0.056 |  0.056 |  0.056 |  0.056 |  0.000 |    0.4 |
| |5>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | thread_cpu | sec    |  0.056 |  0.056 |  0.056 |  0.056 |  0.000 |  100.0 |
|----------------------------------------------------|--------|--------|------------|--------|--------|--------|--------|--------|--------|--------|
| |4>>>       |_run_fib/43                           |      1 |      4 | thread_cpu | sec    |  1.137 |  1.137 |  1.137 |  1.137 |  0.000 |    0.0 |
| |4>>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | thread_cpu | sec    |  1.136 |  1.136 |  1.136 |  1.136 |  0.000 |  100.0 |
|--------------------------------------------------------------------------------------------------------------------------------------------------|

[sys]|0> Outputting 'timemory-ex-cxx-tuple-output/sys.flamegraph.json'...
[sys]|0> Outputting 'timemory-ex-cxx-tuple-output/sys.json'...
[sys]|0> Outputting 'timemory-ex-cxx-tuple-output/sys.txt'...
Opening 'timemory-ex-cxx-tuple-output/sys.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/sys.jpeg'...

|--------------------------------------------------------------------------------------------------------------------------------------------|
|                                                        CPU TIME SPENT IN KERNEL-MODE                                                       |
|--------------------------------------------------------------------------------------------------------------------------------------------|
|                      LABEL                       | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|--------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> test_1_usage@ex_cxx_tuple.cpp:192            |      1 |      0 | sys    | sec    |  0.010 |  0.010 |  0.010 |  0.010 |  0.000 |  100.0 |
| >>> |_time_fibonacci@ex_cxx_tuple.cpp:69         |      1 |      1 | sys    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>> test_2_timing_runtime                        |      1 |      0 | sys    | sec    |  0.010 |  0.010 |  0.010 |  0.010 |  0.000 |    0.0 |
| >>> |_test_2_timing@ex_cxx_tuple.cpp:228         |      1 |      1 | sys    | sec    |  0.010 |  0.010 |  0.010 |  0.010 |  0.000 |  100.0 |
| >>>   |_run_fib/40                               |      1 |      2 | sys    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>>     |_time_fibonacci@ex_cxx_tuple.cpp:69     |      1 |      3 | sys    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>>       |_run_fib/35                           |      7 |      4 | sys    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      7 |      5 | sys    | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>>       |_run_fib/43                           |      1 |      4 | sys    | sec    |  0.010 |  0.010 |  0.010 |  0.010 |  0.000 |    0.0 |
| >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | sys    | sec    |  0.010 |  0.010 |  0.010 |  0.010 |  0.000 |  100.0 |
|--------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-cxx-tuple-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-cxx-tuple-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-cxx-tuple-output/wall.txt'...
Opening 'timemory-ex-cxx-tuple-output/wall.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/wall.jpeg'...

|--------------------------------------------------------------------------------------------------------------------------------------------|
|                                                  REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------|
|                      LABEL                       | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|--------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> test_1_usage@ex_cxx_tuple.cpp:192            |      1 |      0 | wall   | sec    |  0.018 |  0.018 |  0.018 |  0.018 |  0.000 |   86.3 |
| >>> |_time_fibonacci@ex_cxx_tuple.cpp:69         |      1 |      1 | wall   | sec    |  0.002 |  0.002 |  0.002 |  0.002 |  0.000 |  100.0 |
| >>> test_2_timing_runtime                        |      1 |      0 | wall   | sec    |  1.139 |  1.139 |  1.139 |  1.139 |  0.000 |    0.0 |
| >>> |_test_2_timing@ex_cxx_tuple.cpp:228         |      1 |      1 | wall   | sec    |  1.139 |  1.139 |  1.139 |  1.139 |  0.000 |   74.4 |
| >>>   |_run_fib/40                               |      1 |      2 | wall   | sec    |  0.292 |  0.292 |  0.292 |  0.292 |  0.000 |    0.0 |
| >>>     |_time_fibonacci@ex_cxx_tuple.cpp:69     |      1 |      3 | wall   | sec    |  0.292 |  0.292 |  0.292 |  0.292 |  0.000 |    0.0 |
| >>>       |_run_fib/35                           |      7 |      4 | wall   | sec    |  0.309 |  0.044 |  0.034 |  0.056 |  0.009 |    0.3 |
| >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      7 |      5 | wall   | sec    |  0.308 |  0.044 |  0.034 |  0.056 |  0.009 |  100.0 |
| >>>       |_run_fib/43                           |      1 |      4 | wall   | sec    |  1.137 |  1.137 |  1.137 |  1.137 |  0.000 |    0.0 |
| >>>         |_time_fibonacci@ex_cxx_tuple.cpp:69 |      1 |      5 | wall   | sec    |  1.137 |  1.137 |  1.137 |  1.137 |  0.000 |  100.0 |
|--------------------------------------------------------------------------------------------------------------------------------------------|

[vol_cxt_swch]|0> Outputting 'timemory-ex-cxx-tuple-output/vol_cxt_swch.json'...
[vol_cxt_swch]|0> Outputting 'timemory-ex-cxx-tuple-output/vol_cxt_swch.txt'...
Opening 'timemory-ex-cxx-tuple-output/vol_cxt_swch.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/vol_cxt_swch.jpeg'...

|----------------------------------------------------------------------------------------------------------|
|NUMBER OF CONTEXT SWITCHES DUE TO A PROCESS VOLUNTARILY GIVING UP THE PROCESSOR BEFORE ITS TIME SLICE WAS COMPLETED|
|----------------------------------------------------------------------------------------------------------|
|         LABEL          | COUNT | DEPTH |    METRIC    | UNITS | SUM | MEAN | MIN | MAX | STDDEV | % SELF |
|------------------------|-------|-------|--------------|-------|-----|------|-----|-----|--------|--------|
| >>> test_1_usage_delta |     1 |     0 | vol_cxt_swch |       |   0 |    0 |   0 |   0 |      0 |      0 |
|----------------------------------------------------------------------------------------------------------|

[virtual_memory]|0> Outputting 'timemory-ex-cxx-tuple-output/virtual_memory.json'...
[virtual_memory]|0> Outputting 'timemory-ex-cxx-tuple-output/virtual_memory.txt'...
Opening 'timemory-ex-cxx-tuple-output/virtual_memory.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/virtual_memory.jpeg'...

|--------------------------------------------------------------------------------------------------------------------------|
|                                           RECORDS THE CHANGE IN VIRTUAL MEMORY                                           |
|--------------------------------------------------------------------------------------------------------------------------|
|         LABEL          | COUNT  | DEPTH  |     METRIC     | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|------------------------|--------|--------|----------------|--------|--------|--------|--------|--------|--------|--------|
| >>> test_1_usage_delta |      1 |      0 | virtual_memory | MB     | 40.002 | 40.002 | 40.002 | 40.002 |  0.000 |  100.0 |
|--------------------------------------------------------------------------------------------------------------------------|

[prio_cxt_swch]|0> Outputting 'timemory-ex-cxx-tuple-output/prio_cxt_swch.json'...
[prio_cxt_swch]|0> Outputting 'timemory-ex-cxx-tuple-output/prio_cxt_swch.txt'...
Opening 'timemory-ex-cxx-tuple-output/prio_cxt_swch.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/prio_cxt_swch.jpeg'...

|-----------------------------------------------------------------------------------------------------------|
|NUMBER OF CONTEXT SWITCH DUE TO HIGHER PRIORITY PROCESS BECOMING RUNNABLE OR BECAUSE THE CURRENT PROCESS EXCEEDED ITS TIME SLICE)|
|-----------------------------------------------------------------------------------------------------------|
|         LABEL          | COUNT | DEPTH |    METRIC     | UNITS | SUM | MEAN | MIN | MAX | STDDEV | % SELF |
|------------------------|-------|-------|---------------|-------|-----|------|-----|-----|--------|--------|
| >>> test_1_usage_delta |     1 |     0 | prio_cxt_swch |       |   0 |    0 |   0 |   0 |      0 |      0 |
|-----------------------------------------------------------------------------------------------------------|

[peak_rss]|0> Outputting 'timemory-ex-cxx-tuple-output/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-ex-cxx-tuple-output/peak_rss.txt'...
Opening 'timemory-ex-cxx-tuple-output/peak_rss.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/peak_rss.jpeg'...

|---------------------------------------------------------------------------------------------------------------------------------------------|
|              MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF SWAP IS ENABLED            |
|---------------------------------------------------------------------------------------------------------------------------------------------|
|                     LABEL                       | COUNT  | DEPTH  | METRIC   | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|-------------------------------------------------|--------|--------|----------|--------|--------|--------|--------|--------|--------|--------|
| >>> test_1_usage_delta                          |      1 |      0 | peak_rss | MB     | 39.248 | 39.248 | 39.248 | 39.248 |  0.000 |  100.0 |
| >>> test_3_measure/[init]                       |      1 |      0 | peak_rss | MB     |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>> test_3_measure@ex_cxx_tuple.cpp:274/[delta] |      1 |      0 | peak_rss | MB     | 40.464 | 40.464 | 40.464 | 40.464 |  0.000 |  100.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------|

[page_rss]|0> Outputting 'timemory-ex-cxx-tuple-output/page_rss.json'...
[page_rss]|0> Outputting 'timemory-ex-cxx-tuple-output/page_rss.txt'...
Opening 'timemory-ex-cxx-tuple-output/page_rss.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/page_rss.jpeg'...

|---------------------------------------------------------------------------------------------------------------------------------------------|
|               AMOUNT OF MEMORY ALLOCATED IN PAGES OF MEMORY. UNLIKE PEAK_RSS, VALUE WILL FLUCTUATE AS MEMORY IS FREED/ALLOCATED             |
|---------------------------------------------------------------------------------------------------------------------------------------------|
|                     LABEL                       | COUNT  | DEPTH  | METRIC   | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|-------------------------------------------------|--------|--------|----------|--------|--------|--------|--------|--------|--------|--------|
| >>> test_1_usage_delta                          |      1 |      0 | page_rss | MB     | 40.190 | 40.190 | 40.190 | 40.190 |  0.000 |  100.0 |
| >>> test_3_measure/[init]                       |      1 |      0 | page_rss | MB     |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |    0.0 |
| >>> test_3_measure@ex_cxx_tuple.cpp:274/[delta] |      1 |      0 | page_rss | MB     |  0.004 |  0.004 |  0.004 |  0.004 |  0.000 |  100.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------|

[minor_page_flts]|0> Outputting 'timemory-ex-cxx-tuple-output/minor_page_flts.json'...
[minor_page_flts]|0> Outputting 'timemory-ex-cxx-tuple-output/minor_page_flts.txt'...
Opening 'timemory-ex-cxx-tuple-output/minor_page_flts.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/minor_page_flts.jpeg'...

|----------------------------------------------------------------------------------------------------------------|
|NUMBER OF PAGE FAULTS SERVICED WITHOUT ANY I/O ACTIVITY VIA 'RECLAIMING' A PAGE FRAME FROM THE LIST OF PAGES AWAITING REALLOCATION|
|----------------------------------------------------------------------------------------------------------------|
|         LABEL          | COUNT | DEPTH |     METRIC      | UNITS | SUM  | MEAN | MIN  | MAX  | STDDEV | % SELF |
|------------------------|-------|-------|-----------------|-------|------|------|------|------|--------|--------|
| >>> test_1_usage_delta |     1 |     0 | minor_page_flts |       | 9769 | 9769 | 9769 | 9769 |      0 |    100 |
|----------------------------------------------------------------------------------------------------------------|

[major_page_flts]|0> Outputting 'timemory-ex-cxx-tuple-output/major_page_flts.json'...
[major_page_flts]|0> Outputting 'timemory-ex-cxx-tuple-output/major_page_flts.txt'...
Opening 'timemory-ex-cxx-tuple-output/major_page_flts.jpeg' for output...
Closed 'timemory-ex-cxx-tuple-output/major_page_flts.jpeg'...

|-------------------------------------------------------------------------------------------------------------|
|                           NUMBER OF PAGE FAULTS SERVICED THAT REQUIRED I/O ACTIVITY                         |
|-------------------------------------------------------------------------------------------------------------|
|         LABEL          | COUNT | DEPTH |     METRIC      | UNITS | SUM | MEAN | MIN | MAX | STDDEV | % SELF |
|------------------------|-------|-------|-----------------|-------|-----|------|-----|-----|--------|--------|
| >>> test_1_usage_delta |     1 |     0 | major_page_flts |       |   0 |    0 |   0 |   0 |      0 |      0 |
|-------------------------------------------------------------------------------------------------------------|


[metadata::manager::finalize]> Outputting 'timemory-ex-cxx-tuple-output/metadata.json'...


#---------------------- tim::manager destroyed [rank=0][id=0][pid=12303] ----------------------#
```