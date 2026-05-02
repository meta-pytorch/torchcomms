# ex-minimal

These examples demonstrate the usage of timemory markers and timemory library for timing measurements. Finally, the ex-minimal-library_overload example demonstrates overloading timemory library record functions with user defined record functions for measurements. The first example is available in C, C++ and Python whereas the last two examples are available in C and C++.

## Build

See [examples](../README.md##Build). These examples build the following corresponding binaries: `ex_c_minimal`, `ex_cxx_minimal`, `ex_c_minimal_library`, `ex_cxx_minimal_library`, `ex_cxx_minimal_library_overload`, `ex_c_minimal_library_overload`, `ex_python_minimal`.

## Expected Output
```bash
$ ./ex_cxx_minimal
#------------------------- tim::manager initialized [id=0][pid=11923] -------------------------#
Answer = 267914296
>>>  nested/0 :    0.420 sec wall,    0.420 sec cpu [laps: 1]
>>>  main@ex_minimal.cpp:63/total :    0.699 sec wall,    0.690 sec cpu [laps: 1]
Answer = 331160282
>>>  nested/1                     :    0.582 sec wall,    0.590 sec cpu [laps: 1]
>>>  main@ex_minimal.cpp:63/total :    0.841 sec wall,    0.850 sec cpu [laps: 1]
Answer = 433494437
Answer = 267914296
Answer = 331160282
Answer = 433494437
Answer = 267914296
Answer = 331160282
Answer = 433494437
Answer = 267914296
[cpu]|0> Outputting 'timemory-ex-cxx-minimal-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-ex-cxx-minimal-output/cpu.json'...
[cpu]|0> Outputting 'timemory-ex-cxx-minimal-output/cpu.txt'...
Opening 'timemory-ex-cxx-minimal-output/cpu.jpeg' for output...
Closed 'timemory-ex-cxx-minimal-output/cpu.jpeg'...

|----------------------------------------------------------------------------------------------------------------------------|
|                                     TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                                     |
|----------------------------------------------------------------------------------------------------------------------------|
|              LABEL               | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|----------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> main@ex_minimal.cpp:63/total |     10 |      0 | cpu    | sec    |  8.570 |  0.857 |  0.680 |  1.110 |  0.184 |   30.5 |
| >>> |_nested/0                   |      2 |      1 | cpu    | sec    |  1.260 |  0.630 |  0.420 |  0.840 |  0.297 |   87.3 |
| >>>   |_main/occasional/5        |      1 |      2 | cpu    | sec    |  0.160 |  0.160 |  0.160 |  0.160 |  0.000 |  100.0 |
| >>> |_nested/1                   |      2 |      1 | cpu    | sec    |  1.010 |  0.505 |  0.420 |  0.590 |  0.120 |  100.0 |
| >>> |_nested/2                   |      2 |      1 | cpu    | sec    |  1.420 |  0.710 |  0.580 |  0.840 |  0.184 |   88.7 |
| >>>   |_main/occasional/2        |      1 |      2 | cpu    | sec    |  0.160 |  0.160 |  0.160 |  0.160 |  0.000 |  100.0 |
| >>> |_nested/3                   |      2 |      1 | cpu    | sec    |  1.270 |  0.635 |  0.420 |  0.850 |  0.304 |   86.6 |
| >>>   |_main/occasional/8        |      1 |      2 | cpu    | sec    |  0.170 |  0.170 |  0.170 |  0.170 |  0.000 |  100.0 |
| >>> |_nested/4                   |      2 |      1 | cpu    | sec    |  1.000 |  0.500 |  0.420 |  0.580 |  0.113 |  100.0 |
|----------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-cxx-minimal-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-cxx-minimal-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-cxx-minimal-output/wall.txt'...
Opening 'timemory-ex-cxx-minimal-output/wall.jpeg' for output...
Closed 'timemory-ex-cxx-minimal-output/wall.jpeg'...

|----------------------------------------------------------------------------------------------------------------------------|
|                                          REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                          |
|----------------------------------------------------------------------------------------------------------------------------|
|              LABEL               | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|----------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> main@ex_minimal.cpp:63/total |     10 |      0 | wall   | sec    |  8.571 |  0.857 |  0.680 |  1.103 |  0.182 |   30.5 |
| >>> |_nested/0                   |      2 |      1 | wall   | sec    |  1.263 |  0.631 |  0.420 |  0.843 |  0.299 |   87.3 |
| >>>   |_main/occasional/5        |      1 |      2 | wall   | sec    |  0.161 |  0.161 |  0.161 |  0.161 |  0.000 |  100.0 |
| >>> |_nested/1                   |      2 |      1 | wall   | sec    |  1.003 |  0.501 |  0.421 |  0.582 |  0.114 |  100.0 |
| >>> |_nested/2                   |      2 |      1 | wall   | sec    |  1.423 |  0.711 |  0.582 |  0.841 |  0.183 |   88.7 |
| >>>   |_main/occasional/2        |      1 |      2 | wall   | sec    |  0.160 |  0.160 |  0.160 |  0.160 |  0.000 |  100.0 |
| >>> |_nested/3                   |      2 |      1 | wall   | sec    |  1.263 |  0.631 |  0.420 |  0.842 |  0.298 |   87.2 |
| >>>   |_main/occasional/8        |      1 |      2 | wall   | sec    |  0.161 |  0.161 |  0.161 |  0.161 |  0.000 |  100.0 |
| >>> |_nested/4                   |      2 |      1 | wall   | sec    |  1.003 |  0.501 |  0.421 |  0.582 |  0.114 |  100.0 |
|----------------------------------------------------------------------------------------------------------------------------|



#---------------------- tim::manager destroyed [rank=0][id=0][pid=11923] ----------------------#
```