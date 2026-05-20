# ex-optional

This examples demonstrates enabling and disabling timemory instrumentation in either normal of MPI or convnetional setting. This example code builds 4 examples namely `ex_optional_off, ex_optional_off.mpi, ex_optional_on, ex_optional_on.mpi`.

## Build

See [examples](../README.md##Build).

## Expected Output

### timemory-on

```bash
$ ./ex_optional_on
#------------------------- tim::manager initialized [id=0][pid=8036] -------------------------#


#----------------- TIMEMORY is enabled  ----------------#

fibonacci(1) = 1
fibonacci(2) = 1
fibonacci(3) = 2
fibonacci(4) = 3
fibonacci(17) = 1597
fibonacci(18) = 2584
fibonacci(19) = 4181
fibonacci(20) = 6765
fibonacci(5) = 5
fibonacci(6) = 8
fibonacci(13) = 233
fibonacci(7) = 13
fibonacci(14) = 377
fibonacci(8) = 21
fibonacci(15) = 610
fibonacci(16) = 987
fibonacci(21) = 10946
fibonacci(22) = 17711
fibonacci(23) = 28657
fibonacci(24) = 46368
fibonacci(9) = 34
fibonacci(10) = 55
fibonacci(11) = 89
fibonacci(29) = 514229
fibonacci(12) = 144
fibonacci(25) = 75025
fibonacci(26) = 121393
fibonacci(27) = 196418
fibonacci(28) = 317811
fibonacci(30) = 832040
fibonacci(31) = 1346269
fibonacci(33) = 3524578
fibonacci(32) = 2178309
fibonacci(34) = 5702887
fibonacci(36) = 14930352
fibonacci(35) = 9227465
fibonacci(37) = 24157817
fibonacci(39) = 63245986
fibonacci(38) = 39088169
fibonacci(40) = 102334155
fibonacci(42) = 267914296
fibonacci(41) = 165580141
fibonacci(43) = 433494437
fibonacci(44) = 701408733
fibonacci(11) = 89
fibonacci(12) = 144
fibonacci(13) = 233
fibonacci(14) = 377
fibonacci(15) = 610
fibonacci(16) = 987
fibonacci(17) = 1597
fibonacci(18) = 2584
fibonacci(19) = 4181
fibonacci(20) = 6765
fibonacci(21) = 10946
fibonacci(22) = 17711
fibonacci(23) = 28657
fibonacci(24) = 46368
fibonacci(25) = 75025
fibonacci(26) = 121393
fibonacci(27) = 196418
fibonacci(28) = 317811
fibonacci(29) = 514229
fibonacci(10) = 55
fibonacci(11) = 89
fibonacci(12) = 144
fibonacci(13) = 233
fibonacci(14) = 377
fibonacci(15) = 610
fibonacci(16) = 987
fibonacci(17) = 1597
fibonacci(18) = 2584
fibonacci(19) = 4181
fibonacci(20) = 6765
fibonacci(21) = 10946
fibonacci(22) = 17711
fibonacci(23) = 28657
fibonacci(24) = 46368
fibonacci(25) = 75025
fibonacci(26) = 121393
fibonacci(27) = 196418
fibonacci(28) = 317811
fibonacci(29) = 514229
fibonacci(10) = 55
fibonacci(11) = 89
fibonacci(12) = 144
fibonacci(13) = 233
fibonacci(14) = 377
Master rank: 0, Number of elements per process: 201
Avg of all elements is 0.51925808, Avg computed across original data is 0.51925808
Master rank: 0, Number of elements per process: 201
Avg of all elements is 0.51925808, Avg computed across original data is 0.51925808
Master rank: 0, Number of elements per process: 202
Avg of all elements is 0.52161908, Avg computed across original data is 0.52161908
Master rank: 0, Number of elements per process: 203
Avg of all elements is 0.52071512, Avg computed across original data is 0.52071512
Master rank: 0, Number of elements per process: 205
Avg of all elements is 0.52025747, Avg computed across original data is 0.52025747
Master rank: 0, Number of elements per process: 208
Avg of all elements is 0.51978099, Avg computed across original data is 0.51978099
Master rank: 0, Number of elements per process: 213
Avg of all elements is 0.51948100, Avg computed across original data is 0.51948100
Master rank: 0, Number of elements per process: 221
Avg of all elements is 0.51865053, Avg computed across original data is 0.51865053
Master rank: 0, Number of elements per process: 234
Avg of all elements is 0.51232868, Avg computed across original data is 0.51232868
Master rank: 0, Number of elements per process: 255
Avg of all elements is 0.50691003, Avg computed across original data is 0.50691003
Master rank: 0, Number of elements per process: 289
Avg of all elements is 0.50956196, Avg computed across original data is 0.50956196
Master rank: 0, Number of elements per process: 344
Avg of all elements is 0.50841320, Avg computed across original data is 0.50841320
Master rank: 0, Number of elements per process: 433
Avg of all elements is 0.51366031, Avg computed across original data is 0.51366031
Master rank: 0, Number of elements per process: 577
Avg of all elements is 0.50415176, Avg computed across original data is 0.50415176
Master rank: 0, Number of elements per process: 810
Avg of all elements is 0.49682051, Avg computed across original data is 0.49682051
Master rank: 0, Number of elements per process: 1187
Avg of all elements is 0.49292898, Avg computed across original data is 0.49292898
Master rank: 0, Number of elements per process: 1797
Avg of all elements is 0.49064901, Avg computed across original data is 0.49064901
Master rank: 0, Number of elements per process: 2784
Avg of all elements is 0.49006906, Avg computed across original data is 0.49006906
Master rank: 0, Number of elements per process: 4381
Avg of all elements is 0.49453461, Avg computed across original data is 0.49453461
Master rank: 0, Number of elements per process: 6965
Avg of all elements is 0.49365306, Avg computed across original data is 0.49365306
Master rank: 0, Number of elements per process: 11146
Avg of all elements is 0.49324873, Avg computed across original data is 0.49324873
Master rank: 0, Number of elements per process: 17911
Avg of all elements is 0.49661958, Avg computed across original data is 0.49661958
Master rank: 0, Number of elements per process: 28857
Avg of all elements is 0.49844140, Avg computed across original data is 0.49844140
Master rank: 0, Number of elements per process: 46568
Avg of all elements is 0.49770710, Avg computed across original data is 0.49770710
Master rank: 0, Number of elements per process: 200
Avg of all elements is 0.51836574, Avg computed across original data is 0.51836574
Master rank: 0, Number of elements per process: 46568
Avg of all elements is 0.49770710, Avg computed across original data is 0.49770710
Master rank: 0, Number of elements per process: 46568
Avg of all elements is 0.49770710, Avg computed across original data is 0.49770710
Master rank: 0, Number of elements per process: 17911
Avg of all elements is 0.49661958, Avg computed across original data is 0.49661958
Master rank: 0, Number of elements per process: 64279
Avg of all elements is 0.49791399, Avg computed across original data is 0.49791399
Master rank: 0, Number of elements per process: 6965
Avg of all elements is 0.49365306, Avg computed across original data is 0.49365306
Master rank: 0, Number of elements per process: 71044
Avg of all elements is 0.49812654, Avg computed across original data is 0.49812654
Master rank: 0, Number of elements per process: 2784
Avg of all elements is 0.49006906, Avg computed across original data is 0.49006906
Master rank: 0, Number of elements per process: 73628
Avg of all elements is 0.49782935, Avg computed across original data is 0.49782935
Master rank: 0, Number of elements per process: 1187
Avg of all elements is 0.49292898, Avg computed across original data is 0.49292898
Master rank: 0, Number of elements per process: 74615
Avg of all elements is 0.49795702, Avg computed across original data is 0.49795702
Master rank: 0, Number of elements per process: 577
Avg of all elements is 0.50415176, Avg computed across original data is 0.50415176
Master rank: 0, Number of elements per process: 74992
Avg of all elements is 0.49798629, Avg computed across original data is 0.49798629
Master rank: 0, Number of elements per process: 344
Avg of all elements is 0.50841320, Avg computed across original data is 0.50841320
Master rank: 0, Number of elements per process: 75136
Avg of all elements is 0.49797767, Avg computed across original data is 0.49797767
Master rank: 0, Number of elements per process: 255
Avg of all elements is 0.46220887, Avg computed across original data is 0.46220887
Master rank: 0, Number of elements per process: 75191
Avg of all elements is 0.50133753, Avg computed across original data is 0.50133753
Master rank: 0, Number of elements per process: 221
Avg of all elements is 0.49239236, Avg computed across original data is 0.49239236
Master rank: 0, Number of elements per process: 75212
Avg of all elements is 0.50014150, Avg computed across original data is 0.50014150
Master rank: 0, Number of elements per process: 208
Avg of all elements is 0.51172578, Avg computed across original data is 0.51172578

#----------------- TIMEMORY is enabled  ----------------#

[cpu_util]|0> Outputting 'timemory-ex-optional-on-output/cpu_util.json'...
[cpu_util]|0> Outputting 'timemory-ex-optional-on-output/cpu_util.txt'...
Opening 'timemory-ex-optional-on-output/cpu_util.jpeg' for output...
Closed 'timemory-ex-optional-on-output/cpu_util.jpeg'...

|-----------------------------------------------------------------------------------------------------------------------|
|                                 PERCENTAGE OF CPU-CLOCK TIME DIVIDED BY WALL-CLOCK TIME                               |
|-----------------------------------------------------------------------------------------------------------------------|
|             LABEL              | COUNT | DEPTH | METRIC   | UNITS |  SUM   | MEAN  | MIN   |  MAX   | STDDEV | % SELF |
|--------------------------------|-------|-------|----------|-------|--------|-------|-------|--------|--------|--------|
| >>> main@ex_optional.cpp:105   |     1 |     0 | cpu_util | %     |  102.9 | 102.9 | 102.9 |  102.9 |    0.0 |    0.0 |
| >>> |_main@ex_optional.cpp:106 |     1 |     1 | cpu_util | %     |  115.3 | 115.3 | 115.3 |  115.3 |    0.0 |    0.0 |
| >>>   |_fibonacci/(1)          |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(2)          |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>     |_fibonacci/(29)       |     1 |     3 | cpu_util | %     |  667.5 | 667.5 | 667.5 |  667.5 |    0.0 |  100.0 |
| >>>   |_fibonacci/(3)          |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(4)          |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(11)         |     5 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(12)         |     5 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(13)         |     5 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
|--------------------------------|-------|-------|----------|-------|--------|-------|-------|--------|--------|--------|
| >>>   |_fibonacci/(14)         |     5 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(15)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(16)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(17)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(18)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(19)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(20)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(21)         |     4 |     2 | cpu_util | %     | 2022.1 | 505.5 |   0.0 | 5429.4 | 2714.7 |  100.0 |
| >>>   |_fibonacci/(22)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(23)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
|--------------------------------|-------|-------|----------|-------|--------|-------|-------|--------|--------|--------|
| >>>   |_fibonacci/(24)         |     4 |     2 | cpu_util | %     | 1191.9 | 298.0 |   0.0 | 2788.4 | 1394.2 |  100.0 |
| >>>   |_fibonacci/(25)         |     5 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(26)         |     4 |     2 | cpu_util | %     |  777.1 | 194.3 |   0.0 | 1617.3 |  808.7 |  100.0 |
| >>>   |_fibonacci/(27)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(28)         |     4 |     2 | cpu_util | %     |  178.1 |  44.5 |   0.0 | 1046.0 |  523.0 |  100.0 |
| >>>   |_fibonacci/(29)         |     3 |     2 | cpu_util | %     |  134.5 |  44.8 |   0.0 |  229.5 |  132.5 |  100.0 |
| >>>   |_fibonacci/(10)         |     4 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_201    |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_202    |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_203    |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
|--------------------------------|-------|-------|----------|-------|--------|-------|-------|--------|--------|--------|
| >>>   |_fibonacci/(5)          |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_205    |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(6)          |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_208    |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(7)          |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_213    |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(8)          |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_221    |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(9)          |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_234    |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
|--------------------------------|-------|-------|----------|-------|--------|-------|-------|--------|--------|--------|
| >>>   |_scatter_gatther_255    |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_289    |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_344    |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_433    |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_577    |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_810    |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_1187   |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_1797   |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_2784   |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_4381   |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
|--------------------------------|-------|-------|----------|-------|--------|-------|-------|--------|--------|--------|
| >>>   |_scatter_gatther_6965   |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_11146  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_17911  |     2 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_28857  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_46568  |     3 |     2 | cpu_util | %     |  256.0 |  85.3 |   0.0 |  740.9 |  427.7 |  100.0 |
| >>>   |_scatter_gatther_200    |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_scatter_gatther_64279  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(30)         |     2 |     2 | cpu_util | %     |  517.2 | 258.6 | 403.5 |  570.8 |  118.3 |  100.0 |
| >>>   |_fibonacci/(31)         |     2 |     2 | cpu_util | %     |  388.8 | 194.4 |   0.0 |  637.6 |  450.8 |  100.0 |
| >>>   |_scatter_gatther_71044  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
|--------------------------------|-------|-------|----------|-------|--------|-------|-------|--------|--------|--------|
| >>>   |_fibonacci/(32)         |     2 |     2 | cpu_util | %     |  323.4 | 161.7 | 153.1 |  447.8 |  208.4 |  100.0 |
| >>>   |_fibonacci/(33)         |     2 |     2 | cpu_util | %     |  402.4 | 201.2 |  95.4 |  593.4 |  352.1 |  100.0 |
| >>>   |_scatter_gatther_73628  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(34)         |     2 |     2 | cpu_util | %     |  293.6 | 146.8 | 118.0 |  467.5 |  247.1 |  100.0 |
| >>>   |_fibonacci/(35)         |     2 |     2 | cpu_util | %     |  254.9 | 127.4 | 109.4 |  399.8 |  205.3 |  100.0 |
| >>>   |_scatter_gatther_74615  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(36)         |     2 |     2 | cpu_util | %     |  316.6 | 158.3 |  94.3 |  496.9 |  284.7 |  100.0 |
| >>>   |_fibonacci/(37)         |     2 |     2 | cpu_util | %     |  209.3 | 104.7 |  98.2 |  302.7 |  144.6 |  100.0 |
| >>>   |_scatter_gatther_74992  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(38)         |     2 |     2 | cpu_util | %     |  207.0 | 103.5 | 101.0 |  299.5 |  140.4 |  100.0 |
|--------------------------------|-------|-------|----------|-------|--------|-------|-------|--------|--------|--------|
| >>>   |_fibonacci/(39)         |     2 |     2 | cpu_util | %     |  237.3 | 118.6 | 100.3 |  363.1 |  185.8 |  100.0 |
| >>>   |_scatter_gatther_75136  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(40)         |     2 |     2 | cpu_util | %     |  164.2 |  82.1 | 100.5 |  227.7 |   89.9 |  100.0 |
| >>>   |_fibonacci/(41)         |     2 |     2 | cpu_util | %     |  150.4 |  75.2 | 100.3 |  200.5 |   70.8 |  100.0 |
| >>>   |_scatter_gatther_75191  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(42)         |     2 |     2 | cpu_util | %     |  176.7 |  88.3 | 100.4 |  250.0 |  105.7 |  100.0 |
| >>>   |_fibonacci/(43)         |     2 |     2 | cpu_util | %     |  107.0 |  53.5 | 100.4 |  113.6 |    9.4 |  100.0 |
| >>>   |_scatter_gatther_75212  |     1 |     2 | cpu_util | %     |    0.0 |   0.0 |   0.0 |    0.0 |    0.0 |    0.0 |
| >>>   |_fibonacci/(44)         |     2 |     2 | cpu_util | %     |   99.9 |  50.0 |  99.9 |   99.9 |    0.0 |  100.0 |
|-----------------------------------------------------------------------------------------------------------------------|

[cpu]|0> Outputting 'timemory-ex-optional-on-output/cpu.flamegraph.json'...
[cpu]|0> Outputting 'timemory-ex-optional-on-output/cpu.json'...
[cpu]|0> Outputting 'timemory-ex-optional-on-output/cpu.txt'...
Opening 'timemory-ex-optional-on-output/cpu.jpeg' for output...
Closed 'timemory-ex-optional-on-output/cpu.jpeg'...

|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                  TOTAL CPU TIME SPENT IN BOTH USER- AND KERNEL-MODE                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|             LABEL              |    COUNT     |    DEPTH     |    METRIC    |    UNITS     |     SUM      |     MEAN     |     MIN      |     MAX      |    STDDEV    |    % SELF    |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>> main@ex_optional.cpp:105   |            1 |            0 | cpu          | sec          |     9.550000 |     9.550000 |     9.550000 |     9.550000 |     0.000000 |          0.0 |
| >>> |_main@ex_optional.cpp:106 |            1 |            1 | cpu          | sec          |     9.550000 |     9.550000 |     9.550000 |     9.550000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(1)          |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(2)          |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>     |_fibonacci/(29)       |            1 |            3 | cpu          | sec          |     0.030000 |     0.030000 |     0.030000 |     0.030000 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(3)          |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(4)          |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(11)         |            5 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(12)         |            5 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(13)         |            5 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(14)         |            5 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(15)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(16)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(17)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(18)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(19)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(20)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(21)         |            4 |            2 | cpu          | sec          |     0.010000 |     0.002500 |     0.000000 |     0.010000 |     0.005000 |        100.0 |
| >>>   |_fibonacci/(22)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(23)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(24)         |            4 |            2 | cpu          | sec          |     0.020000 |     0.005000 |     0.000000 |     0.020000 |     0.010000 |        100.0 |
| >>>   |_fibonacci/(25)         |            5 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(26)         |            4 |            2 | cpu          | sec          |     0.020000 |     0.005000 |     0.000000 |     0.020000 |     0.010000 |        100.0 |
| >>>   |_fibonacci/(27)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(28)         |            4 |            2 | cpu          | sec          |     0.010000 |     0.002500 |     0.000000 |     0.010000 |     0.005000 |        100.0 |
| >>>   |_fibonacci/(29)         |            3 |            2 | cpu          | sec          |     0.010000 |     0.003333 |     0.000000 |     0.010000 |     0.005774 |        100.0 |
| >>>   |_fibonacci/(10)         |            4 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_201    |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_202    |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_203    |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(5)          |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_205    |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(6)          |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_208    |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(7)          |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_213    |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(8)          |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_221    |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(9)          |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_234    |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_scatter_gatther_255    |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_289    |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_344    |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_433    |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_577    |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_810    |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_1187   |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_1797   |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_2784   |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_4381   |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_scatter_gatther_6965   |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_11146  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_17911  |            2 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_28857  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_46568  |            3 |            2 | cpu          | sec          |     0.010000 |     0.003333 |     0.000000 |     0.010000 |     0.005774 |        100.0 |
| >>>   |_scatter_gatther_200    |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_scatter_gatther_64279  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(30)         |            2 |            2 | cpu          | sec          |     0.040000 |     0.020000 |     0.010000 |     0.030000 |     0.014142 |        100.0 |
| >>>   |_fibonacci/(31)         |            2 |            2 | cpu          | sec          |     0.040000 |     0.020000 |     0.000000 |     0.040000 |     0.028284 |        100.0 |
| >>>   |_scatter_gatther_71044  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(32)         |            2 |            2 | cpu          | sec          |     0.050000 |     0.025000 |     0.010000 |     0.040000 |     0.021213 |        100.0 |
| >>>   |_fibonacci/(33)         |            2 |            2 | cpu          | sec          |     0.110000 |     0.055000 |     0.010000 |     0.100000 |     0.063640 |        100.0 |
| >>>   |_scatter_gatther_73628  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(34)         |            2 |            2 | cpu          | sec          |     0.100000 |     0.050000 |     0.020000 |     0.080000 |     0.042426 |        100.0 |
| >>>   |_fibonacci/(35)         |            2 |            2 | cpu          | sec          |     0.140000 |     0.070000 |     0.030000 |     0.110000 |     0.056569 |        100.0 |
| >>>   |_scatter_gatther_74615  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(36)         |            2 |            2 | cpu          | sec          |     0.300000 |     0.150000 |     0.040000 |     0.260000 |     0.155563 |        100.0 |
| >>>   |_fibonacci/(37)         |            2 |            2 | cpu          | sec          |     0.280000 |     0.140000 |     0.060000 |     0.220000 |     0.113137 |        100.0 |
| >>>   |_scatter_gatther_74992  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(38)         |            2 |            2 | cpu          | sec          |     0.440000 |     0.220000 |     0.100000 |     0.340000 |     0.169706 |        100.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(39)         |            2 |            2 | cpu          | sec          |     0.790000 |     0.395000 |     0.160000 |     0.630000 |     0.332340 |        100.0 |
| >>>   |_scatter_gatther_75136  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(40)         |            2 |            2 | cpu          | sec          |     0.850000 |     0.425000 |     0.260000 |     0.590000 |     0.233345 |        100.0 |
| >>>   |_fibonacci/(41)         |            2 |            2 | cpu          | sec          |     1.260000 |     0.630000 |     0.420000 |     0.840000 |     0.296985 |        100.0 |
| >>>   |_scatter_gatther_75191  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(42)         |            2 |            2 | cpu          | sec          |     2.440000 |     1.220000 |     0.680000 |     1.760000 |     0.763675 |        100.0 |
| >>>   |_fibonacci/(43)         |            2 |            2 | cpu          | sec          |     2.350000 |     1.175000 |     1.100000 |     1.250000 |     0.106066 |        100.0 |
| >>>   |_scatter_gatther_75212  |            1 |            2 | cpu          | sec          |     0.000000 |     0.000000 |     0.000000 |     0.000000 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(44)         |            2 |            2 | cpu          | sec          |     3.540000 |     1.770000 |     1.770000 |     1.770000 |     0.000000 |        100.0 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[wall]|0> Outputting 'timemory-ex-optional-on-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-ex-optional-on-output/wall.json'...
[wall]|0> Outputting 'timemory-ex-optional-on-output/wall.txt'...
Opening 'timemory-ex-optional-on-output/wall.jpeg' for output...
Closed 'timemory-ex-optional-on-output/wall.jpeg'...

|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                       REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                       |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|             LABEL              |    COUNT     |    DEPTH     |    METRIC    |    UNITS     |     SUM      |     MEAN     |     MIN      |     MAX      |    STDDEV    |    % SELF    |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>> main@ex_optional.cpp:105   |            1 |            0 | wall         | sec          |     9.280417 |     9.280417 |     9.280417 |     9.280417 |     0.000000 |         10.8 |
| >>> |_main@ex_optional.cpp:106 |            1 |            1 | wall         | sec          |     8.280217 |     8.280217 |     8.280217 |     8.280217 |     0.000000 |          0.0 |
| >>>   |_fibonacci/(1)          |            2 |            2 | wall         | sec          |     0.000080 |     0.000040 |     0.000008 |     0.000072 |     0.000046 |        100.0 |
| >>>   |_fibonacci/(2)          |            2 |            2 | wall         | sec          |     0.000049 |     0.000025 |     0.000008 |     0.000041 |     0.000023 |          0.0 |
| >>>     |_fibonacci/(29)       |            1 |            3 | wall         | sec          |     0.004495 |     0.004495 |     0.004495 |     0.004495 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(3)          |            2 |            2 | wall         | sec          |     0.000039 |     0.000019 |     0.000008 |     0.000031 |     0.000016 |        100.0 |
| >>>   |_fibonacci/(4)          |            2 |            2 | wall         | sec          |     0.000041 |     0.000020 |     0.000008 |     0.000033 |     0.000017 |        100.0 |
| >>>   |_fibonacci/(11)         |            5 |            2 | wall         | sec          |     0.000144 |     0.000029 |     0.000008 |     0.000071 |     0.000025 |        100.0 |
| >>>   |_fibonacci/(12)         |            5 |            2 | wall         | sec          |     0.000137 |     0.000027 |     0.000009 |     0.000048 |     0.000018 |        100.0 |
| >>>   |_fibonacci/(13)         |            5 |            2 | wall         | sec          |     0.000156 |     0.000031 |     0.000009 |     0.000068 |     0.000024 |        100.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(14)         |            5 |            2 | wall         | sec          |     0.000135 |     0.000027 |     0.000009 |     0.000042 |     0.000016 |        100.0 |
| >>>   |_fibonacci/(15)         |            4 |            2 | wall         | sec          |     0.000146 |     0.000036 |     0.000011 |     0.000049 |     0.000017 |        100.0 |
| >>>   |_fibonacci/(16)         |            4 |            2 | wall         | sec          |     0.000141 |     0.000035 |     0.000011 |     0.000050 |     0.000017 |        100.0 |
| >>>   |_fibonacci/(17)         |            4 |            2 | wall         | sec          |     0.000198 |     0.000049 |     0.000013 |     0.000085 |     0.000030 |        100.0 |
| >>>   |_fibonacci/(18)         |            4 |            2 | wall         | sec          |     0.000215 |     0.000054 |     0.000016 |     0.000071 |     0.000026 |        100.0 |
| >>>   |_fibonacci/(19)         |            4 |            2 | wall         | sec          |     0.000209 |     0.000052 |     0.000021 |     0.000092 |     0.000036 |        100.0 |
| >>>   |_fibonacci/(20)         |            4 |            2 | wall         | sec          |     0.000285 |     0.000071 |     0.000028 |     0.000129 |     0.000049 |        100.0 |
| >>>   |_fibonacci/(21)         |            4 |            2 | wall         | sec          |     0.000495 |     0.000124 |     0.000041 |     0.000226 |     0.000095 |        100.0 |
| >>>   |_fibonacci/(22)         |            4 |            2 | wall         | sec          |     0.000682 |     0.000171 |     0.000061 |     0.000282 |     0.000125 |        100.0 |
| >>>   |_fibonacci/(23)         |            4 |            2 | wall         | sec          |     0.001041 |     0.000260 |     0.000093 |     0.000427 |     0.000191 |        100.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(24)         |            4 |            2 | wall         | sec          |     0.001678 |     0.000419 |     0.000146 |     0.000717 |     0.000315 |        100.0 |
| >>>   |_fibonacci/(25)         |            5 |            2 | wall         | sec          |     0.002277 |     0.000455 |     0.000231 |     0.001061 |     0.000361 |        100.0 |
| >>>   |_fibonacci/(26)         |            4 |            2 | wall         | sec          |     0.002574 |     0.000643 |     0.000368 |     0.001237 |     0.000410 |        100.0 |
| >>>   |_fibonacci/(27)         |            4 |            2 | wall         | sec          |     0.003822 |     0.000955 |     0.000592 |     0.001688 |     0.000499 |        100.0 |
| >>>   |_fibonacci/(28)         |            4 |            2 | wall         | sec          |     0.005615 |     0.001404 |     0.000950 |     0.002757 |     0.000902 |        100.0 |
| >>>   |_fibonacci/(29)         |            3 |            2 | wall         | sec          |     0.007432 |     0.002477 |     0.001538 |     0.004356 |     0.001627 |        100.0 |
| >>>   |_fibonacci/(10)         |            4 |            2 | wall         | sec          |     0.000076 |     0.000019 |     0.000009 |     0.000030 |     0.000011 |        100.0 |
| >>>   |_scatter_gatther_201    |            2 |            2 | wall         | sec          |     0.000060 |     0.000030 |     0.000022 |     0.000038 |     0.000011 |        100.0 |
| >>>   |_scatter_gatther_202    |            1 |            2 | wall         | sec          |     0.000025 |     0.000025 |     0.000025 |     0.000025 |     0.000000 |        100.0 |
| >>>   |_scatter_gatther_203    |            1 |            2 | wall         | sec          |     0.000023 |     0.000023 |     0.000023 |     0.000023 |     0.000000 |        100.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(5)          |            2 |            2 | wall         | sec          |     0.000086 |     0.000043 |     0.000009 |     0.000077 |     0.000048 |        100.0 |
| >>>   |_scatter_gatther_205    |            1 |            2 | wall         | sec          |     0.000024 |     0.000024 |     0.000024 |     0.000024 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(6)          |            2 |            2 | wall         | sec          |     0.000059 |     0.000030 |     0.000010 |     0.000050 |     0.000028 |        100.0 |
| >>>   |_scatter_gatther_208    |            2 |            2 | wall         | sec          |     0.000046 |     0.000023 |     0.000022 |     0.000024 |     0.000002 |        100.0 |
| >>>   |_fibonacci/(7)          |            2 |            2 | wall         | sec          |     0.000062 |     0.000031 |     0.000009 |     0.000052 |     0.000030 |        100.0 |
| >>>   |_scatter_gatther_213    |            1 |            2 | wall         | sec          |     0.000025 |     0.000025 |     0.000025 |     0.000025 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(8)          |            2 |            2 | wall         | sec          |     0.000061 |     0.000031 |     0.000010 |     0.000052 |     0.000030 |        100.0 |
| >>>   |_scatter_gatther_221    |            2 |            2 | wall         | sec          |     0.000046 |     0.000023 |     0.000023 |     0.000024 |     0.000001 |        100.0 |
| >>>   |_fibonacci/(9)          |            2 |            2 | wall         | sec          |     0.000061 |     0.000031 |     0.000010 |     0.000051 |     0.000030 |        100.0 |
| >>>   |_scatter_gatther_234    |            1 |            2 | wall         | sec          |     0.000025 |     0.000025 |     0.000025 |     0.000025 |     0.000000 |        100.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_scatter_gatther_255    |            2 |            2 | wall         | sec          |     0.000051 |     0.000025 |     0.000024 |     0.000027 |     0.000002 |        100.0 |
| >>>   |_scatter_gatther_289    |            1 |            2 | wall         | sec          |     0.000028 |     0.000028 |     0.000028 |     0.000028 |     0.000000 |        100.0 |
| >>>   |_scatter_gatther_344    |            2 |            2 | wall         | sec          |     0.000055 |     0.000028 |     0.000027 |     0.000028 |     0.000001 |        100.0 |
| >>>   |_scatter_gatther_433    |            1 |            2 | wall         | sec          |     0.000028 |     0.000028 |     0.000028 |     0.000028 |     0.000000 |        100.0 |
| >>>   |_scatter_gatther_577    |            2 |            2 | wall         | sec          |     0.000061 |     0.000031 |     0.000029 |     0.000032 |     0.000002 |        100.0 |
| >>>   |_scatter_gatther_810    |            1 |            2 | wall         | sec          |     0.000042 |     0.000042 |     0.000042 |     0.000042 |     0.000000 |        100.0 |
| >>>   |_scatter_gatther_1187   |            2 |            2 | wall         | sec          |     0.000094 |     0.000047 |     0.000046 |     0.000048 |     0.000001 |        100.0 |
| >>>   |_scatter_gatther_1797   |            1 |            2 | wall         | sec          |     0.000065 |     0.000065 |     0.000065 |     0.000065 |     0.000000 |        100.0 |
| >>>   |_scatter_gatther_2784   |            2 |            2 | wall         | sec          |     0.000167 |     0.000084 |     0.000082 |     0.000086 |     0.000003 |        100.0 |
| >>>   |_scatter_gatther_4381   |            1 |            2 | wall         | sec          |     0.000127 |     0.000127 |     0.000127 |     0.000127 |     0.000000 |        100.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_scatter_gatther_6965   |            2 |            2 | wall         | sec          |     0.000366 |     0.000183 |     0.000177 |     0.000189 |     0.000008 |        100.0 |
| >>>   |_scatter_gatther_11146  |            1 |            2 | wall         | sec          |     0.000291 |     0.000291 |     0.000291 |     0.000291 |     0.000000 |        100.0 |
| >>>   |_scatter_gatther_17911  |            2 |            2 | wall         | sec          |     0.000945 |     0.000472 |     0.000470 |     0.000475 |     0.000004 |        100.0 |
| >>>   |_scatter_gatther_28857  |            1 |            2 | wall         | sec          |     0.000865 |     0.000865 |     0.000865 |     0.000865 |     0.000000 |        100.0 |
| >>>   |_scatter_gatther_46568  |            3 |            2 | wall         | sec          |     0.003907 |     0.001302 |     0.001274 |     0.001350 |     0.000041 |        100.0 |
| >>>   |_scatter_gatther_200    |            1 |            2 | wall         | sec          |     0.000023 |     0.000023 |     0.000023 |     0.000023 |     0.000000 |        100.0 |
| >>>   |_scatter_gatther_64279  |            1 |            2 | wall         | sec          |     0.001833 |     0.001833 |     0.001833 |     0.001833 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(30)         |            2 |            2 | wall         | sec          |     0.007734 |     0.003867 |     0.002478 |     0.005256 |     0.001964 |        100.0 |
| >>>   |_fibonacci/(31)         |            2 |            2 | wall         | sec          |     0.010288 |     0.005144 |     0.004014 |     0.006274 |     0.001598 |        100.0 |
| >>>   |_scatter_gatther_71044  |            1 |            2 | wall         | sec          |     0.001995 |     0.001995 |     0.001995 |     0.001995 |     0.000000 |        100.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(32)         |            2 |            2 | wall         | sec          |     0.015462 |     0.007731 |     0.006530 |     0.008932 |     0.001698 |        100.0 |
| >>>   |_fibonacci/(33)         |            2 |            2 | wall         | sec          |     0.027333 |     0.013666 |     0.010480 |     0.016853 |     0.004507 |        100.0 |
| >>>   |_scatter_gatther_73628  |            1 |            2 | wall         | sec          |     0.002067 |     0.002067 |     0.002067 |     0.002067 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(34)         |            2 |            2 | wall         | sec          |     0.034056 |     0.017028 |     0.016944 |     0.017113 |     0.000120 |        100.0 |
| >>>   |_fibonacci/(35)         |            2 |            2 | wall         | sec          |     0.054927 |     0.027464 |     0.027415 |     0.027512 |     0.000068 |        100.0 |
| >>>   |_scatter_gatther_74615  |            1 |            2 | wall         | sec          |     0.002121 |     0.002121 |     0.002121 |     0.002121 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(36)         |            2 |            2 | wall         | sec          |     0.094761 |     0.047381 |     0.042435 |     0.052326 |     0.006993 |        100.0 |
| >>>   |_fibonacci/(37)         |            2 |            2 | wall         | sec          |     0.133759 |     0.066880 |     0.061079 |     0.072681 |     0.008204 |        100.0 |
| >>>   |_scatter_gatther_74992  |            1 |            2 | wall         | sec          |     0.001796 |     0.001796 |     0.001796 |     0.001796 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(38)         |            2 |            2 | wall         | sec          |     0.212570 |     0.106285 |     0.099054 |     0.113516 |     0.010226 |        100.0 |
|--------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| >>>   |_fibonacci/(39)         |            2 |            2 | wall         | sec          |     0.332961 |     0.166481 |     0.159448 |     0.173513 |     0.009945 |        100.0 |
| >>>   |_scatter_gatther_75136  |            1 |            2 | wall         | sec          |     0.001755 |     0.001755 |     0.001755 |     0.001755 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(40)         |            2 |            2 | wall         | sec          |     0.517729 |     0.258865 |     0.258632 |     0.259098 |     0.000330 |        100.0 |
| >>>   |_fibonacci/(41)         |            2 |            2 | wall         | sec          |     0.837510 |     0.418755 |     0.418572 |     0.418938 |     0.000259 |        100.0 |
| >>>   |_scatter_gatther_75191  |            1 |            2 | wall         | sec          |     0.001763 |     0.001763 |     0.001763 |     0.001763 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(42)         |            2 |            2 | wall         | sec          |     1.381135 |     0.690568 |     0.677000 |     0.704135 |     0.019188 |        100.0 |
| >>>   |_fibonacci/(43)         |            2 |            2 | wall         | sec          |     2.196133 |     1.098066 |     1.095881 |     1.100252 |     0.003091 |        100.0 |
| >>>   |_scatter_gatther_75212  |            1 |            2 | wall         | sec          |     0.001775 |     0.001775 |     0.001775 |     0.001775 |     0.000000 |        100.0 |
| >>>   |_fibonacci/(44)         |            2 |            2 | wall         | sec          |     3.542838 |     1.771419 |     1.771036 |     1.771802 |     0.000542 |        100.0 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[peak_rss]|0> Outputting 'timemory-ex-optional-on-output/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-ex-optional-on-output/peak_rss.txt'...
Opening 'timemory-ex-optional-on-output/peak_rss.jpeg' for output...
Closed 'timemory-ex-optional-on-output/peak_rss.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                        MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF SWAP IS ENABLED                       |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|             LABEL              |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> main@ex_optional.cpp:105   |          1 |          0 | peak_rss   | MB         |      2.396 |      2.396 |      2.396 |      2.396 |      0.000 |        0.0 |
| >>> |_main@ex_optional.cpp:106 |          1 |          1 | peak_rss   | MB         |      2.396 |      2.396 |      2.396 |      2.396 |      0.000 |       12.4 |
| >>>   |_fibonacci/(1)          |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(2)          |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(3)          |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(4)          |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(11)         |          5 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(12)         |          5 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(13)         |          5 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(14)         |          5 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_fibonacci/(15)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(16)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(17)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(18)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(19)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(20)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(21)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(22)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(23)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(24)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_fibonacci/(25)         |          5 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(26)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(27)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(28)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(29)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(10)         |          4 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_201    |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_202    |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_203    |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(5)          |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_scatter_gatther_205    |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(6)          |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_208    |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(7)          |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_213    |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(8)          |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_221    |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(9)          |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_234    |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_255    |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_scatter_gatther_289    |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_344    |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_433    |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_577    |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_810    |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_1187   |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_1797   |          1 |          2 | peak_rss   | MB         |      0.780 |      0.780 |      0.780 |      0.780 |      0.000 |      100.0 |
| >>>   |_scatter_gatther_2784   |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_4381   |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_6965   |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_scatter_gatther_11146  |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_17911  |          2 |          2 | peak_rss   | MB         |      0.264 |      0.132 |      0.000 |      0.264 |      0.187 |      100.0 |
| >>>   |_scatter_gatther_28857  |          1 |          2 | peak_rss   | MB         |      0.112 |      0.112 |      0.112 |      0.112 |      0.000 |      100.0 |
| >>>   |_scatter_gatther_46568  |          3 |          2 | peak_rss   | MB         |      0.264 |      0.088 |      0.000 |      0.264 |      0.152 |      100.0 |
| >>>   |_scatter_gatther_200    |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_64279  |          1 |          2 | peak_rss   | MB         |      0.528 |      0.528 |      0.528 |      0.528 |      0.000 |      100.0 |
| >>>   |_fibonacci/(30)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(31)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_71044  |          1 |          2 | peak_rss   | MB         |      0.152 |      0.152 |      0.152 |      0.152 |      0.000 |      100.0 |
| >>>   |_fibonacci/(32)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_fibonacci/(33)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_73628  |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(34)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(35)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_74615  |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(36)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(37)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_74992  |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(38)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(39)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_scatter_gatther_75136  |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(40)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(41)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_75191  |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(42)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(43)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_75212  |          1 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(44)         |          2 |          2 | peak_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|

[page_rss]|0> Outputting 'timemory-ex-optional-on-output/page_rss.json'...
[page_rss]|0> Outputting 'timemory-ex-optional-on-output/page_rss.txt'...
Opening 'timemory-ex-optional-on-output/page_rss.jpeg' for output...
Closed 'timemory-ex-optional-on-output/page_rss.jpeg'...

|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                         AMOUNT OF MEMORY ALLOCATED IN PAGES OF MEMORY. UNLIKE PEAK_RSS, VALUE WILL FLUCTUATE AS MEMORY IS FREED/ALLOCATED                        |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|             LABEL              |   COUNT    |   DEPTH    |   METRIC   |   UNITS    |    SUM     |    MEAN    |    MIN     |    MAX     |   STDDEV   |   % SELF   |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>> main@ex_optional.cpp:105   |          1 |          0 | page_rss   | MB         |      1.335 |      1.335 |      1.335 |      1.335 |      0.000 |        0.0 |
| >>> |_main@ex_optional.cpp:106 |          1 |          1 | page_rss   | MB         |      1.335 |      1.335 |      1.335 |      1.335 |      0.000 |       19.6 |
| >>>   |_fibonacci/(1)          |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(2)          |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>     |_fibonacci/(29)       |          1 |          3 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(3)          |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(4)          |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(11)         |          5 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(12)         |          5 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(13)         |          5 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_fibonacci/(14)         |          5 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(15)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(16)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(17)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(18)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(19)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(20)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(21)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(22)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(23)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_fibonacci/(24)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(25)         |          5 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(26)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(27)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(28)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(29)         |          3 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(10)         |          4 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_201    |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_202    |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_203    |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_fibonacci/(5)          |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_205    |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(6)          |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_208    |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(7)          |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_213    |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(8)          |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_221    |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(9)          |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_234    |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_scatter_gatther_255    |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_289    |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_344    |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_433    |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_577    |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_810    |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_1187   |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_1797   |          1 |          2 | page_rss   | MB         |      0.799 |      0.799 |      0.799 |      0.799 |      0.000 |      100.0 |
| >>>   |_scatter_gatther_2784   |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_4381   |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_scatter_gatther_6965   |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_11146  |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_17911  |          2 |          2 | page_rss   | MB         |      0.119 |      0.059 |      0.000 |      0.119 |      0.084 |      100.0 |
| >>>   |_scatter_gatther_28857  |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_46568  |          3 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_200    |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_64279  |          1 |          2 | page_rss   | MB         |      0.156 |      0.156 |      0.156 |      0.156 |      0.000 |      100.0 |
| >>>   |_fibonacci/(30)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(31)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_71044  |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_fibonacci/(32)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(33)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_73628  |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(34)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(35)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_74615  |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(36)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(37)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_74992  |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(38)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| >>>   |_fibonacci/(39)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_75136  |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(40)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(41)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_75191  |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(42)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(43)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_scatter_gatther_75212  |          1 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
| >>>   |_fibonacci/(44)         |          2 |          2 | page_rss   | MB         |      0.000 |      0.000 |      0.000 |      0.000 |      0.000 |        0.0 |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|


[metadata::manager::finalize]> Outputting 'timemory-ex-optional-on-output/metadata.json'...
>>>  main@ex_optional.cpp:106 :  [laps: 2]
>>>  main@ex_optional.cpp:105 :  [laps: 2]
```

### timemory-off

```bash
#----------------- TIMEMORY is disabled ----------------#

fibonacci(1) = 1
fibonacci(13) = 233
fibonacci(9) = 34
fibonacci(14) = 377
fibonacci(10) = 55
fibonacci(15) = 610
fibonacci(5) = 5
fibonacci(16) = 987
fibonacci(21) = 10946
fibonacci(11) = 89
fibonacci(17) = 1597
fibonacci(12) = 144
fibonacci(6) = 8
fibonacci(7) = 13
fibonacci(8) = 21
fibonacci(18) = 2584
fibonacci(2) = 1
fibonacci(3) = 2
fibonacci(22) = 17711
fibonacci(19) = 4181
fibonacci(4) = 3
fibonacci(20) = 6765
fibonacci(25) = 75025
fibonacci(23) = 28657
fibonacci(24) = 46368
fibonacci(26) = 121393
fibonacci(27) = 196418
fibonacci(29) = 514229
fibonacci(28) = 317811
fibonacci(30) = 832040
fibonacci(31) = 1346269
fibonacci(33) = 3524578
fibonacci(32) = 2178309
fibonacci(34) = 5702887
fibonacci(36) = 14930352
fibonacci(35) = 9227465
fibonacci(37) = 24157817
fibonacci(39) = 63245986
fibonacci(38) = 39088169
fibonacci(40) = 102334155
fibonacci(42) = 267914296
fibonacci(41) = 165580141
fibonacci(43) = 433494437
fibonacci(44) = 701408733
fibonacci(11) = 89
fibonacci(12) = 144
fibonacci(13) = 233
fibonacci(14) = 377
fibonacci(15) = 610
fibonacci(16) = 987
fibonacci(17) = 1597
fibonacci(18) = 2584
fibonacci(19) = 4181
fibonacci(20) = 6765
fibonacci(21) = 10946
fibonacci(22) = 17711
fibonacci(23) = 28657
fibonacci(24) = 46368
fibonacci(25) = 75025
fibonacci(26) = 121393
fibonacci(27) = 196418
fibonacci(28) = 317811
fibonacci(29) = 514229
fibonacci(10) = 55
fibonacci(11) = 89
fibonacci(12) = 144
fibonacci(13) = 233
fibonacci(14) = 377
fibonacci(15) = 610
fibonacci(16) = 987
fibonacci(17) = 1597
fibonacci(18) = 2584
fibonacci(19) = 4181
fibonacci(20) = 6765
fibonacci(21) = 10946
fibonacci(22) = 17711
fibonacci(23) = 28657
fibonacci(24) = 46368
fibonacci(25) = 75025
fibonacci(26) = 121393
fibonacci(27) = 196418
fibonacci(28) = 317811
fibonacci(29) = 514229
fibonacci(10) = 55
fibonacci(11) = 89
fibonacci(12) = 144
fibonacci(13) = 233
fibonacci(14) = 377

#----------------- TIMEMORY is disabled ----------------#
```