# ex-compiler-instrument

These examples demonstrate the usage of timemory compiler instrumentation.

## Build

See [examples](../README.md##Build). These examples build the following corresponding binaries: `ex_compiler_instrument`.

## Expected Console Output

```console
$ ./ex_cxx_minimal
[12301]> timemory-compiler-instrument will close after 'main' returns
[12301]> timemory-compiler-instrument: 47 results
[wall]|0> Outputting 'timemory-compiler-instrumentation-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-compiler-instrumentation-output/wall.json'...
[wall]|0> Outputting 'timemory-compiler-instrumentation-output/wall.tree.json'...
[wall]|0> Outputting 'timemory-compiler-instrumentation-output/wall.txt'...

[12301]> timemory-compiler-instrument: finalizing...
```

## Expected File Output

```console
$ cat timemory-compiler-instrumentation-output/wall.txt
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                      REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                          LABEL                                                           | COUNT  | DEPTH  | METRIC | UNITS  |  SUM   | MEAN   |  MIN   |  MAX   | STDDEV | % SELF |
|--------------------------------------------------------------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| >>> main                                                                                                                 |      1 |      0 | wall   | sec    |  1.548 |  1.548 |  1.548 |  1.548 |  0.000 |    0.0 |
| >>> |_fibonacci(long)                                                                                                    |     10 |      1 | wall   | sec    |  0.096 |  0.010 |  0.009 |  0.010 |  0.000 |    0.4 |
| >>>   |_fibonacci(long)                                                                                                  |     20 |      2 | wall   | sec    |  0.096 |  0.005 |  0.003 |  0.006 |  0.001 |    0.8 |
| >>>     |_fibonacci(long)                                                                                                |     40 |      3 | wall   | sec    |  0.095 |  0.002 |  0.001 |  0.004 |  0.001 |    1.6 |
| >>>       |_fibonacci(long)                                                                                              |     80 |      4 | wall   | sec    |  0.094 |  0.001 |  0.000 |  0.003 |  0.001 |    3.1 |
| >>>         |_fibonacci(long)                                                                                            |    160 |      5 | wall   | sec    |  0.091 |  0.001 |  0.000 |  0.002 |  0.000 |    6.6 |
| >>>           |_fibonacci(long)                                                                                          |    320 |      6 | wall   | sec    |  0.085 |  0.000 |  0.000 |  0.001 |  0.000 |   14.0 |
| >>>             |_fibonacci(long)                                                                                        |    640 |      7 | wall   | sec    |  0.073 |  0.000 |  0.000 |  0.001 |  0.000 |   30.1 |
| >>>               |_fibonacci(long)                                                                                      |   1140 |      8 | wall   | sec    |  0.051 |  0.000 |  0.000 |  0.001 |  0.000 |   52.6 |
| >>>                 |_fibonacci(long)                                                                                    |   1280 |      9 | wall   | sec    |  0.024 |  0.000 |  0.000 |  0.000 |  0.000 |   71.3 |
| >>>                   |_fibonacci(long)                                                                                  |    740 |     10 | wall   | sec    |  0.007 |  0.000 |  0.000 |  0.000 |  0.000 |   82.5 |
| >>>                     |_fibonacci(long)                                                                                |    200 |     11 | wall   | sec    |  0.001 |  0.000 |  0.000 |  0.000 |  0.000 |   93.9 |
| >>>                       |_fibonacci(long)                                                                              |     20 |     12 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>> |_consume(long)                                                                                                      |     10 |      1 | wall   | sec    |  1.452 |  0.145 |  0.145 |  0.145 |  0.000 |   13.2 |
| >>>   |_std::__1::mutex::mutex()                                                                                         |     10 |      2 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   85.0 |
| >>>     |_std::__1::mutex::mutex()                                                                                       |     10 |      3 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::unique_lock<std::__1::mutex>::unique_lock(std::__1::mutex&)                                            |     10 |      2 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   44.7 |
| >>>     |_std::__1::unique_lock<std::__1::mutex>::unique_lock(std::__1::mutex&)                                          |     10 |      3 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   87.0 |
| >>>       |_std::__1::mutex* std::__1::addressof<std::__1::mutex>(std::__1::mutex&)                                      |     10 |      4 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::unique_lock<std::__1::mutex>::unique_lock(std::__1::mutex&, std::__1::defer_lock_t)                    |     10 |      2 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   45.8 |
| >>>     |_std::__1::unique_lock<std::__1::mutex>::unique_lock(std::__1::mutex&, std::__1::defer_lock_t)                  |     10 |      3 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   86.4 |
| >>>       |_std::__1::mutex* std::__1::addressof<std::__1::mutex>(std::__1::mutex&)                                      |     10 |      4 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000l> >::duration<long>(long const&, std::__1::ena... |   2540 |      2 | wall   | sec    |  0.062 |  0.000 |  0.000 |  0.000 |  0.000 |   86.8 |
| >>>     |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000l> >::duration<long>(long const&, std::__1::e... |   2540 |      3 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::chrono::time_point<std::__1::chrono::steady_clock, std::__1::common_type<std::__1::chrono::duration... |   2540 |      2 | wall   | sec    |  0.862 |  0.000 |  0.000 |  0.001 |  0.000 |   16.9 |
| >>>     |_std::__1::chrono::time_point<std::__1::chrono::steady_clock, std::__1::chrono::duration<long long, std::__1... |   2540 |      3 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>     |_std::__1::common_type<std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >, std::__1::c... |   2540 |      3 | wall   | sec    |  0.648 |  0.000 |  0.000 |  0.001 |  0.000 |   30.1 |
| >>>       |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >::count() const                      |   5080 |      4 | wall   | sec    |  0.017 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>       |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >::duration<long long, std::__1::r... |   2540 |      4 | wall   | sec    |  0.374 |  0.000 |  0.000 |  0.000 |  0.000 |   13.3 |
| >>>         |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >::duration<long long, std::__1:... |   2540 |      5 | wall   | sec    |  0.325 |  0.000 |  0.000 |  0.000 |  0.000 |   30.5 |
| >>>           |_std::__1::enable_if<__is_duration<std::__1::chrono::duration<long long, std::__1::ratio<1l, 100000000... |   2540 |      6 | wall   | sec    |  0.218 |  0.000 |  0.000 |  0.000 |  0.000 |   22.6 |
| >>>             |_std::__1::chrono::__duration_cast<std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000l> ... |   2540 |      7 | wall   | sec    |  0.168 |  0.000 |  0.000 |  0.000 |  0.000 |   58.6 |
| >>>               |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000l> >::count() const                    |   2540 |      8 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>               |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >::duration<long long>(lon... |   2540 |      8 | wall   | sec    |  0.062 |  0.000 |  0.000 |  0.000 |  0.000 |   86.1 |
| >>>                 |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >::duration<long long>(l... |   2540 |      9 | wall   | sec    |  0.009 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>           |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >::count() const                  |   2540 |      6 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>       |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >::duration<long long>(long long c... |   2540 |      4 | wall   | sec    |  0.062 |  0.000 |  0.000 |  0.000 |  0.000 |   86.4 |
| >>>         |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >::duration<long long>(long long... |   2540 |      5 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>     |_std::__1::chrono::time_point<std::__1::chrono::steady_clock, std::__1::chrono::duration<long long, std::__1... |   2540 |      3 | wall   | sec    |  0.060 |  0.000 |  0.000 |  0.000 |  0.000 |   87.1 |
| >>>       |_std::__1::chrono::time_point<std::__1::chrono::steady_clock, std::__1::chrono::duration<long long, std::_... |   2540 |      4 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_bool std::__1::chrono::operator<<std::__1::chrono::steady_clock, std::__1::chrono::duration<long long, std::_... |   2540 |      2 | wall   | sec    |  0.326 |  0.000 |  0.000 |  0.000 |  0.000 |   45.0 |
| >>>     |_std::__1::chrono::time_point<std::__1::chrono::steady_clock, std::__1::chrono::duration<long long, std::__1... |   5080 |      3 | wall   | sec    |  0.016 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>     |_bool std::__1::chrono::operator<<long long, std::__1::ratio<1l, 1000000000l>, long long, std::__1::ratio<1l... |   2540 |      3 | wall   | sec    |  0.164 |  0.000 |  0.000 |  0.000 |  0.000 |   29.6 |
| >>>       |_std::__1::chrono::__duration_lt<std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >,... |   2540 |      4 | wall   | sec    |  0.115 |  0.000 |  0.000 |  0.000 |  0.000 |   86.7 |
| >>>         |_std::__1::chrono::duration<long long, std::__1::ratio<1l, 1000000000l> >::count() const                    |   5080 |      5 | wall   | sec    |  0.015 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::unique_lock<std::__1::mutex>::try_lock()                                                               |   2530 |      2 | wall   | sec    |  0.008 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
| >>>   |_std::__1::unique_lock<std::__1::mutex>::~unique_lock()                                                           |     20 |      2 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |   85.6 |
| >>>     |_std::__1::unique_lock<std::__1::mutex>::~unique_lock()                                                         |     20 |      3 | wall   | sec    |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  100.0 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
```
