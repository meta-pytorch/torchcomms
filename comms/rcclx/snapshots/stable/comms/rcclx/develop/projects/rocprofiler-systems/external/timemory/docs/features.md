# Features

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 2
```

## Cross-Language Support

- Variadic interface to all the utilities from C code
- Variadic interface to all the utilities from C++ code
- Variadic interface to all the utilities from Python code
  - Includes context-managers and decorators

It is very common for Python projects to implement expensive routines in C or C++. Implementing a timemory auto-tuple in any combination
of these languages will produce one combined report for all the languages (provided each language links to the same library).
However, this is a feature of timemory. Timemory can be used in standalone C, C++, or Python projects.

## Reusable Components

Performance analysis tools provide a wide variety of routines and capabilities which can be useful to nearly
every application in existence. However, nearly all these tools hide the vast majority of their capabilities
and only provide a very limited API for users to build upon. Worse still, some of these tools only provide APIs
which are generally useless outside of the context of their tool, e.g. the
[Instrumentation and Tracing Technology API (ittnotify)](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/api-support/instrumentation-and-tracing-technology-apis/instrumentation-and-tracing-technology-api-reference/memory-allocation-apis.html)
provided by Intel is a misnomer for an API which essentially provides the same functionality as the appropriately named
[NVIDIA Tools Extension SDK (NVTX)](https://github.com/NVIDIA/NVTX).

Every measurement capability provided by timemory is provided as a stand-alone encapsulation
of how to perform a single individual measurement which can be reused without any added overhead.
For example, the `wall_clock` component is an encapsulation of recording the delta between
two timepoints which cannot be improved by a custom implementation performing the same calculation
because it maps directly to `start()` and `stop()` member functions which maps to:

```cpp
struct timespec ts;
int64_t value = 0, accum = 0;

// get the starting timestamp
clock_gettime(clock_id, &ts);

// convert to nanoseconds
value = (ts.tv_sec * std::nano::den + ts.tv_nsec);

// get the ending timestamp
clock_gettime(clock_id, &ts);

// convert and accumulate the difference
accum += (ts.tv_sec * std::nano::den + ts.tv_nsec) - value;
```

The same goes for the component for collecting CPU and/or GPU hardware counters via the
PAPI API -- the start and stop member functions are just encapsulate two functions calls to
the PAPI interface and accumulate the difference. There is no hidden overhead or data
storage, these components can used individually without interacting with anything else
that timemory provides. **This feature has a very important implication:
timemory components are composable**. If you want the rate of any component or the
ratio of two component measurements, one simply has to create a new component which
performs the desired calculation and that component will have all the capabilities and
composability as the original component(s).

Furthermore, these stand-alone tools are not limited to C++. Through the use of the
excellent C++ template library for generating Python bindings, the vast majority
of these components are also available as
**highly-efficient stand-alone tools for Python codes**:

```python
from timemory.component import WallClock

# encapsulation of how to take a wall-clock measurement
wc = WallClock()

# get the starting timestamp
wc.start()

# get the ending timestamp and the accumulate the difference
wc.stop()

# get the raw value of the last start/stop
wc.get_value()

# get the raw value of the accumulation
wc.get_accum()

# print the value in the globally configured units
print("Elapsed time: {} {}".format(wc.get(), wc.display_units()))
```

## Multithreading

The multithreading overhead is essentially zero.
All timemory components use static thread-local singletons of call-graps that are automatically created when a
new thread starts recording a component. The state of the singleton on the master thread is bookmarked and when the
thread is destroyed, the thread-local call-graph is merged back into the master call-graph. Only during the
one-time merge into the master call-graph is a synchronization lock (mutex) utilized.

## Distributed Memory Parallelism

If a project uses MPI or UPC++, timemory components will automatically provide support for combining this data
into a single output, support per-rank output, or.

## Hardware Counters

CPU hardware counters are available via PAPI components.
GPU hardware counters are available via the `cupti_counters` component.

## Plot Generation in Python

The results from timemory can be serialized to JSON and the JSON output can be used to produce performance plots
via the standalone `timemory-plotter` or `timemory.plotting` Python module. For roofline analysis,
the `timemory.roofline` module can be used.

## Pandas Dataframe Analysis in Python

Timemory supports conversion to pandas dataframe via a Python package called [hatchet](https://github.com/LLNL/hatchet).
Hatchet requires timemory's hierarchical JSON format (file with extension `.tree.json` or `timemory.get(hierarchy=True)` in Python).

## Command-line Tools

The command-line tools `timem` and `pytimem` work like the `time` executable but with more information.
UNIX systems provide `timem` executable that works like `time`. On all systems, `pytimem` is provided.

```shell
$ ./timem sleep 2

> [sleep] total execution time :
        2.119142 sec real
        0.000000 sec user
        0.000000 sec sys
        0.000000 sec cpu
        0.000000 % cpu_util
        0.745472 MB peak_rss
               0 io_in
               0 io_out
             389 minor_page_flts
               0 major_page_flts
               0 num_signals
              46 vol_cxt_swch
              10 prio_cxt_swch
```

## Signal Detection

Timemory provides a facility for catching signals and printing out a backtrace when the signals are raised:

```cpp
tim::signals::enable_signal_detection({ SIGHUP, SIGINT, SIGQUIT, SIGABRT });
// ...
tim::signals::disable_signal_detection();
```

## Cache Information

Timemory provides method on Linux, Windows, and macOS to query the size of L1, L2, and L3 cache.
A `get_max()` function is provided for convenience as some systems (e.g. KNL) do not have an L3 cache.

> Namespace: `tim::ert::cache_size`

| Cache level | Function(s)                                                      |
| ----------- | ---------------------------------------------------------------- |
| L1          | `tim::ert::cache_size::get(1)`, `tim::ert::cache_size::get<1>()` |
| L2          | `tim::ert::cache_size::get(2)`, `tim::ert::cache_size::get<2>()` |
| L3          | `tim::ert::cache_size::get(3)`, `tim::ert::cache_size::get<3>()` |
| max         | `tim::ert::cache_size::get_max()`                                |

## Support for Multiple Third-Party Instrumentation APIs

- [LIKWID](https://github.com/RRZE-HPC/likwid)
- [Caliper](https://github.com/LLNL/Caliper)
- [TAU](https://www.cs.uoregon.edu/research/tau/home.php)
- [gperftools](https://github.com/gperftools/gperftools)
- MPI
- OpenMP
- CrayPAT
- Allinea-MAP
- PAPI
- ittnotify (Intel Parallel Studio API)
- CUPTI (NVIDIA performance API)
- NVTX (NVIDIA marker API)

## Generic Bundling of Multiple Tools

- CPU and GPU hardware counters via PAPI
- NVIDIA GPU hardware counters via CUPTI
- NVIDIA GPU tracing via CUPTI
- Generating a Roofline for performance-critical sections on the CPU and NVIDIA GPUs
  - Classical Roofline (FLOPs)
  - Instruction Roofline
- Memory usage
- Tool insertion around `malloc`, `calloc`, `free`, `cudaMalloc`, `cudaFree`
- Wall-clock, cpu-clock, system-clock timing
- Number of bytes read/written to file-system (and rate)
- Number of context switches
- Trip counts
- CUDA kernel runtime(s)
- Data value tracking

## Powerful GOTCHA Extensions

- [GOTCHA](https://github.com/LLNL/GOTCHA) is an API wrapping function calls similar to the use of LD_PRELOAD
    - Significantly simplify existing implementations
- Scoped GOTCHA
    - Enables temporary wrapping over regions
- Use gotcha component to replace external function calls with custom replacements
    - E.g. replace the C math function `exp` with custom `exp` implementation
- Use gotcha component to wrap external library calls with custom instrumentation
