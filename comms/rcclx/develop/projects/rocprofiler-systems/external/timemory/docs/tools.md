# Tools

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 1

   tools/timemory-timem/README
   tools/timemory-avail/README
   tools/timemory-run/README
   tools/timemory-stubs/README
   tools/timemory-jump/README
   tools/timemory-mpip/README
   tools/timemory-ncclp/README
   tools/timemory-mallocp/README
   tools/timemory-ompt/README
   tools/timemory-nvml/README
   tools/timemory-compiler-instrument/README
   tools/kokkos-connector/README
```

This section covers the executables and libraries that are distributed as part of the library.

- Executables
    - [timem](tools/timemory-timem/README.md)
        - Drop-in replacement for `time` (UNIX)
        - For MPI applications, use `timem-mpi`
    - [timemory-avail](tools/timemory-avail/README.md)
        - Use this executable to query available components, available settings, and available hardware counters
    - [timemory-run](tools/timemory-run/README.md)
        - Use this executable (Linux-only) for dynamic instrumentation
    - [timemory-nvml](tools/timemory-nvml/README.md)
        - Data collection similar to `nvidia-smi`
- Libraries
    - [timemory-stubs](tools/timemory-stubs/README.md)
        - Provides timemory instrumentation stubs for dynamic library preloading
    - [timemory-jump](tools/timemory-jump/README.md)
        - Provides timemory instrumentation via `dlopen` and `dlsym`
    - [timemory-mpip](tools/timemory-mpip/README.md)
        - Provide MPI profiling via GOTCHA
    - [timemory-ncclp](tools/timemory-ncclp/README.md)
        - Provide NCCL profiling via GOTCHA
    - [timemory-mallocp](tools/timemory-mallocp/README.md)
        - Records amount of memory allocated and freed on the CPU and GPU via GOTCHA wrappers around malloc, free, cudaMalloc, etc.
    - [timemory-ompt](tools/timemory-ompt/README.md)
        - Provide OpenMP profiling via OMPT (OpenMP Tools)
    - [timemory-compiler-instrument](tools/timemory-compiler-instrument/README.md)
        - Automatically instrument C and C++ source code via `-finstrument-functions` compiler flag
    - [Kokkos Connectors](tools/kokkos-connector/README.md)
        - Libraries for Kokkos profiling

## Profiling with timemory

### Instrumenting Existing Binary

Timemory includes the [timemory-run](tools/timemory-run/README.md) as a full profiler for Linux systems.
This executable supports dynamic instrumentation (instrumenting at the target applicaiton's runtime), attaching
to a running process, and binary re-writing (creating a new instrumented binary). The instrumented applications
support flat-profiling, call-stack profiling, and timeline profiling and can be configured to use any of the
components timemory provides or, with a little work, can also be used to instrument custom components defined by the user. It is highly recommended for custom tools targetting specific functions to use the combination of
GOTCHA and the dynamic instrumentation. Using the GOTCHA extensions for
profiling specific functions enables creating components which replace the function or audit the
incoming arguments and return values for the functions and the dynamic instrumentation makes it
easy to inject using the GOTCHA wrappers into an executable or library.

### Instrumentation Binary at Compile-Time

Timemory includes support for compile-time instrumentation via the
[timemory-compiler-instrument](tools/timemory-compiler-instrument/README.md) library.
This form of instrumentation is available on UNIX systems and compilers which support the `-finstrument-functions`
compiler flag. It is generally recommended that the compiler instrumentation be propagated all the way to the
compilation of the executable (as opposed to only the libraries called by the executable) since finalization is
typically triggered when the executable returns from `main`. Failure to insert instrumentation
around `main` may or may not result in segmentation faults.
