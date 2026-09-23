# Examples

Examples demonstrating the usage of timemory features, components, dynamic instrumentation, python bindings and profiling support.

## Build

See [examples](../README.md##Build). To build examples along with timemory build using cmake, enable the cmake `-DTIMEMORY_BUILD_EXAMPLES=ON`. If using spack based install, use the `+examples` flag in the spack install command.

## Description

The examples demonstrate the use of timemory components and features in both C/C++ and Python. Several basic, intermediate and advanced level examples demonstrating basic instrumentation to advanced features such as MPI, kokkos, likwid, caliper, dyninst support are presented. The subsequent sections provide an overview of each example functionality.

### [ex-c](ex-c/README.md)

Demonstrates the basic use of timemory timing instrumentation (timers) in C.

### [ex-caliper](ex-caliper/README.md)

Demonstrates an example of instrumentation using caliper markers.

### [ex-cpu-roofline](ex-cpu-roofline/README.md)

Demonstrates an example of executing a set of benchmarks on CPU for roofline analysis and plotting.

### [ex-cuda-event](ex-cuda-event/README.md)

Demonstrates an example of measuring CPU and CPU events such as cpu-malloc, gpu-malloc, assign and so on using cuda events, cupti counters and events and auto tuple of timemory components.

### [ex-custom-dynamic-instr](ex-custom-dynamic-instr/README.md)

Demonstrates an example of implementing a custom instrumentation library which can be used for dynamic instrumentation using `timemory-run` tool. For documentation on `timemory-run`, dynamic instrumentation, refer to [timemory-run](../source/tools/timemory-run/README.md).

### [ex-cxx-basic](ex-cxx-basic/README.md)

Demonstrates an example of basic timemory instrumentation in C++.

### [ex-cxx-overhead](ex-cxx-overhead/README.md)

Demonstrates an example of quanitfication of instrumentation overhead (both time and memory) of timemory.

### [ex-cxx-tuple](ex-cxx-tuple/README.md)

Demonstrates an example of usage of auto tuple, component tuple and papi tuple for performance measurements.

### [ex-derived](ex-derived/README.md)

Demonstrates an example where a custom (derived) component is created from existing timemory components and then used in auto tuple and auto list for performance measurement.

### [ex-ert](ex-ert/README.md)

Demonstrates an example of usage of the empirical roofline toolkit (ERT) component for running microkernels and generation of performance data for roofline analysis with and without using Kokkos.

### [ex-gotcha](ex-gotcha/README.md)

Demonstrates examples of wrapping the C standard library function `puts` and MPI function calls using GOTCHA and instrumenting them using timemory.

### [ex-gpu-roofline](ex-gpu-roofline/README.md)

Demonstrates an example of executing a set of benchmarks on GPU for roofline analysis and plotting (requires CUDA and CUPTI support).

### [ex-likwid](ex-likwid/README.md)

Demonstrates an example of performance measurement using likwid markers in C++ and Python.

### [ex-minimal](ex-minimal/README.md)

Demonstrates an example of basic timemory timing measurement, usage of timemory library, and timemory library overload.

### [ex-optional](ex-optional/README.md)

Demonstrates examples where the timemory is optionally enabled/disabled in either normal of MPI based computations.

### [ex-python](ex-python/README.md)

Demonstrate examples of timemory usage in Python such as profiling, component bundles, MPI instrumentation, python bindings, and generic instrumentation.

### [ex-statistics](ex-statistics/README.md)

Demonstrates an example of generation of flat measurements and statistics from component measurements by setting appropriate component traits.